"""Download and cache NYC TLC data files (idempotent).

Sources (public TLC mirror on CloudFront):

- High Volume FHV (Uber/Lyft/Via) trip records, one Parquet file per month:
    https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{YYYY-MM}.parquet
- Taxi zone lookup table (zone id -> borough / zone name / service zone):
    https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv

Files are cached under ``data/raw/``. Downloads are idempotent: a cached file
is reused when it exists and passes a sanity check (valid Parquet footer for
trip files, minimum size for the CSV). Partial downloads are written to a
``.part`` sibling and atomically renamed only on success, so an interrupted
download never leaves a corrupt cache entry behind.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from rider_sim.config import RAW_DIR

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
TRIP_TEMPLATE = f"{BASE_URL}/trip-data/fhvhv_tripdata_{{month}}.parquet"
_ZONE_URLS = (
    f"{BASE_URL}/misc/taxi+_zone_lookup.csv",
    f"{BASE_URL}/misc/taxi_zone_lookup.csv",
)

_MIN_ZONE_CSV_BYTES = 5_000
_CHUNK_SIZE = 1 << 20

console = Console()


def trips_path(month: str) -> Path:
    return RAW_DIR / f"fhvhv_tripdata_{month}.parquet"


def zones_path() -> Path:
    return RAW_DIR / "taxi_zone_lookup.csv"


def _copy(url: str, dest: Path, label: str) -> None:
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed TLC host
            total = int(resp.headers.get("Content-Length") or 0)
            columns = [
                TextColumn(f"[bold]{label}[/bold]"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ]
            with Progress(*columns, console=console, transient=True) as progress:
                task = progress.add_task("download", total=total or None)
                with tmp.open("wb") as out:
                    while chunk := resp.read(_CHUNK_SIZE):
                        out.write(chunk)
                        progress.advance(task, len(chunk))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def fetch_trips(month: str) -> Path:
    """Download (if needed) the HVFHV trip Parquet for ``month`` (YYYY-MM)."""
    dest = trips_path(month)
    if dest.exists():
        try:
            n_rows = pq.ParquetFile(dest).metadata.num_rows
        except Exception as exc:
            dest.unlink()
            raise ValueError(
                f"Cached file {dest} is corrupt; removed it. Re-run to re-download."
            ) from exc
        console.log(f"[green]{dest.name}[/green] already cached ({n_rows:,} rows).")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = TRIP_TEMPLATE.format(month=month)
    console.log(f"Downloading HVFHV trip records for [bold]{month}[/bold] ...")
    _copy(url, dest, dest.name)
    try:
        n_rows = pq.ParquetFile(dest).metadata.num_rows
    except Exception as exc:
        dest.unlink()
        raise ValueError(f"Downloaded file {dest} is not a valid Parquet file.") from exc
    console.log(f"[green]{dest.name}[/green] cached ({n_rows:,} rows, {_mb(dest):.0f} MB).")
    return dest


def fetch_zones() -> Path:
    """Download (if needed) the TLC taxi zone lookup CSV."""
    dest = zones_path()
    if dest.exists() and dest.stat().st_size >= _MIN_ZONE_CSV_BYTES:
        console.log(f"[green]{dest.name}[/green] already cached.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    console.log("Downloading TLC taxi zone lookup table ...")
    last_error: Exception | None = None
    for url in _ZONE_URLS:
        try:
            _copy(url, dest, dest.name)
            if dest.stat().st_size >= _MIN_ZONE_CSV_BYTES:
                console.log(f"[green]{dest.name}[/green] cached.")
                return dest
            dest.unlink(missing_ok=True)
        except Exception as exc:  # try the alternate URL
            last_error = exc
    raise RuntimeError(f"Could not download taxi zone lookup table: {last_error}")


def fetch_all(month: str) -> tuple[Path, Path]:
    """Fetch both the trip Parquet for ``month`` and the zone lookup CSV."""
    return fetch_trips(month), fetch_zones()


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
