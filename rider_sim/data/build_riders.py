"""Build a semi-synthetic rider panel from real NYC TLC HVFHV trip records.

.. important::
    Real TLC trip records contain **no rider identifier** -- every row is an
    anonymous trip. This module therefore does not recover real riders; it
    constructs *pseudo-riders*: sequences of real trips grouped so that each
    pseudo-rider owns 5-40 trips sharing a coherent commuting signature. The
    resulting panel is **semi-synthetic**: every individual trip row (zones,
    times, fares, tips, waits, driver pay) is real, but the attribution of
    trips to a specific "rider" is synthesized. Conclusions about
    individual-level behavior drawn from this panel are modeling artifacts;
    only distributional statements are grounded in the real data.

Construction (deterministic for a given seed):

1. Load one month of HVFHV trips with DuckDB, filtering to usable rows
   (positive miles/time/fare, complete timestamps and zone IDs).
2. Bucket every trip into a *commute cell* on the four clustering dimensions
   (pickup zone, hour-of-day bucket, weekday/weekend, trip-miles bucket).
3. Cluster the cells themselves (KMeans over the scaled cell features) so
   each cell has a fallback neighborhood of similar cells.
4. For each of ``n_riders`` pseudo-riders: draw a home cell with probability
   proportional to that cell's real trip volume, draw a trip count k in
   [5, 40], then sample k real trips without replacement from the home cell,
   spilling into the cell's KMeans neighborhood only when the cell pool is
   exhausted. Each pseudo-rider's trips therefore concentrate in one pickup
   zone with similar hours -- a coherent home zone and commute pattern.
5. Derive real per-trip observables: base_passenger_fare, tips, trip_miles,
   trip_time, request-to-pickup wait seconds, and a computed surge ratio
   ``driver_pay / (trip_miles * median_pay_per_mile)``.

Outputs (``data/processed/``):

- ``trips.parquet`` -- every assigned trip with ``rider_id`` + observables.
- ``riders.parquet`` -- ``rider_id``, home/work zone, trip count, home cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rider_sim.config import PROCESSED_DIR, RIDERS_PATH, TRIPS_PATH
from rider_sim.data.fetch import fetch_trips

console = Console()

MIN_TRIPS = 5
MAX_TRIPS = 40
MAX_WAIT_SECS = 3600

_CELL_COLS = ["pickup_zone", "pickup_hour", "is_weekend", "miles_bucket"]

_TRIP_SQL = """
SELECT
    PULocationID::INTEGER AS pickup_zone,
    DOLocationID::INTEGER AS dropoff_zone,
    EXTRACT(hour FROM pickup_datetime)::INTEGER AS pickup_hour,
    CASE WHEN EXTRACT(dayofweek FROM pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    CASE
        WHEN trip_miles < 1.0 THEN 0
        WHEN trip_miles < 2.0 THEN 1
        WHEN trip_miles < 4.0 THEN 2
        WHEN trip_miles < 8.0 THEN 3
        ELSE 4
    END AS miles_bucket,
    base_passenger_fare,
    tips,
    trip_miles,
    trip_time,
    LEAST(GREATEST(CAST(EPOCH(pickup_datetime) - EPOCH(request_datetime) AS INTEGER), 0), 3600)
        AS wait_secs,
    driver_pay,
    base_passenger_fare + tips AS fare_paid
FROM read_parquet(?)
WHERE trip_miles > 0
  AND trip_time > 0
  AND base_passenger_fare > 0
  AND driver_pay >= 0
  AND PULocationID IS NOT NULL
  AND DOLocationID IS NOT NULL
  AND request_datetime IS NOT NULL
  AND pickup_datetime IS NOT NULL
  AND pickup_datetime >= request_datetime
"""


def load_trips(month: str) -> pd.DataFrame:
    """Load and clean one month of HVFHV trips, with derived observables."""
    trips_file = fetch_trips(month)
    con = duckdb.connect()
    try:
        median_ppm_row = con.execute(
            """
            SELECT median(driver_pay / trip_miles)
            FROM read_parquet(?)
            WHERE trip_miles > 0 AND driver_pay >= 0 AND base_passenger_fare > 0
            """,
            [str(trips_file)],
        ).fetchone()
        df = con.execute(_TRIP_SQL, [str(trips_file)]).df()
    finally:
        con.close()

    median_ppm = float(median_ppm_row[0]) if median_ppm_row else float("nan")
    if not np.isfinite(median_ppm) or median_ppm <= 0:
        raise ValueError(f"Could not compute a valid median pay-per-mile from {trips_file}")
    df["surge_ratio"] = (df["driver_pay"] / (df["trip_miles"] * median_ppm)).clip(0.0, 10.0)
    console.log(
        f"Loaded [bold]{len(df):,}[/bold] usable trips; median driver pay per mile = "
        f"[bold]${median_ppm:.2f}[/bold]."
    )
    return df


def assign_riders(
    df: pd.DataFrame,
    n_riders: int,
    seed: int,
    min_trips: int = MIN_TRIPS,
    max_trips: int = MAX_TRIPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group trips into ``n_riders`` pseudo-riders with coherent commute cells.

    Returns ``(riders, trips)`` DataFrames.
    """
    if len(df) < n_riders * min_trips:
        raise ValueError(
            f"Only {len(df):,} usable trips; need at least {n_riders * min_trips:,} "
            f"for {n_riders} riders with >= {min_trips} trips each."
        )
    rng = np.random.default_rng(seed)

    cell_ids, _ = pd.factorize(df[_CELL_COLS].apply(tuple, axis=1), sort=True)
    df = df.copy()
    df["cell_id"] = cell_ids
    n_cells = int(cell_ids.max()) + 1

    uniq = df[["cell_id", *_CELL_COLS]].drop_duplicates("cell_id").sort_values("cell_id")
    cell_zone = dict(zip(uniq["cell_id"].to_numpy(), uniq["pickup_zone"].to_numpy()))

    n_clusters = max(16, n_riders // 8)
    scaled = StandardScaler().fit_transform(uniq[_CELL_COLS].astype(float))
    clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(scaled)
    cell_to_cluster = dict(zip(uniq["cell_id"].to_numpy(), clusters))
    df["cluster"] = df["cell_id"].map(cell_to_cluster)

    cell_pools = cast(dict[int, np.ndarray], df.groupby("cell_id", sort=True).indices)
    cluster_pools = cast(dict[int, np.ndarray], df.groupby("cluster", sort=True).indices)

    counts = np.bincount(df["cell_id"].to_numpy())
    cell_probs = counts / counts.sum()
    all_idx = np.arange(len(df))
    used = np.zeros(len(df), dtype=bool)

    rider_rows: dict[int, np.ndarray] = {}
    rider_home: dict[int, int] = {}
    rider_cells: dict[int, int] = {}

    for rider_id in range(n_riders):
        cell_id = int(rng.choice(n_cells, p=cell_probs))
        k = int(rng.integers(min_trips, max_trips + 1))
        picked: list[np.ndarray] = []
        need = k
        for pool in (cell_pools[cell_id], cluster_pools[cell_to_cluster[cell_id]], all_idx):
            candidates = pool[~used[pool]]
            if candidates.size >= need:
                picked.append(rng.choice(candidates, size=need, replace=False))
                need = 0
                break
            picked.append(candidates)
            need -= candidates.size
        if need > 0:  # only reachable if the data were absurdly small
            free = all_idx[~used]
            picked.append(rng.choice(free, size=need, replace=len(free) < need))
        idx = np.concatenate(picked)
        used[idx] = True
        rider_rows[rider_id] = idx
        rider_home[rider_id] = int(cast(int, cell_zone[cell_id]))
        rider_cells[rider_id] = cell_id

    trip_parts: list[pd.DataFrame] = []
    rider_parts: list[dict[str, int]] = []
    for rider_id, idx in rider_rows.items():
        sub = df.iloc[idx].copy()
        sub["rider_id"] = rider_id
        trip_parts.append(sub)
        home_cell = uniq.iloc[rider_cells[rider_id]]
        work_zone = int(cast(int, sub["dropoff_zone"].value_counts().sort_index().idxmax()))
        rider_parts.append(
            {
                "rider_id": rider_id,
                "home_zone": rider_home[rider_id],
                "work_zone": work_zone,
                "observed_trip_count": int(len(idx)),
                "home_pickup_hour": int(home_cell["pickup_hour"]),
                "home_is_weekend": int(home_cell["is_weekend"]),
                "home_miles_bucket": int(home_cell["miles_bucket"]),
            }
        )

    trips = pd.concat(trip_parts, ignore_index=True)
    trips = trips.drop(columns=["cell_id", "cluster"])
    col_order = [
        "rider_id",
        "pickup_zone",
        "dropoff_zone",
        "pickup_hour",
        "is_weekend",
        "miles_bucket",
        "base_passenger_fare",
        "tips",
        "trip_miles",
        "trip_time",
        "wait_secs",
        "driver_pay",
        "fare_paid",
        "surge_ratio",
    ]
    trips = trips[col_order]
    riders = pd.DataFrame(rider_parts)
    return riders, trips


def build_riders(
    month: str,
    n_riders: int,
    seed: int,
    min_trips: int = MIN_TRIPS,
    max_trips: int = MAX_TRIPS,
    riders_path: Path = RIDERS_PATH,
    trips_path: Path = TRIPS_PATH,
) -> tuple[Path, Path]:
    """Build and persist the pseudo-rider panel and its trips."""
    df = load_trips(month)
    riders, trips = assign_riders(df, n_riders=n_riders, seed=seed,
                                  min_trips=min_trips, max_trips=max_trips)

    riders_path.parent.mkdir(parents=True, exist_ok=True)
    trips_path.parent.mkdir(parents=True, exist_ok=True)
    riders.to_parquet(riders_path, index=False)
    trips.to_parquet(trips_path, index=False)

    table = Table(title="Pseudo-rider panel")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("riders", f"{len(riders):,}")
    table.add_row("trips assigned", f"{len(trips):,}")
    table.add_row("trips per rider (min/med/max)", (
        f"{riders.observed_trip_count.min()}/{riders.observed_trip_count.median():.0f}"
        f"/{riders.observed_trip_count.max()}"
    ))
    table.add_row("unique home zones", f"{riders.home_zone.nunique():,}")
    console.print(table)
    console.print(f"[green]wrote[/green] {riders_path}")
    console.print(f"[green]wrote[/green] {trips_path}")
    return riders_path, trips_path
