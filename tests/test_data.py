"""Tests for the data layer: fetch (idempotent download) and build_riders."""

from __future__ import annotations

import datetime as dt
import importlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rider_sim.data.fetch as fetch_module
from rider_sim.data.build_riders import assign_riders, build_riders, load_trips

build_module = importlib.import_module("rider_sim.data.build_riders")


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        return self._data.read(size)


def _trips_parquet_bytes(n_rows: int = 50) -> bytes:
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "PULocationID": pa.array(rng.integers(1, 100, n_rows)),
            "DOLocationID": pa.array(rng.integers(1, 100, n_rows)),
            "pickup_datetime": pa.array(
                [
                    dt.datetime(2024, 1, 1, 8, 0) + dt.timedelta(minutes=10 * i)
                    for i in range(n_rows)
                ]
            ),
            "request_datetime": pa.array(
                [
                    dt.datetime(2024, 1, 1, 7, 55) + dt.timedelta(minutes=10 * i)
                    for i in range(n_rows)
                ]
            ),
            "trip_miles": pa.array(rng.uniform(1.0, 10.0, n_rows)),
            "trip_time": pa.array(rng.uniform(300.0, 1200.0, n_rows)),
            "base_passenger_fare": pa.array(rng.uniform(8.0, 40.0, n_rows)),
            "tips": pa.array(rng.uniform(0.0, 3.0, n_rows)),
            "driver_pay": pa.array(rng.uniform(4.0, 30.0, n_rows)),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_fetch_trips_idempotent_and_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fetch_module, "RAW_DIR", tmp_path)
    data = _trips_parquet_bytes()
    calls: list[str] = []

    def fake_urlopen(url: str) -> FakeResponse:
        calls.append(url)
        return FakeResponse(data)

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", fake_urlopen)
    first = fetch_module.fetch_trips("2024-01")
    second = fetch_module.fetch_trips("2024-01")
    assert first == second == tmp_path / "fhvhv_tripdata_2024-01.parquet"
    assert len(calls) == 1  # second call is a cache hit


def test_fetch_trips_removes_corrupt_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fetch_module, "RAW_DIR", tmp_path)
    dest = tmp_path / "fhvhv_tripdata_2024-01.parquet"
    dest.write_bytes(b"not a parquet file")
    with pytest.raises(ValueError, match="corrupt"):
        fetch_module.fetch_trips("2024-01")
    assert not dest.exists()


def test_fetch_zones_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fetch_module, "RAW_DIR", tmp_path)
    csv = ("LocationID,Borough,Zone\n1,EWR,Newark Airport\n" * 200).encode()
    calls: list[str] = []

    def fake_urlopen(url: str) -> FakeResponse:
        calls.append(url)
        return FakeResponse(csv)

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", fake_urlopen)
    first = fetch_module.fetch_zones()
    second = fetch_module.fetch_zones()
    assert first == second == tmp_path / "taxi_zone_lookup.csv"
    assert len(calls) == 1


def test_load_trips_and_assign_riders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = tmp_path / "fhvhv_tripdata_2024-01.parquet"
    raw.write_bytes(_trips_parquet_bytes(200))
    monkeypatch.setattr(build_module, "fetch_trips", lambda month: raw)

    df = load_trips("2024-01")
    assert {"pickup_zone", "dropoff_zone", "surge_ratio", "wait_secs", "fare_paid"} <= set(
        df.columns
    )
    riders, trips = assign_riders(df, n_riders=6, seed=7, min_trips=5, max_trips=40)
    assert len(riders) == 6
    assert riders["observed_trip_count"].between(5, 40).all()
    assert {"rider_id", "home_zone", "work_zone"} <= set(riders.columns)
    assert len(trips) == riders["observed_trip_count"].sum()
    counts = trips["rider_id"].value_counts().sort_index()
    assert (counts.to_numpy() == riders["observed_trip_count"].to_numpy()).all()


def test_build_riders_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(_trips_parquet_bytes(200))
    monkeypatch.setattr(build_module, "fetch_trips", lambda month: raw)
    riders_path = tmp_path / "riders.parquet"
    trips_path = tmp_path / "trips.parquet"
    out_riders, out_trips = build_riders(
        month="2024-01", n_riders=5, seed=7, riders_path=riders_path, trips_path=trips_path
    )
    assert out_riders.exists() and out_trips.exists()
    assert len(pd.read_parquet(out_riders)) == 5
