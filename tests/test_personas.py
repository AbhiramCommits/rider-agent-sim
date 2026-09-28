"""Tests for persona schema validity and seed-deterministic sampling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rider_sim.personas import PURPOSE_KEYS, RiderPersona, load_personas, sample_personas

ZONES = [1, 132, 138, 230, 186, 100]


@pytest.fixture()
def panel(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny synthetic rider/trips panel shaped like the real processed data."""
    rng = np.random.default_rng(11)
    rider_rows: list[dict[str, int]] = []
    trip_rows: list[dict[str, float]] = []
    for rider_id in range(10):
        home_zone = int(rng.choice(ZONES))
        n_trips = int(rng.integers(6, 12))
        rider_rows.append(
            {
                "rider_id": rider_id,
                "home_zone": home_zone,
                "work_zone": int(rng.choice(ZONES)),
                "observed_trip_count": n_trips,
            }
        )
        for _ in range(n_trips):
            miles = float(rng.uniform(0.5, 15.0))
            fare = miles * float(rng.uniform(2.5, 8.0))
            wait = float(rng.uniform(30.0, 900.0))
            trip_rows.append(
                {
                    "rider_id": rider_id,
                    "pickup_zone": home_zone,
                    "dropoff_zone": int(rng.choice(ZONES)),
                    "pickup_hour": int(rng.integers(0, 24)),
                    "is_weekend": int(rng.integers(0, 2)),
                    "miles_bucket": int(rng.integers(0, 5)),
                    "base_passenger_fare": fare,
                    "tips": fare * float(rng.choice([0.0, 0.1, 0.2])),
                    "trip_miles": miles,
                    "trip_time": miles * float(rng.uniform(120.0, 400.0)),
                    "wait_secs": wait,
                    "driver_pay": fare * float(rng.uniform(0.3, 0.9)),
                    "fare_paid": fare * 1.05,
                    "surge_ratio": float(rng.uniform(0.8, 2.5)),
                }
            )
    riders = pd.DataFrame(rider_rows)
    trips = pd.DataFrame(trip_rows)
    riders_path = tmp_path / "riders.parquet"
    trips_path = tmp_path / "trips.parquet"
    riders.to_parquet(riders_path, index=False)
    trips.to_parquet(trips_path, index=False)
    return riders_path, trips_path


def test_persona_schema_valid(panel: tuple[Path, Path], tmp_path: Path) -> None:
    riders_path, trips_path = panel
    output = tmp_path / "personas.parquet"

    frame = sample_personas(
        n=5, seed=7, riders_path=riders_path, trips_path=trips_path, output_path=output
    )
    assert output.exists()
    assert len(frame) == 5

    personas = load_personas(output)
    assert len(personas) == 5
    for persona in personas:
        assert 0.0 <= persona.price_sensitivity <= 1.0
        assert persona.wait_tolerance_minutes > 0.0
        assert persona.observed_trip_count >= 1
        assert persona.median_fare_paid >= 0.0
        assert persona.median_wait_experienced >= 0.0
        assert set(persona.trip_purpose_mix) == set(PURPOSE_KEYS)
        assert math.isclose(sum(persona.trip_purpose_mix.values()), 1.0, abs_tol=1e-6)
        assert persona.income_bracket in {"low", "middle", "upper_middle", "high"}
        assert persona.loyalty_tier in {"member", "silver", "gold", "platinum"}
        assert isinstance(persona.has_transit_alternative, bool)


def test_persona_schema_rejects_invalid_mix() -> None:
    good = dict(
        rider_id=0,
        home_zone=1,
        work_zone=2,
        trip_purpose_mix={"commute": 0.5, "social": 0.2, "errand": 0.2, "airport": 0.1},
        price_sensitivity=0.4,
        wait_tolerance_minutes=5.0,
        income_bracket="middle",
        has_transit_alternative=False,
        loyalty_tier="gold",
        observed_trip_count=10,
        median_fare_paid=25.0,
        median_wait_experienced=3.0,
    )
    RiderPersona(**good)

    with pytest.raises(ValidationError):
        RiderPersona(**{**good, "trip_purpose_mix": {"commute": 1.0}})  # missing keys
    with pytest.raises(ValidationError):
        RiderPersona(  # shares don't sum to 1
            **{
                **good,
                "trip_purpose_mix": {"commute": 0.9, "social": 0.5, "errand": 0.0, "airport": 0.0},
            }
        )
    with pytest.raises(ValidationError):
        RiderPersona(**{**good, "price_sensitivity": 1.7})
    with pytest.raises(ValidationError):
        RiderPersona(**{**good, "wait_tolerance_minutes": -2.0})


def test_sampling_is_seed_deterministic(panel: tuple[Path, Path], tmp_path: Path) -> None:
    riders_path, trips_path = panel
    first = sample_personas(
        n=6,
        seed=42,
        riders_path=riders_path,
        trips_path=trips_path,
        output_path=tmp_path / "a.parquet",
    )
    second = sample_personas(
        n=6,
        seed=42,
        riders_path=riders_path,
        trips_path=trips_path,
        output_path=tmp_path / "b.parquet",
    )
    pd.testing.assert_frame_equal(first, second)


def test_sample_personas_errors_when_n_too_large(panel: tuple[Path, Path], tmp_path: Path) -> None:
    riders_path, trips_path = panel
    with pytest.raises(ValueError, match="only 10 pseudo-riders"):
        sample_personas(
            n=11,
            seed=1,
            riders_path=riders_path,
            trips_path=trips_path,
            output_path=tmp_path / "c.parquet",
        )
