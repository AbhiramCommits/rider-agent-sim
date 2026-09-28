"""Data-grounded rider personas.

Every persona is fit from that pseudo-rider's *own* empirical trip stats, so
personas reflect observed behavior rather than hand-invented archetypes:

- ``price_sensitivity``  1 - (0.6 * fare-per-mile percentile + 0.4 * tip rate).
  Riders revealed to pay high fares per mile and/or tip often are treated as
  less price sensitive (revealed preference). Percentiles are computed across
  the full pseudo-rider panel.
- ``wait_tolerance_minutes``  max of the 75th percentile of waits the rider
  actually accepted and 1.25x their median accepted wait. Because every real
  trip was accepted, accepted waits are a *lower bound* on tolerance; this is
  a conservative estimate, not a survey answer.
- ``income_bracket``  quartile of the rider's median fare paid (spend-based
  proxy: low / middle / upper_middle / high).
- ``has_transit_alternative``  True when the rider's median trip is short
  (< 2.5 mi) and they are price sensitive (>= 0.55); a derived heuristic,
  not observed transit access.
- ``loyalty_tier``  frequency-based from observed_trip_count (>= 25 platinum,
  >= 15 gold, >= 8 silver, else member).
- ``trip_purpose_mix``  share of the rider's real trips classified by
  zone/hour: airport (zone 1, 132, 138), social (nights 21:00-04:59),
  commute (weekday 06-09 / 15-19), errand (everything else). Shares sum to 1.

``sample_personas`` is deterministic for a given (data, seed) and persists
the panel to ``data/processed/personas.parquet``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.console import Console
from rich.table import Table
from scipy.stats import rankdata

from rider_sim.config import PERSONAS_PATH, RIDERS_PATH, TRIPS_PATH

console = Console()

PURPOSE_KEYS = ("commute", "social", "errand", "airport")
AIRPORT_ZONES = frozenset({1, 132, 138})

IncomeBracket = Literal["low", "middle", "upper_middle", "high"]
LoyaltyTier = Literal["member", "silver", "gold", "platinum"]

_FARE_PCTL_WEIGHT = 0.6
_TIP_RATE_WEIGHT = 0.4
_TRANSIT_MAX_MILES = 2.5
_TRANSIT_MIN_SENSITIVITY = 0.55


class RiderPersona(BaseModel):
    """A behavioral persona fit from one pseudo-rider's real trip history."""

    model_config = ConfigDict(frozen=True)

    rider_id: int = Field(ge=0)
    home_zone: int = Field(ge=1)
    work_zone: int = Field(ge=1)
    trip_purpose_mix: dict[str, float]
    price_sensitivity: float = Field(ge=0.0, le=1.0)
    wait_tolerance_minutes: float = Field(gt=0.0)
    income_bracket: IncomeBracket
    has_transit_alternative: bool
    loyalty_tier: LoyaltyTier
    observed_trip_count: int = Field(ge=1)
    median_fare_paid: float = Field(ge=0.0)
    median_wait_experienced: float = Field(ge=0.0)

    @field_validator("trip_purpose_mix")
    @classmethod
    def _validate_purpose_mix(cls, value: dict[str, float]) -> dict[str, float]:
        keys = set(value)
        if keys != set(PURPOSE_KEYS):
            raise ValueError(f"trip_purpose_mix must have exactly keys {PURPOSE_KEYS}")
        if any(share < 0.0 for share in value.values()):
            raise ValueError("trip_purpose_mix shares must be non-negative")
        if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-6):
            raise ValueError("trip_purpose_mix shares must sum to 1")
        return value


def classify_purpose(trips: pd.DataFrame) -> dict[str, float]:
    """Classify a rider's trips into purpose shares via zone/hour rules."""
    airport = trips["pickup_zone"].isin(AIRPORT_ZONES) | trips["dropoff_zone"].isin(AIRPORT_ZONES)
    social = ~airport & ((trips["pickup_hour"] >= 21) | (trips["pickup_hour"] <= 4))
    commute = (
        ~airport
        & ~social
        & (trips["is_weekend"] == 0)
        & (trips["pickup_hour"].between(6, 9) | trips["pickup_hour"].between(15, 19))
    )
    errand = ~(airport | social | commute)
    counts = {
        "airport": int(airport.sum()),
        "social": int(social.sum()),
        "commute": int(commute.sum()),
        "errand": int(errand.sum()),
    }
    total = float(sum(counts.values())) or 1.0
    return {key: counts[key] / total for key in PURPOSE_KEYS}


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def fit_personas(riders: pd.DataFrame, trips: pd.DataFrame) -> list[RiderPersona]:
    """Fit one persona per pseudo-rider from the rider's own trip stats."""
    trips = trips.copy()
    trips["fare_per_mile"] = trips["fare_paid"] / trips["trip_miles"]
    stats = trips.groupby("rider_id", sort=True).agg(
        median_fare_paid=("fare_paid", "median"),
        median_fpm=("fare_per_mile", "median"),
        median_miles=("trip_miles", "median"),
        median_wait_min=("wait_secs", lambda s: float(s.median()) / 60.0),
        p75_wait_min=("wait_secs", lambda s: float(s.quantile(0.75)) / 60.0),
        tip_rate=("tips", lambda s: float((s > 0).mean())),
        observed_trip_count=("rider_id", "size"),
    )
    stats["fare_pctl"] = _percentile_rank(stats["median_fpm"].to_numpy())
    stats["spend_pctl"] = _percentile_rank(stats["median_fare_paid"].to_numpy())

    mix_by_rider = {
        rider_id: classify_purpose(group)
        for rider_id, group in trips.groupby("rider_id", sort=True)
    }

    personas: list[RiderPersona] = []
    for row in riders.itertuples(index=False):
        rider_id = int(cast(int, row.rider_id))
        if rider_id not in stats.index:
            continue
        s = stats.loc[rider_id]
        price_sensitivity = float(
            np.clip(
                1.0
                - (
                    _FARE_PCTL_WEIGHT * cast(float, s["fare_pctl"])
                    + _TIP_RATE_WEIGHT * cast(float, s["tip_rate"])
                ),
                0.0,
                1.0,
            )
        )
        wait_tolerance = max(
            float(cast(float, s["p75_wait_min"])),
            1.25 * float(cast(float, s["median_wait_min"])),
            0.5,
        )
        spend_pctl = float(cast(float, s["spend_pctl"]))
        if spend_pctl < 0.25:
            income: IncomeBracket = "low"
        elif spend_pctl < 0.5:
            income = "middle"
        elif spend_pctl < 0.75:
            income = "upper_middle"
        else:
            income = "high"
        n_trips = int(cast(int, row.observed_trip_count))
        if n_trips >= 25:
            tier: LoyaltyTier = "platinum"
        elif n_trips >= 15:
            tier = "gold"
        elif n_trips >= 8:
            tier = "silver"
        else:
            tier = "member"
        personas.append(
            RiderPersona(
                rider_id=rider_id,
                home_zone=int(cast(int, row.home_zone)),
                work_zone=int(cast(int, row.work_zone)),
                trip_purpose_mix=mix_by_rider[rider_id],
                price_sensitivity=price_sensitivity,
                wait_tolerance_minutes=wait_tolerance,
                income_bracket=income,
                has_transit_alternative=(
                    float(cast(float, s["median_miles"])) <= _TRANSIT_MAX_MILES
                    and price_sensitivity >= _TRANSIT_MIN_SENSITIVITY
                ),
                loyalty_tier=tier,
                observed_trip_count=n_trips,
                median_fare_paid=float(cast(float, s["median_fare_paid"])),
                median_wait_experienced=float(cast(float, s["median_wait_min"])),
            )
        )
    return personas


def personas_to_frame(personas: list[RiderPersona]) -> pd.DataFrame:
    records = [p.model_dump() for p in personas]
    frame = pd.DataFrame(records)
    frame["trip_purpose_mix"] = [json.dumps(rec["trip_purpose_mix"]) for rec in records]
    return frame


def load_personas(path: Path = PERSONAS_PATH) -> list[RiderPersona]:
    """Load a persisted persona panel back into validated model objects."""
    frame = pd.read_parquet(path)
    personas: list[RiderPersona] = []
    for record in frame.to_dict(orient="records"):
        record["trip_purpose_mix"] = cast(dict[str, float], json.loads(record["trip_purpose_mix"]))
        personas.append(RiderPersona.model_validate(record))
    return personas


def sample_personas(
    n: int,
    seed: int,
    riders_path: Path = RIDERS_PATH,
    trips_path: Path = TRIPS_PATH,
    output_path: Path = PERSONAS_PATH,
) -> pd.DataFrame:
    """Sample ``n`` pseudo-riders, fit personas, persist to ``output_path``.

    Deterministic for a given (data, seed). Raises ``ValueError`` when ``n``
    exceeds the number of available pseudo-riders.
    """
    riders = pd.read_parquet(riders_path)
    trips = pd.read_parquet(trips_path)
    if n > len(riders):
        raise ValueError(
            f"Requested {n} personas but only {len(riders)} pseudo-riders available; "
            f"run `build` with --n-riders >= {n} first."
        )
    if n < 1:
        raise ValueError("n must be >= 1")

    picked = riders.sample(n=n, random_state=seed).reset_index(drop=True)
    personas = fit_personas(picked, trips)
    frame = personas_to_frame(personas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    console.print(_summary_table(frame))
    console.print(f"[green]wrote[/green] {output_path}")
    return frame


def _summary_table(frame: pd.DataFrame) -> Table:
    table = Table(title=f"Personas (n={len(frame)})")
    table.add_column("field")
    table.add_column("distribution", justify="right")
    table.add_row("price_sensitivity (med)", f"{frame.price_sensitivity.median():.2f}")
    table.add_row("wait_tolerance_min (med)", f"{frame.wait_tolerance_minutes.median():.1f}")
    table.add_row("has_transit_alternative", f"{frame.has_transit_alternative.mean():.0%}")
    table.add_row("income brackets", str(frame.income_bracket.value_counts().to_dict()))
    table.add_row("loyalty tiers", str(frame.loyalty_tier.value_counts().to_dict()))
    return table
