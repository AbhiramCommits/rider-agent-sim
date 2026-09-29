"""Counterfactual offer generation from real trips.

Every offer is derived from a real trip in ``trips.parquet``: real zone pair,
purpose (zone/hour rule, same as personas), base fare, base surge, and real
request-to-pickup wait. Variants apply price and surge multipliers:

- ``(price_mult, surge_mult) = (1.0, 1.0)`` -> the base offer; the real rider
  took this trip, so it carries a real accept label (y=1).
- ``(1.8, 1.0)`` -> clearly-too-expensive counterfactual; labeled y=0.
- the remaining variants are unlabeled (no honest real counterpart).

Documented synthetic fields (absent from TLC data): ``weather`` (seeded draw),
``eta_framing`` (deterministic rotation over the variant grid), ``time_of_day``
(computed from the trip hour), ``discount_pct`` (the 0.8x price variant is a
computed 20% discount; zero otherwise), ``transit_alt_minutes`` (lane median
miles at 12 mph, same derivation as the transit tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from rider_sim.agent.schemas import EtaFraming, Purpose, RideOffer, TimeOfDay
from rider_sim.personas import AIRPORT_ZONES

OFFER_GRID: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),
    (0.8, 1.0),
    (1.25, 1.0),
    (1.8, 1.0),
    (1.0, 2.0),
    (1.25, 2.0),
)
_ETA_FRAMINGS = ("numeric", "range", "reassuring")
_TRANSIT_SPEED_MPH = 12.0
_WEATHER = ("clear", "rain", "snow")
_MAX_SURGE = 5.0


def classify_trip_purpose(row: pd.Series) -> str:
    """Same zone/hour purpose rule used by the persona layer, per trip."""
    if row["pickup_zone"] in AIRPORT_ZONES or row["dropoff_zone"] in AIRPORT_ZONES:
        return "airport"
    hour = int(row["pickup_hour"])
    if hour >= 21 or hour <= 4:
        return "social"
    if row["is_weekend"] == 0 and (6 <= hour <= 9 or 15 <= hour <= 19):
        return "commute"
    return "errand"


def time_of_day(hour: int) -> str:
    if hour <= 5:
        return "night"
    if hour <= 11:
        return "morning"
    if hour <= 16:
        return "midday"
    return "evening"


def _lane_transit_minutes(trips: pd.DataFrame, origin: int, dest: int) -> float:
    lane = trips[(trips["pickup_zone"] == origin) & (trips["dropoff_zone"] == dest)]
    if lane.empty:
        lane = trips[trips["pickup_zone"] == origin]
    if lane.empty:
        return 60.0
    miles = float(lane["trip_miles"].median())
    return round(miles / _TRANSIT_SPEED_MPH * 60.0, 1)


@dataclass(frozen=True)
class EvalOffer:
    """One counterfactual offer row plus its variant metadata."""

    rider_id: int
    offer_id: str
    origin_zone: int
    dest_zone: int
    purpose: str
    quoted_fare: float
    surge: float
    eta_min: float
    eta_framing: str
    discount_pct: float
    time_of_day: str
    weather: str
    transit_alt_minutes: float
    price_mult: float
    surge_mult: float
    is_base: bool
    reject_counterfactual: bool

    def to_ride_offer(self) -> RideOffer:
        return RideOffer(
            offer_id=self.offer_id,
            origin_zone=self.origin_zone,
            dest_zone=self.dest_zone,
            trip_purpose=cast(Purpose, self.purpose),
            quoted_fare=round(self.quoted_fare, 2),
            surge_multiplier=round(self.surge, 2),
            eta_minutes=round(self.eta_min, 1),
            eta_framing=cast(EtaFraming, self.eta_framing),
            discount_pct=round(self.discount_pct, 1),
            time_of_day=cast(TimeOfDay, self.time_of_day),
            weather=self.weather,
            transit_alt_minutes=self.transit_alt_minutes,
        )


def generate_offers(
    trips: pd.DataFrame,
    n_riders: int,
    trips_per_rider: int,
    seed: int,
) -> pd.DataFrame:
    """Generate the counterfactual offer grid, deterministically seeded."""
    rng = np.random.default_rng(seed)
    rider_ids = trips["rider_id"].unique()
    sampled = rng.choice(rider_ids, size=min(n_riders, len(rider_ids)), replace=False)

    rows: list[dict[str, object]] = []
    for rider_id in sorted(int(r) for r in sampled):
        mine = trips[trips["rider_id"] == rider_id]
        bases = mine.sample(
            n=min(trips_per_rider, len(mine)), random_state=int(rng.integers(0, 2**31))
        )
        for trip_index, base_record in enumerate(bases.to_dict("records")):
            base = pd.Series(base_record)
            base_fare = float(base["fare_paid"])
            base_surge = float(min(max(float(base["surge_ratio"]), 1.0), 3.0))
            eta_min = float(base["wait_secs"]) / 60.0
            purpose = classify_trip_purpose(base)
            transit_min = _lane_transit_minutes(
                trips, int(base["pickup_zone"]), int(base["dropoff_zone"])
            )
            for variant_index, (price_mult, surge_mult) in enumerate(OFFER_GRID):
                surge = round(min(base_surge * surge_mult, _MAX_SURGE), 2)
                quoted_fare = round(base_fare * price_mult, 2)
                rows.append(
                    {
                        "rider_id": int(rider_id),
                        "offer_id": f"r{int(rider_id)}t{trip_index}v{variant_index}",
                        "origin_zone": int(base["pickup_zone"]),
                        "dest_zone": int(base["dropoff_zone"]),
                        "purpose": purpose,
                        "pickup_hour": int(base["pickup_hour"]),
                        "quoted_fare": quoted_fare,
                        "surge": surge,
                        "eta_min": eta_min,
                        "eta_framing": _ETA_FRAMINGS[variant_index % len(_ETA_FRAMINGS)],
                        "discount_pct": round((1.0 - price_mult) * 100.0, 1)
                        if price_mult < 1.0
                        else 0.0,
                        "time_of_day": time_of_day(int(base["pickup_hour"])),
                        "weather": str(rng.choice(_WEATHER)),
                        "transit_alt_minutes": transit_min,
                        "price_mult": price_mult,
                        "surge_mult": surge_mult,
                        "is_base": bool(price_mult == 1.0 and surge_mult == 1.0),
                        "reject_counterfactual": bool(price_mult == 1.8 and surge_mult == 1.0),
                    }
                )
    return pd.DataFrame(rows)
