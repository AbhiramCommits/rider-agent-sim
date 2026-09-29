"""Trace store: one Parquet per (run, config), one row per decision.

Traces are self-contained for downstream metrics: they snapshot the persona
traits (for segmentation), the offer fields, the decision, and the accept
probability used for calibration.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rider_sim.config import TRACES_DIR

TRACE_COLUMNS = [
    "run_id",
    "config",
    "rider_id",
    "offer_id",
    "origin_zone",
    "dest_zone",
    "purpose",
    "pickup_hour",
    "quoted_fare",
    "surge",
    "eta_min",
    "eta_framing",
    "discount_pct",
    "time_of_day",
    "weather",
    "transit_alt_minutes",
    "price_mult",
    "surge_mult",
    "is_base",
    "reject_counterfactual",
    "action",
    "accepted",
    "confidence",
    "reservation_fare",
    "p_accept",
    "price_sensitivity",
    "wait_tolerance_minutes",
    "has_transit_alternative",
    "income_bracket",
    "loyalty_tier",
    "cost_usd",
    "prompt_version",
]


def trace_path(run_id: str, config: str) -> Path:
    return TRACES_DIR / run_id / f"{config}.parquet"


def load_trace(run_id: str, config: str) -> pd.DataFrame:
    path = trace_path(run_id, config)
    if not path.exists():
        raise FileNotFoundError(f"trace {path} not found; run the simulation first")
    return pd.read_parquet(path)


def save_trace(run_id: str, config: str, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    path = trace_path(run_id, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
