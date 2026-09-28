"""Validate simulated rider choices against real behavioral data.

Compares the simulated choice output to the real trip distributions of the
simulated personas and checks behavioral sanity:

- base accept rate at the observed (1.0 price / 1.0 wait) offer,
- downward-sloping price demand (accept rate at x0.8 price > x1.25 price),
- downward-sloping wait demand (accept rate at x0.5 wait > x2.0 wait),
- zonal coherence (offer zone pairs come from the persona's real trips),
- hour-of-day distribution distance (Jensen-Shannon vs real trip hours),
- a LightGBM classifier fit on the simulated choices to quantify how much
  persona traits (price sensitivity, wait tolerance) drive decisions (AUC +
  feature importances).

Writes a summary to ``data/processed/eval_metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from rider_sim.config import EVAL_METRICS_PATH, PERSONAS_PATH, SIM_CHOICES_PATH, TRIPS_PATH

console = Console()

_FEATURES = [
    "price_mult",
    "wait_mult",
    "price_sensitivity",
    "wait_tolerance_minutes",
    "offered_fare",
    "trip_miles",
]


def evaluate(
    sim_path: Path = SIM_CHOICES_PATH,
    trips_path: Path = TRIPS_PATH,
    personas_path: Path = PERSONAS_PATH,
    metrics_path: Path = EVAL_METRICS_PATH,
) -> dict[str, Any]:
    if not sim_path.exists():
        raise FileNotFoundError(f"{sim_path} missing; run `simulate` first.")

    choices = pd.read_parquet(sim_path)
    trips = pd.read_parquet(trips_path)
    personas = pd.read_parquet(personas_path)

    sim_rider_ids = sorted(choices["rider_id"].unique().tolist())
    real = trips[trips["rider_id"].isin(sim_rider_ids)]
    sim = choices.merge(
        personas[["rider_id", "price_sensitivity", "wait_tolerance_minutes"]],
        on="rider_id",
        how="left",
    )

    def accept_rate(**filters: Any) -> float:
        mask = np.ones(len(sim), dtype=bool)
        for col, value in filters.items():
            mask &= sim[col] == value
        return float(sim.loc[mask, "accepted"].mean())

    base_accept = accept_rate(price_mult=1.0, wait_mult=1.0)
    price_elasticity = accept_rate(price_mult=0.8, wait_mult=1.0) - accept_rate(
        price_mult=1.25, wait_mult=1.0
    )
    wait_elasticity = accept_rate(price_mult=1.0, wait_mult=0.5) - accept_rate(
        price_mult=1.0, wait_mult=2.0
    )

    real_pairs = set(zip(real["pickup_zone"], real["dropoff_zone"], strict=False))
    zone_pair_coverage = float(
        np.mean(
            [
                (pickup, dropoff) in real_pairs
                for pickup, dropoff in zip(sim["pickup_zone"], sim["dropoff_zone"], strict=False)
            ]
        )
    )

    hour_js = _hour_js(sim, real)

    lgbm_auc, top_features = _lgbm_fit(sim)

    checks = {
        "price_elasticity_positive": price_elasticity > 0.0,
        "wait_elasticity_positive": wait_elasticity > 0.0,
        "base_accept_plausible": 0.3 < base_accept < 1.0,
        "zone_pair_coverage": zone_pair_coverage > 0.95,
        "hour_js_low": hour_js < 0.3,
        "lgbm_auc_above_chance": lgbm_auc > 0.55,
    }
    metrics: dict[str, Any] = {
        "n_choices": len(sim),
        "n_personas": len(sim_rider_ids),
        "decision_mode": (
            str(choices["decision_mode"].iloc[0]) if "decision_mode" in choices else "unknown"
        ),
        "base_accept_rate": base_accept,
        "price_elasticity": price_elasticity,
        "wait_elasticity": wait_elasticity,
        "zone_pair_coverage": zone_pair_coverage,
        "hour_dist_js": hour_js,
        "lgbm_auc": lgbm_auc,
        "top_gain_features": top_features,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    table = Table(title="Evaluation vs real behavioral data")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("status", justify="center")
    for key, check_key, label in [
        ("base_accept_rate", "base_accept_plausible", "base accept rate"),
        ("price_elasticity", "price_elasticity_positive", "price elasticity"),
        ("wait_elasticity", "wait_elasticity_positive", "wait elasticity"),
        ("zone_pair_coverage", "zone_pair_coverage", "zone pair coverage"),
        ("hour_dist_js", "hour_js_low", "hour dist (JS)"),
        ("lgbm_auc", "lgbm_auc_above_chance", "LightGBM AUC"),
    ]:
        value = metrics[key]
        table.add_row(label, f"{value:.3f}", _status_icon(checks.get(check_key, True)))
    console.print(table)
    if metrics["all_checks_pass"]:
        console.print("[green]all sanity checks passed[/green]")
    else:
        console.print("[yellow]some sanity checks failed[/yellow]")
    console.print(f"[green]wrote[/green] {metrics_path}")
    return metrics


def _hour_js(sim: pd.DataFrame, real: pd.DataFrame) -> float:
    def hist(hours: pd.Series) -> np.ndarray:
        counts: np.ndarray = np.bincount(hours.clip(0, 23).astype(int), minlength=24)
        return counts / max(int(counts.sum()), 1)

    return float(jensenshannon(hist(sim["pickup_hour"]), hist(real["pickup_hour"])))


def _lgbm_fit(sim: pd.DataFrame) -> tuple[float, dict[str, float]]:
    frame = sim.dropna(subset=[*_FEATURES, "accepted"])
    x = frame[_FEATURES].astype(float)
    y = frame["accepted"].astype(int)
    if len(frame) < 40 or y.nunique() < 2:
        return float("nan"), {}
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )
    model = lgb.LGBMClassifier(n_estimators=200, verbosity=-1, random_state=0)
    model.fit(x_train, y_train)
    proba = cast(np.ndarray, model.predict_proba(x_test))
    auc = float(roc_auc_score(y_test, proba[:, 1]))
    gains = dict(
        zip(
            _FEATURES,
            model.booster_.feature_importance(importance_type="gain"),
            strict=False,
        )
    )
    top = {k: float(v) for k, v in sorted(gains.items(), key=lambda kv: kv[1], reverse=True)[:3]}
    return auc, top


def _status_icon(ok: bool) -> str:
    return "[green]pass[/green]" if ok else "[red]fail[/red]"
