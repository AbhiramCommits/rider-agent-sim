"""Screening orchestrator: trace -> cell lifts -> power -> back-test -> report."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from rider_sim.config import BACKTEST_HOLDOUT, DEFAULT_MONTH, TRACES_DIR, TRIPS_PATH
from rider_sim.eval.traces import load_trace
from rider_sim.screening.backtest import run_backtest
from rider_sim.screening.power import power_analysis
from rider_sim.screening.report import render_screening_report
from rider_sim.screening.screen import CONTROL_CELL, INTERVENTION_CELLS, cell_rows, screen_cell

console = Console()

DECISION_RULE = (
    "Screen in iff the CUPED-adjusted accept-rate lift's lower 95% CI bound is "
    "> 0 (raw CI used when CUPED does not reduce variance), and -- for cells "
    "with a back-tested counterpart -- the back-test does not contradict the "
    "direction (verdict != FAIL)."
)


def _control_accept_rate(trace: pd.DataFrame) -> float:
    control = cell_rows(trace, CONTROL_CELL)
    return float(control["accepted"].mean())


def run_screening(
    run_id: str,
    with_backtest: bool = True,
    config: str = "full",
    trips_path: Path = TRIPS_PATH,
    month: str = DEFAULT_MONTH,
    seed: int = 7,
    n_boot_riders: int = 1000,
    n_boot_samples: int = 200,
) -> Path:
    """Screen all intervention cells in a run's trace; write the report."""
    if not (TRACES_DIR / run_id).exists():
        raise FileNotFoundError(
            f"no traces for run {run_id!r} under {TRACES_DIR}; "
            "run `python -m rider_sim evaluate --run-id " + run_id + "` first"
        )
    trace = load_trace(run_id, config)
    trips = pd.read_parquet(trips_path)
    control_rate = _control_accept_rate(trace)

    cell_results = []
    for cell in INTERVENTION_CELLS:
        try:
            result = screen_cell(
                trace,
                cell,
                seed=seed,
                n_boot_riders=n_boot_riders,
                n_boot_samples=n_boot_samples,
            )
        except ValueError as exc:
            console.log(f"[yellow]skipping cell {cell.name}: {exc}[/yellow]")
            continue
        primary = result["metrics"]["accept_rate"]
        if not np.isfinite(primary["lift"]):
            console.log(f"[yellow]skipping cell {cell.name}: no finite primary lift[/yellow]")
            continue
        lift = (
            primary["lift_cuped"]
            if np.isfinite(primary["lift_cuped"]) and primary["cuped_var_reduction"] > 0
            else primary["lift"]
        )
        try:
            result["power"] = power_analysis(lift, control_rate, month=month)
        except ValueError:
            result["power"] = {
                "control_rate": control_rate,
                "treatment_rate": control_rate,
                "effect_size": 0.0,
                "n_per_arm": None,
                "n_total": None,
                "daily_offers": None,
                "days": None,
                "note": "zero simulated effect; no finite sample size exists",
            }
        cell_results.append(result)

    cell_results.sort(key=lambda r: r["metrics"]["accept_rate"]["lift_cuped"], reverse=True)

    backtest = (
        run_backtest(trace, trips, BACKTEST_HOLDOUT, n_bootstrap=n_boot_riders, seed=seed)
        if with_backtest
        else None
    )
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    return render_screening_report(
        run_id=run_id,
        config=config,
        control_rate=control_rate,
        cell_results=cell_results,
        backtest=backtest,
        decision_rule=DECISION_RULE,
        generated_at=generated_at,
        month=month,
    )
