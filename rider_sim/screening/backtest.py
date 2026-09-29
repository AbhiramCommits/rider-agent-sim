"""Back-test: one real behavioral relationship held out, predicted blind.

The holdout (see ``config.BACKTEST_HOLDOUT``, echoed in the report) is the
real demand response across two surge bands. It is genuinely held out: the
personas are fit from fares, waits, and miles only -- surge never enters the
fitting -- so the surge response is out-of-sample for the simulation.

Real effect: log ratio of the trip-count shares in the high vs low surge
bands, computed from the processed real trips. Simulated effect: log ratio
of the simulated accept rates on trace offers whose surge falls in the same
bands. Verdicts: PASS (CI covers the real value and directions agree),
PARTIAL (direction agrees but CI misses), FAIL (direction disagrees).

Honesty note: the real trip shares reflect the *equilibrium* allocation of
rides (demand and supply together), not a controlled experiment, so exact
agreement is not expected; the back-test is a directional sanity gate and a
documented failure mode is a real result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rider_sim.config import BACKTEST_HOLDOUT

_LAPLACE = 0.5


def _band_mask(surge: pd.Series, band: list[float]) -> pd.Series:
    lo, hi = band
    return (surge >= lo) & (surge < hi)


def real_surge_demand(trips: pd.DataFrame, holdout: dict[str, Any]) -> dict[str, Any]:
    """Real log demand ratio across the holdout surge bands."""
    high = trips[_band_mask(trips["surge_ratio"], holdout["high_band"])]
    low = trips[_band_mask(trips["surge_ratio"], holdout["low_band"])]
    n_total = len(trips)
    share_high = len(high) / n_total
    share_low = len(low) / n_total
    return {
        "n_total": n_total,
        "n_high": len(high),
        "n_low": len(low),
        "share_high": share_high,
        "share_low": share_low,
        "log_ratio": float(np.log(share_high / share_low)),
    }


def _laplace_rate(accepted: pd.Series) -> float:
    return (float(accepted.sum()) + _LAPLACE) / (len(accepted) + 2.0 * _LAPLACE)


def sim_surge_demand(
    trace: pd.DataFrame, holdout: dict[str, Any], n_bootstrap: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Simulated log demand ratio across the same bands, with a rider-bootstrap CI."""
    high = trace[_band_mask(trace["surge"], holdout["high_band"])]
    low = trace[_band_mask(trace["surge"], holdout["low_band"])]
    if len(high) < 5 or len(low) < 5:
        return {
            "n_high": len(high),
            "n_low": len(low),
            "log_ratio": float("nan"),
            "ci": (float("nan"), float("nan")),
        }
    log_ratio = float(np.log(_laplace_rate(high["accepted"]) / _laplace_rate(low["accepted"])))

    rng = np.random.default_rng(seed)
    riders = trace["rider_id"].unique()
    boot_log_ratios = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled = rng.choice(riders, size=len(riders), replace=True)
        mask = trace["rider_id"].isin(sampled)
        boot_high = trace[mask & _band_mask(trace["surge"], holdout["high_band"])]
        boot_low = trace[mask & _band_mask(trace["surge"], holdout["low_band"])]
        if len(boot_high) == 0 or len(boot_low) == 0:
            boot_log_ratios[b] = log_ratio
            continue
        boot_log_ratios[b] = float(
            np.log(_laplace_rate(boot_high["accepted"]) / _laplace_rate(boot_low["accepted"]))
        )
    ci = (float(np.percentile(boot_log_ratios, 2.5)), float(np.percentile(boot_log_ratios, 97.5)))
    return {
        "n_high": len(high),
        "n_low": len(low),
        "log_ratio": log_ratio,
        "ci": ci,
    }


def run_backtest(
    trace: pd.DataFrame,
    trips: pd.DataFrame,
    holdout: dict[str, Any] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare simulated vs real held-out surge response."""
    holdout = holdout if holdout is not None else BACKTEST_HOLDOUT
    real = real_surge_demand(trips, holdout)
    sim = sim_surge_demand(trace, holdout, n_bootstrap=n_bootstrap, seed=seed)
    signed_error = (
        sim["log_ratio"] - real["log_ratio"] if np.isfinite(sim["log_ratio"]) else float("nan")
    )
    direction_agrees = bool(
        np.isfinite(signed_error) and (sim["log_ratio"] * real["log_ratio"] > 0)
    )
    coverage = bool(np.isfinite(sim["ci"][0]) and sim["ci"][0] <= real["log_ratio"] <= sim["ci"][1])
    if direction_agrees and coverage:
        verdict = "PASS"
    elif direction_agrees:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return {
        "holdout": holdout,
        "real": real,
        "sim": sim,
        "signed_error": signed_error,
        "direction_agrees": direction_agrees,
        "ci_covers_real": coverage,
        "verdict": verdict,
        "note": (
            "The real ratio reflects the equilibrium allocation of rides "
            "(demand and supply together), not a controlled experiment; "
            "exact agreement is not expected -- the back-test is a "
            "directional sanity gate."
        ),
    }
