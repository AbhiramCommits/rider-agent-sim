"""Mechanism check: demand elasticity of accept rate w.r.t. surge.

The simulated trace is binned by surge multiplier; a log-log regression of
accept rate on surge gives the demand elasticity of the simulated policy.
A bootstrap (resampling decision events) yields a 95% CI.

The estimate is compared against a documented real-world ride-hailing
elasticity range (see README "Literature benchmark"):

    Cohen, Hahn, Hall, Levitt & Metcalfe (2016), "Using Big Data to Estimate
    Consumer Surplus: The Case of Uber", NBER Working Paper 22627.
    Own-price elasticity of demand for Uber rides: roughly -0.4 to -0.6.

Sign PASS requires a negative elasticity; magnitude PASS requires the
bootstrap CI to overlap the literature range. Both must pass for the
mechanism verdict. This is the "plausible mechanism, not just surface fit"
check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LITERATURE_ELASTICITY_RANGE = (-0.6, -0.4)
LITERATURE_CITATION = (
    "Cohen, Hahn, Hall, Levitt & Metcalfe (2016), 'Using Big Data to Estimate "
    "Consumer Surplus: The Case of Uber', NBER Working Paper 22627"
)
MIN_SURGE_LEVELS = 3
MAX_SURGE_BANDS = 8


def _band_rates(trace: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Accept rate per surge band (Laplace-smoothed) and band centers."""
    surge = trace["surge"].to_numpy(dtype=float)
    accepted = trace["accepted"].to_numpy(dtype=float)
    edges = np.quantile(surge, np.linspace(0.0, 1.0, MAX_SURGE_BANDS + 1))
    band = np.digitize(surge, edges, right=True)
    band = np.clip(band, 1, MAX_SURGE_BANDS)
    centers: list[float] = []
    rates: list[float] = []
    for b in np.unique(band):
        mask = band == b
        n = int(mask.sum())
        k = float(accepted[mask].sum())
        rates.append((k + 0.5) / (n + 1.0))
        centers.append(float(np.mean(surge[mask])))
    return np.asarray(centers), np.asarray(rates)


def _log_log_slope(centers: np.ndarray, rates: np.ndarray) -> float:
    log_x = np.log(centers)
    log_y = np.log(rates)
    if len(centers) < MIN_SURGE_LEVELS or np.unique(np.round(log_x, 3)).size < MIN_SURGE_LEVELS:
        return float("nan")
    slope, _ = np.polyfit(log_x, log_y, 1)
    return float(slope)


def surge_elasticity(
    trace: pd.DataFrame, n_bootstrap: int = 500, seed: int = 0
) -> dict[str, object]:
    """Estimate surge elasticity with a bootstrap CI; PASS/FAIL vs literature."""
    centers, rates = _band_rates(trace)
    slope = _log_log_slope(centers, rates)

    rng = np.random.default_rng(seed)
    boot_slopes: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(trace), size=len(trace))
        resampled = trace.iloc[index]
        boot_centers, boot_rates = _band_rates(resampled)
        boot_slopes.append(_log_log_slope(boot_centers, boot_rates))
    finite = np.asarray([s for s in boot_slopes if np.isfinite(s)])
    if len(finite):
        ci = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
    else:
        ci = (float("nan"), float("nan"))

    literature_lo, literature_hi = LITERATURE_ELASTICITY_RANGE
    sign_pass = bool(np.isfinite(slope) and slope < 0.0)
    magnitude_pass = bool(np.isfinite(ci[0]) and ci[0] <= literature_hi and ci[1] >= literature_lo)
    return {
        "elasticity": slope,
        "elasticity_ci": ci,
        "literature_range": list(LITERATURE_ELASTICITY_RANGE),
        "citation": LITERATURE_CITATION,
        "n_bands": len(centers),
        "sign_pass": sign_pass,
        "magnitude_pass": magnitude_pass,
        "verdict": "PASS" if (sign_pass and magnitude_pass) else "FAIL",
    }
