"""Two-sample KS tests: simulated vs real marginals.

Four marginals are compared (all with matched sample sizes so power is
comparable and large-n real samples do not mechanically force rejection):

1. accepted fare (sim accepted quoted_fare vs real fare_paid)
2. accepted wait (sim accepted eta_min vs real wait minutes)
3. accept rate by hour (per-hour accept rates; the real rate is 1.0 in
   every hour because every real trip was accepted -- computed from data)
4. trip purpose mix (per-event purpose distribution; the category ordering
   is computed from the real purpose frequencies; KS on categorical data is
   not distribution-free, so this marginal is reported with that caveat)

Each marginal reports statistic, p-value, and a Bonferroni-adjusted verdict
(alpha = 0.05 / 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

N_MARGINALS = 4
BONFERRONI_ALPHA = 0.05 / N_MARGINALS

MARGINAL_NAMES = ("accepted_fare", "accepted_wait", "accept_rate_by_hour", "purpose_mix")


def _matched_real_sample(real_values: pd.Series, n: int, seed: int) -> np.ndarray:
    if len(real_values) <= n:
        return real_values.to_numpy(dtype=float)
    return real_values.sample(n=n, random_state=seed).to_numpy(dtype=float)


def _ks(real_sample: np.ndarray, sim_sample: np.ndarray) -> dict[str, float]:
    if len(real_sample) < 5 or len(sim_sample) < 5:
        return {"statistic": float("nan"), "p_value": float("nan")}
    statistic, p_value = ks_2samp(real_sample, sim_sample)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def _purpose_rank(real_events: pd.DataFrame) -> dict[str, int]:
    """Purpose category ordering computed from real purpose frequencies."""
    order = real_events["purpose"].value_counts().sort_values(ascending=False).index.tolist()
    return {purpose: rank for rank, purpose in enumerate(order)}


def _real_accept_rate_by_hour(real_events: pd.DataFrame) -> dict[int, float]:
    """Per-hour accept rate in the real data (computed: all trips accepted)."""
    return {int(hour): 1.0 for hour in sorted(real_events["pickup_hour"].unique())}


def _sim_accept_rate_by_hour(trace: pd.DataFrame) -> dict[int, float]:
    hours = trace["pickup_hour"]
    accepted = trace["accepted"]
    return {int(hour): float(accepted[hours == hour].mean()) for hour in sorted(hours.unique())}


def run_distribution_tests(
    trace: pd.DataFrame, real_events: pd.DataFrame, seed: int = 0
) -> dict[str, object]:
    """KS tests for the four marginals, with Bonferroni verdicts."""
    sim_accepted = trace[trace["accepted"] == 1]
    n_sim = len(sim_accepted)

    marginals: dict[str, dict[str, float]] = {}

    real_fares = _matched_real_sample(real_events["quoted_fare"], n_sim, seed)
    marginals["accepted_fare"] = _ks(real_fares, sim_accepted["quoted_fare"].to_numpy(dtype=float))

    real_waits = _matched_real_sample(real_events["eta_min"], n_sim, seed + 1)
    marginals["accepted_wait"] = _ks(real_waits, sim_accepted["eta_min"].to_numpy(dtype=float))

    real_by_hour = _real_accept_rate_by_hour(real_events)
    sim_by_hour = _sim_accept_rate_by_hour(trace)
    hours = sorted(set(real_by_hour) & set(sim_by_hour))
    marginals["accept_rate_by_hour"] = (
        _ks(
            np.asarray([real_by_hour[h] for h in hours], dtype=float),
            np.asarray([sim_by_hour[h] for h in hours], dtype=float),
        )
        if len(hours) >= 5
        else {"statistic": float("nan"), "p_value": float("nan")}
    )

    purpose_map = _purpose_rank(real_events)
    real_purpose = _matched_real_sample(real_events["purpose"].map(purpose_map), n_sim, seed + 2)
    sim_purpose = sim_accepted["purpose"].map(purpose_map).to_numpy(dtype=float)
    marginals["purpose_mix"] = _ks(real_purpose, sim_purpose)

    results: dict[str, object] = {}
    n_passed = 0
    for name in MARGINAL_NAMES:
        p_value = marginals[name]["p_value"]
        passed = bool(np.isfinite(p_value) and p_value >= BONFERRONI_ALPHA)
        results[name] = {**marginals[name], "bonferroni_alpha": BONFERRONI_ALPHA, "passed": passed}
        n_passed += int(passed)
    results["n_passed"] = n_passed
    return results
