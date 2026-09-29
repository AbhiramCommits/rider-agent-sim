"""Pre-experiment deliverable: sample size and days for a real A/B test.

Given the simulated effect size (accept-rate lift from screening) and real
traffic assumptions (daily offers computed from the raw TLC month), returns
the per-arm sample size from the closed-form two-proportion formula at the
requested power and alpha, and the number of days a real test would need.
"""

from __future__ import annotations

import calendar
import math

import duckdb
from scipy.stats import norm

from rider_sim.data.fetch import fetch_trips


def sample_size_two_proportions(
    p0: float, p1: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Per-arm sample size (closed-form two-proportion test)."""
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_power = float(norm.ppf(power))
    p_bar = (p0 + p1) / 2.0
    numerator: float = (
        z_alpha * (2.0 * p_bar * (1.0 - p_bar)) ** 0.5
        + z_power * (p0 * (1.0 - p0) + p1 * (1.0 - p1)) ** 0.5
    ) ** 2
    denominator: float = (p1 - p0) ** 2
    if denominator <= 0:
        raise ValueError("effect size is zero; no finite sample size exists")
    return math.ceil(numerator / denominator)


def daily_offers(month: str) -> int:
    """Real traffic: mean daily HVFHV trip count for the month."""
    raw = fetch_trips(month)
    con = duckdb.connect()
    try:
        row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(raw)]).fetchone()
    finally:
        con.close()
    count = int(row[0]) if row is not None else 0
    year, month_num = (int(part) for part in month.split("-"))
    days = calendar.monthrange(year, month_num)[1]
    return count // days


def days_required(n_per_arm: int, daily: int, traffic_share: float = 0.5) -> float:
    """Days for a real test, assuming ``traffic_share`` of daily offers per arm."""
    return n_per_arm / (daily * traffic_share)


def power_analysis(
    lift: float,
    control_rate: float,
    month: str,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, object]:
    """Sample size + days needed to detect ``lift`` vs ``control_rate``."""
    p1 = min(max(control_rate + lift, 1e-6), 1.0 - 1e-6)
    n_per_arm = sample_size_two_proportions(control_rate, p1, alpha=alpha, power=power)
    daily = daily_offers(month)
    days = days_required(n_per_arm, daily)
    return {
        "control_rate": control_rate,
        "treatment_rate": p1,
        "effect_size": p1 - control_rate,
        "n_per_arm": n_per_arm,
        "n_total": 2 * n_per_arm,
        "daily_offers": daily,
        "days": round(days, 2),
        "alpha": alpha,
        "power": power,
    }
