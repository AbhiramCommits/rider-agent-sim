"""Screening tests with known ground truth."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rider_sim.screening.power import days_required, sample_size_two_proportions
from rider_sim.screening.screen import ScreeningCell, screen_cell


def _synthetic_trace(
    rng: np.random.Generator,
    n_riders: int = 60,
    offers_per_cell: int = 10,
    control_p: float = 0.5,
    lift: float = 0.1,
    heterogeneity: float = 0.0,
    price_mult: float = 0.8,
    fare_scale: float = 0.8,
) -> pd.DataFrame:
    """Trace with a known injected accept-rate lift in the intervention cell.

    ``heterogeneity`` adds an interaction: intervention p = control + lift +
    heterogeneity * (control - mean(control)), so riders with higher baseline
    acceptance respond more -- the CUPED covariate becomes predictive.
    """
    rows = []
    mean_control = control_p
    for rider in range(n_riders):
        rider_control = float(rng.uniform(0.25, 0.75)) if heterogeneity else control_p
        rider_intervention = min(
            0.99, max(0.01, rider_control + lift + heterogeneity * (rider_control - mean_control))
        )
        base_fare = float(rng.uniform(15.0, 35.0))
        for i in range(offers_per_cell):
            rows.append(
                {
                    "rider_id": rider,
                    "offer_id": f"c{i:03d}",
                    "price_mult": 1.0,
                    "surge_mult": 1.0,
                    "p_accept": rider_control,
                    "quoted_fare": base_fare,
                    "surge": 1.1,
                    "accepted": int(rng.random() < rider_control),
                }
            )
            rows.append(
                {
                    "rider_id": rider,
                    "offer_id": f"t{i:03d}",
                    "price_mult": price_mult,
                    "surge_mult": 1.0,
                    "p_accept": rider_intervention,
                    "quoted_fare": base_fare * fare_scale,
                    "surge": 1.1,
                    "accepted": int(rng.random() < rider_intervention),
                }
            )
    return pd.DataFrame(rows)


def test_screen_recovers_injected_lift_inside_ci() -> None:
    rng = np.random.default_rng(3)
    trace = _synthetic_trace(
        rng, n_riders=80, offers_per_cell=16, control_p=0.5, lift=0.1, price_mult=0.8
    )
    cell = ScreeningCell("discount_20pct", 0.8, 1.0, "test")
    result = screen_cell(trace, cell, seed=7, n_boot_riders=400, n_boot_samples=100)

    primary = result["metrics"]["accept_rate"]
    assert abs(primary["lift"] - 0.1) < 0.05
    ci_low, ci_high = primary["ci"]
    assert ci_low <= 0.1 <= ci_high
    assert ci_low > 0.0  # effect is strong enough to screen in
    assert primary["se_rider"] > 0 and primary["se_sampling"] > 0

    fare = result["metrics"]["mean_fare"]
    assert np.isfinite(fare["lift"])
    assert fare["lift"] < 0.0  # discounted cell accepted fares are lower

    completed = result["metrics"]["completed_rides"]
    assert completed["lift"] > 0
    abandonment = result["metrics"]["abandonment_rate"]
    assert abs(abandonment["lift"] + primary["lift"]) < 1e-9


def test_screen_null_cell_lift_is_zero_within_ci() -> None:
    rng = np.random.default_rng(4)
    trace = _synthetic_trace(
        rng, n_riders=60, offers_per_cell=16, control_p=0.5, lift=0.0, price_mult=0.8
    )
    cell = ScreeningCell("discount_20pct", 0.8, 1.0, "test")
    result = screen_cell(trace, cell, seed=7, n_boot_riders=300, n_boot_samples=80)
    primary = result["metrics"]["accept_rate"]
    ci_low, ci_high = primary["ci"]
    assert ci_low <= 0.0 <= ci_high


def test_cuped_reduces_variance_with_correlated_deltas() -> None:
    rng = np.random.default_rng(5)
    trace = _synthetic_trace(
        rng,
        n_riders=60,
        offers_per_cell=60,
        control_p=0.5,
        lift=0.1,
        heterogeneity=1.5,
        price_mult=0.8,
    )
    cell = ScreeningCell("discount_20pct", 0.8, 1.0, "test")
    result = screen_cell(trace, cell, seed=7, n_boot_riders=300, n_boot_samples=60)
    primary = result["metrics"]["accept_rate"]
    assert np.isfinite(primary["cuped_var_reduction"])
    assert primary["cuped_var_reduction"] > 0.3
    assert abs(primary["lift_cuped"] - 0.1) < 0.05


def test_power_matches_closed_form_two_proportion() -> None:
    # Standard result: p0=0.5, p1=0.55, alpha=0.05, power=0.8 -> n=1565 per arm.
    n_per_arm = sample_size_two_proportions(0.5, 0.55, alpha=0.05, power=0.8)
    assert n_per_arm == 1565

    # Closed-form cross-check with explicit z-scores.
    from scipy.stats import norm

    z_alpha = norm.ppf(0.975)
    z_power = norm.ppf(0.8)
    p_bar = 0.525
    expected = (
        z_alpha * (2 * p_bar * (1 - p_bar)) ** 0.5 + z_power * (0.5 * 0.5 + 0.55 * 0.45) ** 0.5
    ) ** 2 / 0.05**2
    assert math.isclose(n_per_arm, math.ceil(expected))

    # Days: 1565 per arm at 500 offers/day per arm -> 3.13 days.
    assert math.isclose(days_required(1565, 1000, traffic_share=0.5), 3.13)

    # Zero effect has no finite sample size.
    try:
        sample_size_two_proportions(0.5, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("zero effect should raise")
