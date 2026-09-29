"""Tests for the fidelity evaluation package with known ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rider_sim.eval.ablations import auc_difference_permutation_test
from rider_sim.eval.calibration import run_calibration
from rider_sim.eval.discriminator import (
    FEATURE_NAMES,
    run_discriminator,
    sequence_features,
)
from rider_sim.eval.distributions import run_distribution_tests
from rider_sim.eval.mechanism import surge_elasticity
from rider_sim.eval.runner import run_evaluation
from rider_sim.personas import RiderPersona, personas_to_frame

ZONES = [1, 132, 138, 230, 186, 100]
PURPOSES = ["commute", "social", "errand", "airport"]


@pytest.fixture()
def trips_fixture(tmp_path: Path) -> Path:
    """Synthetic real trips: 60 riders x 12 trips with varied marginals."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, float]] = []
    for rider_id in range(60):
        for _ in range(12):
            miles = float(rng.uniform(1.0, 12.0))
            fare = miles * float(rng.uniform(2.5, 8.0))
            rows.append(
                {
                    "rider_id": rider_id,
                    "pickup_zone": int(rng.choice(ZONES)),
                    "dropoff_zone": int(rng.choice(ZONES)),
                    "pickup_hour": int(rng.integers(0, 24)),
                    "is_weekend": int(rng.integers(0, 2)),
                    "miles_bucket": 1,
                    "base_passenger_fare": fare,
                    "tips": 0.0,
                    "trip_miles": miles,
                    "trip_time": miles * 180.0,
                    "wait_secs": float(rng.uniform(60.0, 900.0)),
                    "driver_pay": fare * 0.7,
                    "fare_paid": fare,
                    "surge_ratio": float(rng.uniform(1.0, 2.2)),
                }
            )
    path = tmp_path / "trips.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture()
def personas_fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    personas = []
    for rider_id in range(60):
        personas.append(
            RiderPersona(
                rider_id=rider_id,
                home_zone=int(rng.choice(ZONES)),
                work_zone=int(rng.choice(ZONES)),
                trip_purpose_mix={"commute": 0.5, "social": 0.2, "errand": 0.2, "airport": 0.1},
                price_sensitivity=round(float(rng.uniform(0.1, 0.9)), 3),
                wait_tolerance_minutes=round(float(rng.uniform(2.0, 12.0)), 2),
                income_bracket="middle",
                has_transit_alternative=bool(rng.integers(0, 2)),
                loyalty_tier="gold",
                observed_trip_count=12,
                median_fare_paid=round(float(rng.uniform(15.0, 40.0)), 2),
                median_wait_experienced=round(float(rng.uniform(2.0, 8.0)), 2),
            )
        )
    path = tmp_path / "personas.parquet"
    personas_to_frame(personas).to_parquet(path, index=False)
    return path


def make_events(
    rng: np.random.Generator,
    n_riders: int,
    trips_per_rider: int,
    fare_scale: float = 1.0,
    accept_rate: float | None = None,
) -> pd.DataFrame:
    rows = []
    for rider_id in range(n_riders):
        for _ in range(trips_per_rider):
            accepted = 1 if accept_rate is None or rng.random() < accept_rate else 0
            rows.append(
                {
                    "rider_id": rider_id,
                    "quoted_fare": float(rng.uniform(8.0, 60.0)) * fare_scale,
                    "surge": float(rng.uniform(1.0, 2.5)),
                    "eta_min": float(rng.uniform(2.0, 15.0)),
                    "purpose": str(rng.choice(PURPOSES)),
                    "action": "accept" if accepted else "reject",
                    "accepted": accepted,
                    "pickup_hour": int(rng.integers(0, 24)),
                }
            )
    return pd.DataFrame(rows)


def test_discriminator_identical_distributions_auc_ci_covers_05() -> None:
    """Identical distributions: real/sim labels are exchangeable, so the
    discriminator cannot separate them and the AUC CI covers 0.5."""
    rng = np.random.default_rng(0)
    events = make_events(rng, 120, 12)
    market_fares = np.linspace(5.0, 70.0, 500)
    feats = sequence_features(events, market_fares, seed=0)

    order = np.random.default_rng(9).permutation(len(feats))
    real_feats = feats.iloc[order[:60]].reset_index(drop=True)
    sim_feats = feats.iloc[order[60:]].reset_index(drop=True)
    result = run_discriminator(real_feats, sim_feats, n_bootstrap=300, seed=0)

    assert set(FEATURE_NAMES).issubset(real_feats.columns)
    ci_low, ci_high = result["auc_ci"]
    assert ci_low <= 0.5 <= ci_high
    assert set(result["importances"]) == set(FEATURE_NAMES)


def test_distribution_tests_shifted_sim_is_rejected() -> None:
    real_rng = np.random.default_rng(0)
    sim_rng = np.random.default_rng(2)
    real = make_events(real_rng, 40, 10)
    sim = make_events(sim_rng, 40, 10, fare_scale=2.5, accept_rate=0.8)

    results = run_distribution_tests(sim, real, seed=0)
    fare_test = results["accepted_fare"]
    assert fare_test["p_value"] < fare_test["bonferroni_alpha"]
    assert fare_test["passed"] is False
    wait_test = results["accepted_wait"]
    assert wait_test["passed"] is True  # waits are identical across fixtures


def test_distribution_tests_identical_is_accepted() -> None:
    real_rng = np.random.default_rng(3)
    sim_rng = np.random.default_rng(4)
    real = make_events(real_rng, 40, 10)
    sim = make_events(sim_rng, 40, 10)

    results = run_distribution_tests(sim, real, seed=0)
    for name in ("accepted_fare", "accepted_wait"):
        assert results[name]["passed"] is True
    assert results["n_passed"] >= 2


def test_calibration_random_is_worse_than_calibrated() -> None:
    n = 400
    offers = pd.DataFrame(
        {
            "is_base": [True] * n + [False] * (n // 2),
            "reject_counterfactual": [False] * n + [True] * (n // 2),
            "p_accept": [0.95] * n + [0.05] * (n // 2),
            "price_sensitivity": np.linspace(0.1, 0.9, n + n // 2),
            "wait_tolerance_minutes": np.linspace(2.0, 12.0, n + n // 2),
            "has_transit_alternative": [True, False] * ((n + n // 2) // 2),
        }
    )
    calibrated = run_calibration(offers)
    random_offers = offers.copy()
    random_offers["p_accept"] = 0.25
    random_result = run_calibration(random_offers)

    assert calibrated["brier"] < 0.05
    assert random_result["brier"] > calibrated["brier"]
    curves = calibrated["curves"]
    assert "overall" in curves
    assert any(key.startswith("price_sensitivity=") for key in curves)
    assert any(key.startswith("has_transit_alternative=") for key in curves)


def _trace_with_elasticity(slope: float, seed: int) -> pd.DataFrame:
    """Synthetic trace whose accept rate follows rate = 0.9 * surge^slope."""
    rng = np.random.default_rng(seed)
    rows = []
    for surge in (1.0, 1.5, 2.0, 2.5, 3.0):
        rate = 0.9 * surge**slope
        n = 600
        accepted = rng.random(n) < rate
        for value in accepted:
            rows.append(
                {
                    "surge": surge,
                    "accepted": int(value),
                    "is_base": surge == 1.0,
                    "reject_counterfactual": False,
                }
            )
    return pd.DataFrame(rows)


def test_mechanism_elasticity_in_literature_range_passes() -> None:
    trace = _trace_with_elasticity(slope=-0.5, seed=0)
    result = surge_elasticity(trace, n_bootstrap=100, seed=0)
    assert result["sign_pass"] is True
    assert result["magnitude_pass"] is True
    assert result["verdict"] == "PASS"
    assert -0.65 < result["elasticity"] < -0.35


def test_mechanism_flat_demand_fails_sign() -> None:
    trace = _trace_with_elasticity(slope=0.0, seed=1)
    result = surge_elasticity(trace, n_bootstrap=100, seed=0)
    assert result["sign_pass"] is False
    assert result["verdict"] == "FAIL"


def test_permutation_test_detects_auc_differences() -> None:
    rng = np.random.default_rng(0)
    real_scores = rng.normal(0.5, 0.1, 100)
    sim_identical = rng.normal(0.5, 0.1, 100)
    sim_separated = rng.normal(0.9, 0.1, 100)

    p_identical = auc_difference_permutation_test(
        real_scores, sim_identical, sim_identical.copy(), n_permutations=100, seed=0
    )
    assert p_identical > 0.05

    p_separated = auc_difference_permutation_test(
        real_scores, sim_identical, sim_separated, n_permutations=100, seed=0
    )
    assert p_separated < 0.05


def test_end_to_end_evaluation(
    tmp_path: Path, trips_fixture: Path, personas_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rider_sim.eval.report as report_module
    import rider_sim.eval.traces as traces_module

    traces_dir = tmp_path / "traces"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(traces_module, "TRACES_DIR", traces_dir)
    monkeypatch.setattr(report_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(report_module, "FIGURES_DIR", reports_dir / "figures")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report_path = run_evaluation(
        run_id="testrun",
        ablations="all",
        n_riders=8,
        offers_per_rider=1,
        seed=7,
        provider="offline",
        trips_path=trips_fixture,
        personas_path=personas_fixture,
        cache_db=None,
        n_bootstrap=50,
        n_permutations=50,
    )
    assert report_path.exists()
    text = report_path.read_text()
    assert "Ablation table" in text and "Mechanism" in text
    figures_dir = reports_dir / "figures" / "testrun"
    assert (figures_dir / "ablations.png").exists()
    assert (figures_dir / "distributions.png").exists()
    assert len(list((traces_dir / "testrun").glob("*.parquet"))) == 7
