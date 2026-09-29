"""Tests for simulate.py (statistical mode) and the screening pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rider_sim.eval.traces as eval_traces
import rider_sim.screening.report as report_module
import rider_sim.screening.runner as screening_runner
from rider_sim.eval.traces import TRACE_COLUMNS, save_trace
from rider_sim.personas import RiderPersona, personas_to_frame
from rider_sim.screening.backtest import run_backtest
from rider_sim.screening.runner import run_screening
from rider_sim.simulate import simulate

ZONES = [1, 132, 138, 230, 186, 100]


def make_persona(rider_id: int) -> RiderPersona:
    return RiderPersona(
        rider_id=rider_id,
        home_zone=230,
        work_zone=186,
        trip_purpose_mix={"commute": 0.5, "social": 0.2, "errand": 0.2, "airport": 0.1},
        price_sensitivity=0.5,
        wait_tolerance_minutes=6.0,
        income_bracket="middle",
        has_transit_alternative=False,
        loyalty_tier="gold",
        observed_trip_count=12,
        median_fare_paid=22.0,
        median_wait_experienced=4.0,
    )


@pytest.fixture()
def trips_fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(2)
    rows = []
    for rider_id in range(10):
        for _ in range(8):
            miles = float(rng.uniform(1.0, 10.0))
            fare = miles * float(rng.uniform(2.5, 8.0))
            rows.append(
                {
                    "rider_id": rider_id,
                    "pickup_zone": 230,
                    "dropoff_zone": 186,
                    "pickup_hour": int(rng.integers(0, 24)),
                    "is_weekend": 0,
                    "miles_bucket": 1,
                    "base_passenger_fare": fare,
                    "tips": 0.0,
                    "trip_miles": miles,
                    "trip_time": miles * 180.0,
                    "wait_secs": float(rng.uniform(60.0, 900.0)),
                    "driver_pay": fare * 0.7,
                    "fare_paid": fare,
                    "surge_ratio": float(rng.uniform(1.0, 2.5)),
                }
            )
    path = tmp_path / "trips.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _personas_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "personas.parquet"
    personas_to_frame([make_persona(i) for i in range(10)]).to_parquet(path, index=False)
    return path


def test_simulate_statistical_mode(tmp_path: Path, trips_fixture: Path) -> None:
    personas_path = _personas_fixture(tmp_path)
    output = tmp_path / "simulated_choices.parquet"
    frame = simulate(
        n=5,
        seed=7,
        mode="statistical",
        trips_per_persona=2,
        personas_path=personas_path,
        trips_path=trips_fixture,
        output_path=output,
    )
    assert output.exists()
    assert len(frame) == 5 * 2 * 5  # 2 trips x 5 offer variants
    assert set(frame["decision_mode"]) == {"statistical"}
    assert frame["accepted"].isin([0, 1]).all()


def test_simulate_pick_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    import rider_sim.simulate as simulate_module
    from rider_sim.simulate import pick_provider

    monkeypatch.setattr(simulate_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert pick_provider("statistical") is None
    assert pick_provider("auto") is None
    with pytest.raises(ValueError, match="llm"):
        pick_provider("llm")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert pick_provider("auto") == "openai"
    assert pick_provider("llm") == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert pick_provider("auto") == "anthropic"


def _synthetic_trace(run_id: str, config: str, n_riders: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for rider in range(n_riders):
        for variant, (price_mult, surge_mult) in enumerate(
            [(1.0, 1.0), (0.8, 1.0), (1.25, 1.0), (1.8, 1.0), (1.0, 2.0), (1.25, 2.0)]
        ):
            for trip in range(3):
                surge = 1.1 * surge_mult
                p_accept = max(0.1, 0.8 - 0.25 * (surge - 1.0))
                rows.append(
                    {
                        "run_id": run_id,
                        "config": config,
                        "rider_id": rider,
                        "offer_id": f"r{rider}t{trip}v{variant}",
                        "origin_zone": 230,
                        "dest_zone": 186,
                        "purpose": "commute",
                        "pickup_hour": 8,
                        "quoted_fare": 22.0 * price_mult,
                        "surge": surge,
                        "eta_min": 5.0,
                        "eta_framing": "numeric",
                        "discount_pct": 0.0,
                        "time_of_day": "morning",
                        "weather": "clear",
                        "transit_alt_minutes": 30.0,
                        "price_mult": price_mult,
                        "surge_mult": surge_mult,
                        "is_base": price_mult == 1.0 and surge_mult == 1.0,
                        "reject_counterfactual": price_mult == 1.8 and surge_mult == 1.0,
                        "action": "accept",
                        "accepted": int(rng.random() < p_accept),
                        "confidence": 0.7,
                        "reservation_fare": 22.0,
                        "p_accept": p_accept,
                        "price_sensitivity": 0.5,
                        "wait_tolerance_minutes": 6.0,
                        "has_transit_alternative": False,
                        "income_bracket": "middle",
                        "loyalty_tier": "gold",
                        "cost_usd": 0.0,
                        "prompt_version": "rider_agent.v1",
                    }
                )
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def test_backtest_pass_when_sim_matches_real() -> None:
    rng = np.random.default_rng(3)
    trips_rows = []
    for _ in range(4000):
        surge = float(rng.choice([1.05, 1.05, 1.05, 2.5]))
        trips_rows.append({"surge_ratio": surge})
    trips = pd.DataFrame(trips_rows)
    # real: high band share 0.25, low band share 0.75 -> log(1/3) < 0

    trace_rows = []
    for _ in range(600):
        trace_rows.append({"rider_id": 0, "surge": 2.5, "accepted": int(rng.random() < 0.2)})
    for _ in range(600):
        trace_rows.append({"rider_id": 0, "surge": 1.05, "accepted": int(rng.random() < 0.6)})
    trace = pd.DataFrame(trace_rows)

    result = run_backtest(trace, trips, n_bootstrap=100, seed=0)
    assert result["direction_agrees"]
    assert result["verdict"] in {"PASS", "PARTIAL"}


def test_backtest_direction_mismatch_fails() -> None:
    trips = pd.DataFrame({"surge_ratio": [1.05] * 60 + [2.5] * 20})
    # real: high share 0.25, low 0.75 -> negative log ratio
    trace = pd.DataFrame(
        {
            "rider_id": [0] * 60,
            "surge": [1.05] * 20 + [2.5] * 40,
            "accepted": [0] * 20 + [1] * 40,  # sim accepts MORE at high surge
        }
    )
    result = run_backtest(trace, trips, n_bootstrap=100, seed=0)
    assert result["direction_agrees"] is False
    assert result["verdict"] == "FAIL"


def test_run_screening_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trace = _synthetic_trace("srun", "full")
    monkeypatch.setattr(eval_traces, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(screening_runner, "TRACES_DIR", tmp_path / "traces")
    save_trace("srun", "full", trace.to_dict("records"))

    trips_path = tmp_path / "trips.parquet"
    pd.DataFrame({"surge_ratio": [1.05] * 600 + [2.5] * 200}).to_parquet(trips_path, index=False)

    monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        screening_runner,
        "power_analysis",
        lambda lift, control_rate, month: {
            "control_rate": control_rate,
            "treatment_rate": control_rate + lift,
            "effect_size": lift,
            "n_per_arm": 1234,
            "n_total": 2468,
            "daily_offers": 500_000,
            "days": 0.01,
        },
    )
    report_path = run_screening(
        run_id="srun",
        with_backtest=True,
        config="full",
        trips_path=trips_path,
        seed=7,
        n_boot_riders=50,
        n_boot_samples=30,
    )
    text = report_path.read_text()
    assert "Ranked intervention cells" in text
    assert "screen in" in text or "screen out" in text
    assert "Back-test" in text
