"""Tests for the fidelity gate and remaining small units."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rider_sim.agent.schemas import DecisionParseError, extract_json_object, parse_decision
from rider_sim.agent.tools import EtaReliabilityTool, ToolRegistry, TransitAlternativeTool
from rider_sim.gate import check_fidelity, load_json, main


def _metrics(configs: dict[str, dict[str, object]]) -> dict[str, object]:
    return {"configs": configs}


def _full_row(auc: float = 1.0, ks: int = 2, sign: str = "negative") -> dict[str, object]:
    return {
        "discriminator_auc": auc,
        "ks_passed": ks,
        "elasticity_sign": sign,
    }


def _baselines() -> dict[str, object]:
    return {
        "discriminator_auc": 1.0,
        "discriminator_auc_tolerance": 0.02,
        "min_ks_marginals_passed": 2,
        "elasticity_sign": "negative",
    }


def test_gate_passes_on_baseline() -> None:
    passed, rows = check_fidelity(_metrics({"full": _full_row()}), _baselines())
    assert passed
    assert len(rows) == 3


def test_gate_fails_on_auc_drift() -> None:
    passed, rows = check_fidelity(_metrics({"full": _full_row(auc=0.8)}), _baselines())
    assert not passed
    assert rows[0]["passed"] is False


def test_gate_fails_on_ks_drop() -> None:
    passed, rows = check_fidelity(_metrics({"full": _full_row(ks=1)}), _baselines())
    assert not passed
    assert rows[1]["passed"] is False


def test_gate_fails_on_sign_flip() -> None:
    passed, rows = check_fidelity(_metrics({"full": _full_row(sign="non_negative")}), _baselines())
    assert not passed
    assert rows[2]["passed"] is False


def test_gate_main_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import rider_sim.gate as gate_module

    metrics_path = tmp_path / "metrics.json"
    baselines_path = tmp_path / "baselines.json"
    metrics_path.write_text(json.dumps(_metrics({"full": _full_row()})))
    baselines_path.write_text(json.dumps(_baselines()))
    monkeypatch.setattr(gate_module, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(gate_module, "BASELINES_PATH", baselines_path)
    assert main() == 0

    metrics_path.write_text(json.dumps(_metrics({"full": _full_row(auc=0.5)})))
    assert main() == 1

    monkeypatch.setattr(gate_module, "METRICS_PATH", tmp_path / "missing.json")
    assert main() == 2


def test_gate_load_json_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "nope.json")


def test_parse_decision_fenced_and_errors() -> None:
    fenced = (
        '```json\n{"action": "reject", "reasoning": "too expensive", '
        '"confidence": 0.9, "reservation_fare": 10.0}\n```'
    )
    decision = parse_decision(fenced)
    assert decision.action == "reject"

    with pytest.raises(DecisionParseError):
        parse_decision("no json here at all")
    with pytest.raises(DecisionParseError):
        parse_decision('{"action": "fly", "reasoning": "x", "confidence": 0.5}')
    with pytest.raises(DecisionParseError):
        extract_json_object("still no object")


def test_eta_and_transit_tools(tmp_path: Path) -> None:
    import pandas as pd

    trips = pd.DataFrame(
        {
            "pickup_zone": [230] * 120,
            "pickup_hour": [8] * 120,
            "dropoff_zone": [186] * 120,
            "trip_miles": [3.0] * 120,
            "wait_secs": [300.0] * 120,
        }
    )
    path = tmp_path / "trips.parquet"
    trips.to_parquet(path, index=False)

    eta = EtaReliabilityTool(trips_path=path).call(origin_zone=230, hour=8)
    assert eta["scope"] == "zone_hour"
    assert eta["wait_p50_minutes"] == 5.0

    transit = TransitAlternativeTool(trips_path=path).call(origin_zone=230, dest_zone=186)
    assert transit["transit_minutes"] == 15.0
    assert transit["transit_cost"] == 2.9

    registry = ToolRegistry([EtaReliabilityTool(trips_path=path)])
    assert "nope" not in registry
    assert registry.call("nope", {}) == {"error": "unknown tool 'nope'"}
    result = registry.call("check_eta_reliability", {"bogus_arg": 8})
    assert "error" in result
