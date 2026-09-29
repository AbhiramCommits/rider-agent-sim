"""Tests for the LLM rider agent: tools, memory, repair, and cache."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rider_sim.agent.llm import LLMClient, LLMRequest, LLMResponse, ToolCall
from rider_sim.agent.memory import RiderMemory
from rider_sim.agent.policy import LLMRiderAgent, LogitRiderAgent, RandomRiderAgent
from rider_sim.agent.schemas import Decision, RideOffer
from rider_sim.personas import RiderPersona

ACCEPT_JSON = json.dumps(
    {
        "action": "accept",
        "reasoning": "Fare is below the rider's usual range and the wait is short.",
        "confidence": 0.87,
        "reservation_fare": 27.5,
        "tools_called": [],
    }
)


def make_persona(rider_id: int = 7) -> RiderPersona:
    return RiderPersona(
        rider_id=rider_id,
        home_zone=230,
        work_zone=186,
        trip_purpose_mix={"commute": 0.5, "social": 0.2, "errand": 0.2, "airport": 0.1},
        price_sensitivity=0.6,
        wait_tolerance_minutes=8.0,
        income_bracket="middle",
        has_transit_alternative=False,
        loyalty_tier="gold",
        observed_trip_count=12,
        median_fare_paid=22.0,
        median_wait_experienced=4.0,
    )


def make_offer(offer_id: str = "o1") -> RideOffer:
    return RideOffer(
        offer_id=offer_id,
        origin_zone=230,
        dest_zone=186,
        trip_purpose="commute",
        quoted_fare=21.5,
        surge_multiplier=1.2,
        eta_minutes=6.0,
        eta_framing="numeric",
        discount_pct=0.0,
        time_of_day="morning",
        weather="clear",
        transit_alt_minutes=28.0,
    )


@pytest.fixture()
def trips_fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    rows: list[dict[str, float]] = []
    for rider_id in range(4):
        for _ in range(15):
            miles = float(rng.uniform(1.0, 10.0))
            fare = miles * float(rng.uniform(2.5, 8.0))
            rows.append(
                {
                    "rider_id": rider_id,
                    "pickup_zone": 230 if rider_id % 2 == 0 else 100,
                    "dropoff_zone": 186,
                    "pickup_hour": int(rng.integers(0, 24)),
                    "is_weekend": 0,
                    "miles_bucket": 1,
                    "base_passenger_fare": fare,
                    "tips": 0.0,
                    "trip_miles": miles,
                    "trip_time": miles * 180.0,
                    "wait_secs": float(rng.uniform(60, 900)),
                    "driver_pay": fare * 0.7,
                    "fare_paid": fare,
                    "surge_ratio": float(rng.uniform(1.0, 2.0)),
                }
            )
    path = tmp_path / "trips.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class ScriptedBackend:
    """Fake provider: returns queued responses, records every request."""

    name = "fake"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError("scripted backend ran out of responses")
        return self.responses.pop(0)


def test_tool_calls_fire_and_decision_parses(tmp_path: Path, trips_fixture: Path) -> None:
    tool_call_response = LLMResponse(
        content="",
        tool_calls=(
            ToolCall(
                id="tc1",
                name="check_price_history",
                arguments={"origin_zone": 230, "dest_zone": 186},
            ),
        ),
    )
    backend = ScriptedBackend([tool_call_response, LLMResponse(content=ACCEPT_JSON)])
    client = LLMClient(backend=backend, cache_db=tmp_path / "cache.duckdb")
    agent = LLMRiderAgent(
        persona=make_persona(),
        memory=RiderMemory(rider_id=7),
        client=client,
        provider="anthropic",
        trips_path=trips_fixture,
        seed=1,
    )

    decision = agent.decide(make_offer())

    assert decision.action == "accept"
    assert decision.tools_called == ["check_price_history"]
    assert decision.confidence == 0.87
    assert len(backend.calls) == 2
    tool_message = backend.calls[1].messages[-1]
    assert tool_message.role == "tool"
    assert json.loads(tool_message.content)["tool"] == "check_price_history"
    result = json.loads(tool_message.content)["result"]
    assert result["scope"] == "lane"
    assert "market_fare_p50" in result


def test_memory_grows_and_aggregates_update(tmp_path: Path, trips_fixture: Path) -> None:
    backend = ScriptedBackend([LLMResponse(content=ACCEPT_JSON), LLMResponse(content=ACCEPT_JSON)])
    client = LLMClient(backend=backend, cache_db=tmp_path / "cache.duckdb")
    memory = RiderMemory(rider_id=7)
    agent = LLMRiderAgent(
        persona=make_persona(),
        memory=memory,
        client=client,
        provider="anthropic",
        trips_path=trips_fixture,
        seed=2,
    )

    offer = make_offer("o1")
    first = agent.decide(offer)
    memory.record_realization("o1", realized_fare=21.0, realized_wait=7.0, satisfaction_delta=-0.1)
    second = agent.decide(make_offer("o2"))

    assert first.action == "accept" and second.action == "accept"
    assert len(memory) == 2
    aggregates = memory.aggregates()
    assert aggregates["episodes_total"] == 2
    assert aggregates["mean_fare_paid"] == 21.0
    assert aggregates["bad_wait_experiences"] == 0

    recall = memory.recall(make_offer("o3"), k=1)
    assert recall["aggregates"]["episodes_total"] == 2
    assert len(recall["relevant_episodes"]) == 1


def test_malformed_json_is_repaired(tmp_path: Path, trips_fixture: Path) -> None:
    backend = ScriptedBackend(
        [LLMResponse(content="not json at all"), LLMResponse(content=ACCEPT_JSON)]
    )
    client = LLMClient(backend=backend, cache_db=tmp_path / "cache.duckdb")
    agent = LLMRiderAgent(
        persona=make_persona(),
        memory=RiderMemory(rider_id=7),
        client=client,
        provider="anthropic",
        trips_path=trips_fixture,
        seed=3,
    )

    decision = agent.decide(make_offer())

    assert decision.action == "accept"
    assert len(backend.calls) == 2
    assert "not valid JSON" in backend.calls[1].messages[-1].content
    assert agent.parse_failures == []


def test_unrepairable_output_records_parse_failure(tmp_path: Path, trips_fixture: Path) -> None:
    backend = ScriptedBackend(
        [
            LLMResponse(content="garbage one"),
            LLMResponse(content="still garbage"),
        ]
    )
    client = LLMClient(backend=backend, cache_db=tmp_path / "cache.duckdb")
    agent = LLMRiderAgent(
        persona=make_persona(),
        memory=RiderMemory(rider_id=7),
        client=client,
        provider="anthropic",
        trips_path=trips_fixture,
        seed=4,
    )

    decision = agent.decide(make_offer())

    assert decision.action == "reject"
    assert decision.confidence == 0.0
    assert len(agent.parse_failures) == 1
    assert agent.parse_failures[0]["offer_id"] == "o1"


def test_cache_prevents_second_api_call(tmp_path: Path, trips_fixture: Path) -> None:
    backend = ScriptedBackend([LLMResponse(content=ACCEPT_JSON)])
    client = LLMClient(backend=backend, cache_db=tmp_path / "cache.duckdb")

    async def run() -> None:
        for _ in range(2):
            agent = LLMRiderAgent(
                persona=make_persona(),
                memory=RiderMemory(rider_id=7),
                client=client,
                provider="anthropic",
                trips_path=trips_fixture,
                seed=5,
            )
            await agent._decide_async(make_offer("o1"))

    asyncio.run(run())

    assert len(backend.calls) == 1
    assert client.cache_stats() == (1, 2)


def test_baselines_share_interface(tmp_path: Path, trips_fixture: Path) -> None:
    offer = make_offer()
    logit = LogitRiderAgent(persona=make_persona(), trips_path=trips_fixture, seed=0)
    random_agent = RandomRiderAgent(seed=0)

    for agent in (logit, random_agent):
        decision = agent.decide(offer)
        assert isinstance(decision, Decision)
        assert decision.action in {"accept", "reject", "wait_for_better", "switch_mode"}
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.tools_called == []

    assert RandomRiderAgent(seed=1).decide(offer) == RandomRiderAgent(seed=1).decide(offer)


def test_offer_schema_strict() -> None:
    with pytest.raises(ValueError):
        RideOffer(
            offer_id="bad",
            origin_zone=230,
            dest_zone=186,
            trip_purpose="commute",
            quoted_fare="21",  # strict: string rejected for float
            surge_multiplier=1.0,
            eta_minutes=5.0,
            eta_framing="numeric",
            discount_pct=0.0,
            time_of_day="morning",
            weather="clear",
            transit_alt_minutes=30.0,
        )
    with pytest.raises(ValueError):
        RideOffer.model_validate({**make_offer().model_dump(), "eta_framing": "vague"})


def test_decision_rejects_long_reasoning() -> None:
    with pytest.raises(ValueError):
        Decision(
            action="accept",
            reasoning=" ".join(["word"] * 61),
            confidence=0.9,
            reservation_fare=10.0,
        )
