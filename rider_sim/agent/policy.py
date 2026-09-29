"""Rider agent policies: the LLM agent and two non-LLM baselines.

:class:`LLMRiderAgent` builds its prompt from the persona + recalled memory +
offer, runs up to 2 tool-call rounds, parses the model output into a
:class:`Decision`, and appends an episode to memory. Invalid model output is
rejected and retried once with a repair prompt; if it still fails, a
``parse_failure`` row is recorded and a safe fallback decision (reject) is
returned instead of crashing. Every decision row logs the prompt version.

The two baselines share the same ``decide(offer) -> Decision`` interface:

- :class:`LogitRiderAgent` -- a multinomial logit fit on real trip data.
  Real trips are all accepted by construction, so non-accept labels are
  counterfactual variants of real trips (large fare / surge perturbations
  labeled reject / wait_for_better / switch_mode). This is documented
  honestly: the accept coefficients are identified from real behavior; the
  other action classes are counterfactual constructs.
- :class:`RandomRiderAgent` -- uniform random actions (the comparison floor).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.linear_model import LogisticRegression

from rider_sim.agent.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    LLMClient,
    LLMMessage,
    LLMRequest,
    ToolSpec,
)
from rider_sim.agent.memory import RiderMemory
from rider_sim.agent.schemas import (
    Action,
    Decision,
    DecisionParseError,
    RideOffer,
    parse_decision,
)
from rider_sim.agent.tools import ToolRegistry, build_rider_tools, tool_result_message
from rider_sim.config import RIDER_AGENT_PROMPT_V1, TRIPS_PATH
from rider_sim.personas import RiderPersona

console = Console()

PROMPT_VERSION = "rider_agent.v1"
MAX_TOOL_ROUNDS = 2
ACTIONS: tuple[Action, ...] = ("accept", "reject", "wait_for_better", "switch_mode")


class RiderAgent(Protocol):
    def decide(self, offer: RideOffer) -> Decision: ...


def _default_model(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_ANTHROPIC_MODEL


class LLMRiderAgent:
    """LLM-driven rider agent with memory, tool use, repair, and logging."""

    def __init__(
        self,
        persona: RiderPersona,
        memory: RiderMemory,
        client: LLMClient,
        provider: str,
        trips_path: Path = TRIPS_PATH,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        prompt_path: Path = RIDER_AGENT_PROMPT_V1,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        memory_k: int = 5,
        tools: ToolRegistry | None = None,
        persona_payload: dict[str, Any] | None = None,
    ) -> None:
        self.persona = persona
        self.memory = memory
        self.client = client
        self.provider = provider
        self.model = model or _default_model(provider)
        self.temperature = temperature
        self.seed = seed
        self.max_tool_rounds = max_tool_rounds
        self.memory_k = memory_k
        self.tools = tools if tools is not None else build_rider_tools(persona.rider_id, trips_path)
        self.persona_payload = persona_payload
        self.prompt_template = prompt_path.read_text()
        self.decisions: list[dict[str, Any]] = []
        self.parse_failures: list[dict[str, Any]] = []

    def build_prompt(self, offer: RideOffer) -> tuple[str, str]:
        """Render the system prompt (persona + memory + offer) and user message."""
        persona_json = json.dumps(
            self.persona_payload if self.persona_payload is not None else self.persona.model_dump(),
            sort_keys=True,
        )
        context = {
            "persona_json": persona_json,
            "memory_json": json.dumps(self.memory.recall(offer, self.memory_k), sort_keys=True),
            "offer_json": json.dumps(offer.model_dump(), sort_keys=True),
        }
        system = self.prompt_template
        for key, value in context.items():
            system = system.replace("{{" + key + "}}", value)
        return system, context["offer_json"]

    def _request(self, system: str, messages: list[LLMMessage]) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            system=system,
            messages=tuple(messages),
            tools=tuple(
                ToolSpec(
                    name=s["name"], description=s["description"], input_schema=s["input_schema"]
                )
                for s in self.tools.specs
            ),
            temperature=self.temperature,
            seed=self.seed,
        )

    @staticmethod
    def _repair_prompt(raw: str, error: str) -> str:
        return (
            "Your previous response was not valid JSON matching the Decision schema.\n"
            f"Validation error: {error}\n"
            "Reply with ONLY a corrected JSON object with exactly these fields:\n"
            '{"action": "accept|reject|wait_for_better|switch_mode", '
            '"reasoning": "at most 60 words", "confidence": 0.0, '
            '"reservation_fare": 0.0, "tools_called": []}\n'
            f"Your previous response was:\n{raw[:2000]}"
        )

    @staticmethod
    def _fallback_decision(offer: RideOffer, error: str, tools_called: list[str]) -> Decision:
        return Decision(
            action="reject",
            reasoning="model output failed validation twice; recorded as parse failure",
            confidence=0.0,
            reservation_fare=round(offer.quoted_fare, 2),
            tools_called=list(tools_called),
        )

    async def _decide_async(self, offer: RideOffer) -> Decision:
        system, user_content = self.build_prompt(offer)
        messages: list[LLMMessage] = [LLMMessage(role="user", content=user_content)]
        tools_called: list[str] = []

        response = await self.client.complete(self._request(system, messages))
        rounds = 0
        while response.tool_calls and rounds < self.max_tool_rounds:
            rounds += 1
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                result = self.tools.call(call.name, call.arguments)
                tools_called.append(call.name)
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=tool_result_message(call.name, result),
                        tool_call_id=call.id,
                    )
                )
            response = await self.client.complete(self._request(system, messages))

        try:
            decision = parse_decision(response.content)
        except DecisionParseError as first_error:
            messages.append(LLMMessage(role="assistant", content=response.content))
            messages.append(
                LLMMessage(
                    role="user",
                    content=self._repair_prompt(response.content, str(first_error)),
                )
            )
            repaired = await self.client.complete(self._request(system, messages))
            try:
                decision = parse_decision(repaired.content)
            except DecisionParseError as second_error:
                self.parse_failures.append(
                    {
                        "offer_id": offer.offer_id,
                        "first_error": str(first_error),
                        "second_error": str(second_error),
                        "logged_at": dt.datetime.now().isoformat(),
                    }
                )
                decision = self._fallback_decision(offer, str(second_error), tools_called)
                parse_ok = False
            else:
                parse_ok = True
        else:
            parse_ok = True

        decision = decision.model_copy(update={"tools_called": tools_called})
        self.memory.append_decision(offer, decision.action)
        self.decisions.append(
            {
                "offer_id": offer.offer_id,
                "action": decision.action,
                "confidence": decision.confidence,
                "reservation_fare": decision.reservation_fare,
                "prompt_version": PROMPT_VERSION,
                "tools_called": decision.tools_called,
                "parse_ok": parse_ok,
                "logged_at": dt.datetime.now().isoformat(),
            }
        )
        return decision

    def decide(self, offer: RideOffer) -> Decision:
        return asyncio.run(self._decide_async(offer))


def fit_logit_model(trips_path: Path, seed: int | None) -> tuple[LogisticRegression, list[str]]:
    """Fit a multinomial logit over the 4 actions on real trips + counterfactuals.

    Every real trip is an "accept" example (all real trips were taken). For a
    subset of trips we add counterfactual variants: a big fare increase ->
    reject, a surge spike -> wait_for_better, and fast-cheap transit ->
    switch_mode. The coefficients for accept/reject are thus identified from
    real fare/eta distributions; the other classes are counterfactual
    constructs -- there is no real reject data in TLC records.
    """
    trips = pd.read_parquet(trips_path)
    rider_median_fare = trips.groupby("rider_id")["fare_paid"].transform("median")
    trips = trips.assign(
        fare_rel=trips["fare_paid"] / rider_median_fare,
        surge=trips["surge_ratio"].clip(1.0, 3.0),
        eta_min=trips["wait_secs"] / 60.0,
        transit_min=trips["trip_miles"] / 12.0 * 60.0,
        is_airport=(trips["pickup_zone"].isin([1, 132, 138]))
        | (trips["dropoff_zone"].isin([1, 132, 138])),
    )
    feature_cols = ["fare_rel", "surge", "eta_min", "transit_min", "is_airport"]

    rows: list[pd.DataFrame] = [trips.assign(action="accept")[[*feature_cols, "action"]]]
    counterfactuals = trips.sample(frac=0.5, random_state=seed) if seed is not None else trips
    rows.append(
        counterfactuals.assign(fare_rel=lambda df: df["fare_rel"] * 1.8, action="reject")[
            [*feature_cols, "action"]
        ]
    )
    rows.append(
        counterfactuals.assign(
            surge=2.5,
            fare_rel=lambda df: df["fare_rel"] * 1.3,
            eta_min=lambda df: df["eta_min"] * 1.5,
            action="wait_for_better",
        )[[*feature_cols, "action"]]
    )
    rows.append(
        counterfactuals.assign(
            surge=1.8,
            transit_min=lambda df: df["eta_min"] * 0.6,
            action="switch_mode",
        )[[*feature_cols, "action"]]
    )
    train = pd.concat(rows, ignore_index=True)
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    model.fit(train[feature_cols], train["action"])
    return model, feature_cols


class LogitRiderAgent:
    """Multinomial logit baseline fit on real (accepted) trips + counterfactuals."""

    def __init__(
        self,
        persona: RiderPersona,
        trips_path: Path = TRIPS_PATH,
        seed: int | None = None,
        model: LogisticRegression | None = None,
    ) -> None:
        self.persona = persona
        self.rng = np.random.default_rng(seed)
        if model is None:
            model, features = fit_logit_model(trips_path, seed)
        else:
            features = [str(name) for name in model.feature_names_in_]
        self.model = model
        self.features = features
        self.classes: list[str] = [str(c) for c in self.model.classes_]

    def _features(self, offer: RideOffer) -> list[float]:
        fare_rel = offer.quoted_fare / max(self.persona.median_fare_paid, 1e-6)
        transit_min = offer.transit_alt_minutes
        if transit_min <= 0:
            transit_min = offer.eta_minutes * 2.0
        is_airport = float(offer.origin_zone in (1, 132, 138) or offer.dest_zone in (1, 132, 138))
        return [
            fare_rel,
            float(min(max(offer.surge_multiplier, 1.0), 3.0)),
            offer.eta_minutes,
            transit_min,
            is_airport,
        ]

    def _feature_frame(self, offer: RideOffer) -> pd.DataFrame:
        return pd.DataFrame([self._features(offer)], columns=self.features)

    def accept_probability(self, offer: RideOffer) -> float:
        proba = self.model.predict_proba(self._feature_frame(offer))[0]
        index = self.classes.index("accept")
        return float(proba[index])

    def _reservation_fare(self, offer: RideOffer) -> float:
        lo, hi = 0.0, max(offer.quoted_fare * 10.0, 1.0)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            candidate = offer.model_copy(update={"quoted_fare": mid})
            if self.accept_probability(candidate) >= 0.5:
                lo = mid
            else:
                hi = mid
        return round(lo, 2)

    def decide(self, offer: RideOffer) -> Decision:
        proba = self.model.predict_proba(self._feature_frame(offer))[0]
        action = cast(Action, self.classes[int(self.rng.choice(len(self.classes), p=proba))])
        confidence = float(proba.max())
        reservation = self._reservation_fare(offer)
        if action == "accept" and reservation < offer.quoted_fare:
            reservation = round(offer.quoted_fare, 2)
        return Decision(
            action=action,
            reasoning=(
                f"multinomial logit baseline: accept prob "
                f"{self.accept_probability(offer):.2f}, fare {offer.quoted_fare:.2f} "
                f"vs persona median {self.persona.median_fare_paid:.2f}"
            ),
            confidence=confidence,
            reservation_fare=reservation,
            tools_called=[],
        )


class RandomRiderAgent:
    """Uniform-random baseline; the evaluation floor."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def decide(self, offer: RideOffer) -> Decision:
        action = ACTIONS[int(self.rng.integers(0, len(ACTIONS)))]
        return Decision(
            action=action,
            reasoning="random baseline: uniform action draw",
            confidence=round(float(self.rng.uniform(0.0, 1.0)), 3),
            reservation_fare=round(offer.quoted_fare * float(self.rng.uniform(0.9, 1.3)), 2),
            tools_called=[],
        )
