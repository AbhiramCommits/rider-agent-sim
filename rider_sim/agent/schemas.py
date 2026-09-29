"""Strict Pydantic schemas for ride offers, rider decisions, and parsing.

Both models run in strict mode: no implicit type coercion, and unknown fields
are rejected. Model outputs that fail validation are rejected by the policy
layer and retried once with a repair prompt.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Purpose = Literal["commute", "social", "errand", "airport"]
EtaFraming = Literal["numeric", "range", "reassuring"]
TimeOfDay = Literal["morning", "midday", "evening", "night"]
Action = Literal["accept", "reject", "wait_for_better", "switch_mode"]

MAX_REASONING_WORDS = 60


class RideOffer(BaseModel):
    """A counterfactual trip offer presented to a rider agent."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    offer_id: str = Field(min_length=1)
    origin_zone: int = Field(ge=1)
    dest_zone: int = Field(ge=1)
    trip_purpose: Purpose
    quoted_fare: float = Field(gt=0.0)
    surge_multiplier: float = Field(ge=1.0)
    eta_minutes: float = Field(gt=0.0)
    eta_framing: EtaFraming
    discount_pct: float = Field(ge=0.0, le=100.0)
    time_of_day: TimeOfDay
    weather: str = Field(min_length=1)
    transit_alt_minutes: float = Field(ge=0.0)


class Decision(BaseModel):
    """A rider agent's decision on a ride offer."""

    model_config = ConfigDict(strict=True, extra="forbid")

    action: Action
    reasoning: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reservation_fare: float = Field(ge=0.0)
    tools_called: list[str] = Field(default_factory=list)

    @field_validator("reasoning")
    @classmethod
    def _check_reasoning_words(cls, value: str) -> str:
        if len(value.split()) > MAX_REASONING_WORDS:
            raise ValueError(f"reasoning must be at most {MAX_REASONING_WORDS} words")
        return value


class DecisionParseError(ValueError):
    """Raised when model output cannot be parsed into a valid Decision."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of (possibly fenced) model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index in range(len(cleaned)):
        if cleaned[index] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise DecisionParseError("no JSON object found in output")


def parse_decision(text: str) -> Decision:
    """Parse and validate a Decision from raw model text. Raises on failure."""
    try:
        payload = extract_json_object(text)
        return Decision.model_validate(payload)
    except (DecisionParseError, json.JSONDecodeError, ValueError) as exc:
        raise DecisionParseError(str(exc)) from exc
