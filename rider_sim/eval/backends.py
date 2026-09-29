"""Deterministic offline backend for pipeline validation without API keys.

The offline backend is NOT a model: it derives a decision from the persona
and offer JSON embedded in the prompt (the persona / memory / offer blocks,
in that order) using a trait-calibrated rule. It exists so the full
evaluation pipeline -- traces, discriminator, calibration, mechanism,
ablations, reports -- can run end to end without LLM credentials. Any
fidelity number produced with this backend is a pipeline self-test, not a
scientific result about LLM behavior.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, cast

from rider_sim.agent.llm import LLMRequest, LLMResponse

_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_PERSONA_BLOCK = 0
_OFFER_BLOCK = 2


class OfflineBackend:
    """Deterministic rule-based stand-in for an LLM provider."""

    name = "offline"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        persona = self._block(request.system, _PERSONA_BLOCK)
        offer = self._block(request.system, _OFFER_BLOCK)
        decision = self._decide(persona, offer)
        return LLMResponse(content=json.dumps(decision), tool_calls=())

    @staticmethod
    def _block(system: str, index: int) -> dict[str, Any]:
        blocks = _JSON_BLOCK.findall(system)
        if len(blocks) <= index:
            raise ValueError(f"prompt has no JSON block {index} (offline backend)")
        return cast(dict[str, Any], json.loads(blocks[index]))

    @staticmethod
    def _decide(persona: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
        sensitivity = float(persona.get("price_sensitivity", 0.5))
        tolerance = float(persona.get("wait_tolerance_minutes", 5.0))
        surge = float(offer.get("surge_multiplier", 1.0))
        eta = float(offer.get("eta_minutes", 5.0))
        transit = float(offer.get("transit_alt_minutes", 60.0))
        quoted_fare = float(offer.get("quoted_fare", 20.0))

        price_norm = surge - 1.0
        wait_norm = (eta - tolerance) / max(tolerance, 1.0)
        logit = 2.0 - 2.5 * sensitivity * price_norm - 1.5 * wait_norm
        prob = 1.0 / (1.0 + math.exp(-logit))

        if prob >= 0.6:
            action = "accept"
        elif surge > 1.5 and prob >= 0.35:
            action = "wait_for_better"
        elif transit < eta and surge > 1.2:
            action = "switch_mode"
        else:
            action = "reject"
        return {
            "action": action,
            "reasoning": (
                f"offline deterministic policy: accept prob {prob:.2f} at surge {surge:.2f}"
            ),
            "confidence": round(abs(prob - 0.5) * 2.0, 3),
            "reservation_fare": round(quoted_fare * (1.0 + max(prob - 0.5, 0.0) * 0.8), 2),
            "tools_called": [],
        }
