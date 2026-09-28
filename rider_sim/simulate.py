"""LLM-driven generative rider simulation.

For each sampled persona we generate *counterfactual trip offers* derived from
that rider's own real trips (real zone pair, hour, miles, fare, wait), varying
price (x0.8 / x1.0 / x1.25) and wait (x0.5 / x1.0 / x2.0) multipliers. Each
offer is judged by a decision model:

- ``llm``: the persona + offer are sent to an LLM (Anthropic Claude by
  preference, OpenAI as fallback) which returns an accept/reject decision.
- ``statistical``: a logit calibrated on the persona's traits
  (price_sensitivity, wait_tolerance_minutes).

``mode="auto"`` (default) uses the LLM when an API key is present and falls
back to the statistical model otherwise. Output is persisted to
``data/processed/simulated_choices.parquet``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from rider_sim.config import PERSONAS_PATH, SIM_CHOICES_PATH, TRIPS_PATH
from rider_sim.personas import RiderPersona, load_personas

console = Console()

PRICE_MULTS = (0.8, 1.0, 1.25)
WAIT_MULTS = (0.5, 1.0, 2.0)
OFFER_GRID = ((0.8, 1.0), (1.0, 1.0), (1.25, 1.0), (1.0, 0.5), (1.0, 2.0))

_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You are simulating the accept/reject decision of a ride-hailing rider. "
    "You receive a rider persona and a trip offer as JSON. Decide whether this "
    "rider would accept the trip given their traits, the price, and the wait. "
    'Reply with JSON only: {"accept": true} or {"accept": false}.'
)


def pick_provider(mode: str) -> str | None:
    """Return "anthropic" | "openai" | None (statistical fallback) for mode."""
    load_dotenv()
    if mode == "statistical":
        return None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if mode == "llm":
        if anthropic_key:
            return "anthropic"
        if openai_key:
            return "openai"
        raise ValueError(
            "mode='llm' requires ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(copy .env.example to .env and set one)"
        )
    if anthropic_key:
        return "anthropic"
    if openai_key:
        return "openai"
    return None


def llm_decide(provider: str, persona: RiderPersona, offer: dict[str, Any]) -> bool:
    payload = json.dumps({"persona": persona.model_dump(), "offer": offer})
    if provider == "anthropic":
        import anthropic

        model = os.getenv("ANTHROPIC_MODEL", _DEFAULT_ANTHROPIC_MODEL)
        anthropic_client = anthropic.Anthropic()
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=64,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
        )
        text = "".join(getattr(block, "text", "") for block in message.content)
    else:
        from openai import OpenAI

        model = os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        openai_client = OpenAI()
        completion = openai_client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=64,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        text = completion.choices[0].message.content or ""
    parsed = json.loads(text.strip().strip("`"))
    return bool(parsed["accept"])


def statistical_decide(
    persona: RiderPersona, offer: dict[str, Any], rng: np.random.Generator
) -> bool:
    """Logit accept decision calibrated on persona traits."""
    price_norm = float(offer["price_mult"]) - 1.0
    wait_norm = (float(offer["offered_wait_min"]) - persona.wait_tolerance_minutes) / max(
        persona.wait_tolerance_minutes, 1.0
    )
    logit = 2.0 - 2.5 * persona.price_sensitivity * price_norm - 1.5 * wait_norm
    prob = 1.0 / (1.0 + math.exp(-logit))
    return bool(rng.random() < prob)


def _build_offers(
    persona: RiderPersona, trips: pd.DataFrame, rng: np.random.Generator, trips_per_persona: int
) -> list[dict[str, Any]]:
    mine = trips[trips["rider_id"] == persona.rider_id]
    n_bases = min(trips_per_persona, len(mine))
    bases = mine.sample(n=n_bases, random_state=int(rng.integers(0, 2**31)))
    offers: list[dict[str, Any]] = []
    for base in bases.itertuples(index=False):
        base_wait_min = float(cast(float, base.wait_secs)) / 60.0
        base_fare = float(cast(float, base.fare_paid))
        for price_mult, wait_mult in OFFER_GRID:
            offers.append(
                {
                    "rider_id": persona.rider_id,
                    "pickup_zone": int(cast(int, base.pickup_zone)),
                    "dropoff_zone": int(cast(int, base.dropoff_zone)),
                    "pickup_hour": int(cast(int, base.pickup_hour)),
                    "trip_miles": float(cast(float, base.trip_miles)),
                    "base_fare": base_fare,
                    "base_wait_min": base_wait_min,
                    "price_mult": price_mult,
                    "wait_mult": wait_mult,
                    "offered_fare": base_fare * price_mult,
                    "offered_wait_min": base_wait_min * wait_mult,
                }
            )
    return offers


def simulate(
    n: int,
    seed: int,
    mode: str = "auto",
    trips_per_persona: int = 2,
    personas_path: Path = PERSONAS_PATH,
    trips_path: Path = TRIPS_PATH,
    output_path: Path = SIM_CHOICES_PATH,
) -> pd.DataFrame:
    """Simulate accept/reject choices for ``n`` personas and persist them."""
    if mode not in {"auto", "llm", "statistical"}:
        raise ValueError(f"Unknown mode {mode!r}; expected auto | llm | statistical")
    personas = load_personas(personas_path)
    trips = pd.read_parquet(trips_path)
    if n > len(personas):
        n = len(personas)
        console.log(f"[yellow]clamped[/yellow] n to {n} (only {len(personas)} personas available).")
    rng = np.random.default_rng(seed)
    if n == len(personas):
        picked = personas
    else:
        picked = [personas[int(i)] for i in rng.choice(len(personas), size=n, replace=False)]

    provider = pick_provider(mode)
    decision_mode = provider if provider else "statistical"
    console.log(f"decision mode: [bold]{decision_mode}[/bold]")

    rows: list[dict[str, Any]] = []
    for persona in picked:
        for offer in _build_offers(persona, trips, rng, trips_per_persona):
            if provider is not None:
                accepted = llm_decide(provider, persona, offer)
            else:
                accepted = statistical_decide(persona, offer, rng)
            rows.append({**offer, "accepted": accepted, "decision_mode": decision_mode})

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    table = Table(title="Simulated choices")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("personas simulated", str(len(picked)))
    table.add_row("offers generated", f"{len(frame):,}")
    table.add_row("accept rate (base offer)", f"{_base_accept_rate(frame):.1%}")
    console.print(table)
    console.print(f"[green]wrote[/green] {output_path}")
    return frame


def _base_accept_rate(frame: pd.DataFrame) -> float:
    base = frame[(frame.price_mult == 1.0) & (frame.wait_mult == 1.0)]
    if base.empty:
        return float("nan")
    return float(base.accepted.mean())
