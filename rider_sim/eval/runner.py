"""End-to-end fidelity evaluation orchestrator.

Flow per run:

1. Load personas + real trips; generate the counterfactual offer grid.
2. For each ablation config, run the agent policy over every offer and save
   a decision trace to ``data/processed/traces/<run_id>/<config>.parquet``.
3. Compute, per config: discriminator AUC (+ bootstrap CI), calibration
   Brier, KS marginals (Bonferroni), surge elasticity vs the literature
   range, and cost per 1k decisions.
4. Pairwise permutation tests of AUC differences across configs.
5. Render ``reports/fidelity_<run_id>.md`` plus matplotlib figures.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

from rider_sim.agent.llm import LLMBackend, LLMClient, resolve_backend
from rider_sim.agent.memory import RiderMemory
from rider_sim.agent.policy import (
    LLMRiderAgent,
    LogitRiderAgent,
    RandomRiderAgent,
    RiderAgent,
    fit_logit_model,
)
from rider_sim.agent.tools import ToolRegistry
from rider_sim.config import (
    LLM_CACHE_DB,
    PERSONAS_PATH,
    RIDER_AGENT_PROMPT_V1,
    RIDER_AGENT_PROMPT_V2,
    TRIPS_PATH,
)
from rider_sim.eval.ablations import (
    DEMOGRAPHIC_FIELDS,
    AblationConfig,
    permutation_matrix,
    select_configs,
    summary_row,
)
from rider_sim.eval.backends import OfflineBackend
from rider_sim.eval.calibration import run_calibration
from rider_sim.eval.discriminator import (
    real_events,
    run_discriminator,
    sequence_features,
    sim_events,
)
from rider_sim.eval.distributions import run_distribution_tests
from rider_sim.eval.mechanism import surge_elasticity
from rider_sim.eval.offers import EvalOffer, generate_offers
from rider_sim.eval.report import render_report
from rider_sim.eval.traces import save_trace
from rider_sim.personas import RiderPersona, load_personas

console = Console()


def resolve_eval_backend(provider: str) -> LLMBackend:
    """Resolve the LLM backend for a run; fall back to the offline backend."""
    load_dotenv()
    if provider in ("anthropic", "openai"):
        return resolve_backend(provider)
    if provider == "offline":
        return OfflineBackend()
    if os.getenv("ANTHROPIC_API_KEY"):
        return resolve_backend("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        return resolve_backend("openai")
    console.log(
        "[yellow]no LLM API key found; using the deterministic offline backend "
        "(pipeline validation only, not a scientific LLM result)[/yellow]"
    )
    return OfflineBackend()


def _persona_payload(persona: RiderPersona, persona_mode: str) -> dict[str, Any] | None:
    if persona_mode == "full":
        return None
    return {key: value for key, value in persona.model_dump().items() if key in DEMOGRAPHIC_FIELDS}


def _llm_accept_probability(action: str, confidence: float) -> float:
    return confidence if action == "accept" else 1.0 - confidence


def _make_agent(
    config: AblationConfig,
    persona: RiderPersona,
    backend: LLMBackend,
    client: LLMClient | None,
    seed: int,
    shared_logit_model: Any | None = None,
) -> RiderAgent:
    rider_seed = seed + persona.rider_id
    if config.kind == "logit":
        return LogitRiderAgent(persona=persona, seed=rider_seed, model=shared_logit_model)
    if config.kind == "random":
        return RandomRiderAgent(seed=rider_seed)
    assert client is not None
    memory = RiderMemory(rider_id=persona.rider_id)
    return LLMRiderAgent(
        persona=persona,
        memory=memory,
        client=client,
        provider=backend.name,
        seed=rider_seed,
        prompt_path=RIDER_AGENT_PROMPT_V2 if config.prompt == "v2" else RIDER_AGENT_PROMPT_V1,
        memory_k=5 if config.use_memory else 0,
        tools=None if config.use_tools else ToolRegistry([]),
        persona_payload=_persona_payload(persona, config.persona_mode),
    )


def run_config(
    config: AblationConfig,
    offers: pd.DataFrame,
    personas: dict[int, RiderPersona],
    backend: LLMBackend,
    seed: int,
    run_id: str,
    cache_db: Path | None,
    trips_path: Path,
) -> tuple[pd.DataFrame, float]:
    """Run one config over all offers; return (trace, total_cost_usd)."""
    rows: list[dict[str, object]] = []
    client: LLMClient | None = (
        LLMClient(backend=backend, cache_db=cache_db) if config.kind == "llm" else None
    )
    shared_logit_model = fit_logit_model(trips_path, seed)[0] if config.kind == "logit" else None

    for rider_id, rider_offers in offers.groupby("rider_id", sort=True):
        persona = personas[int(cast(int, rider_id))]
        agent = _make_agent(config, persona, backend, client, seed, shared_logit_model)
        for offer_record in rider_offers.to_dict("records"):
            offer = EvalOffer(
                rider_id=int(offer_record["rider_id"]),
                offer_id=str(offer_record["offer_id"]),
                origin_zone=int(offer_record["origin_zone"]),
                dest_zone=int(offer_record["dest_zone"]),
                purpose=str(offer_record["purpose"]),
                quoted_fare=float(offer_record["quoted_fare"]),
                surge=float(offer_record["surge"]),
                eta_min=float(offer_record["eta_min"]),
                eta_framing=str(offer_record["eta_framing"]),
                discount_pct=float(offer_record["discount_pct"]),
                time_of_day=str(offer_record["time_of_day"]),
                weather=str(offer_record["weather"]),
                transit_alt_minutes=float(offer_record["transit_alt_minutes"]),
                price_mult=float(offer_record["price_mult"]),
                surge_mult=float(offer_record["surge_mult"]),
                is_base=bool(offer_record["is_base"]),
                reject_counterfactual=bool(offer_record["reject_counterfactual"]),
            ).to_ride_offer()
            cost_before = client.cost_usd if client is not None else 0.0
            decision = agent.decide(offer)
            cost_delta = (client.cost_usd if client is not None else 0.0) - cost_before
            if isinstance(agent, LogitRiderAgent):
                p_accept = agent.accept_probability(offer)
            elif isinstance(agent, RandomRiderAgent):
                p_accept = 1.0 / 4.0
            else:
                p_accept = _llm_accept_probability(decision.action, decision.confidence)
            rows.append(
                {
                    "run_id": run_id,
                    "config": config.name,
                    "rider_id": int(offer_record["rider_id"]),
                    "offer_id": str(offer_record["offer_id"]),
                    "origin_zone": int(offer_record["origin_zone"]),
                    "dest_zone": int(offer_record["dest_zone"]),
                    "purpose": str(offer_record["purpose"]),
                    "pickup_hour": int(offer_record["pickup_hour"]),
                    "quoted_fare": float(offer_record["quoted_fare"]),
                    "surge": float(offer_record["surge"]),
                    "eta_min": float(offer_record["eta_min"]),
                    "eta_framing": str(offer_record["eta_framing"]),
                    "discount_pct": float(offer_record["discount_pct"]),
                    "time_of_day": str(offer_record["time_of_day"]),
                    "weather": str(offer_record["weather"]),
                    "transit_alt_minutes": float(offer_record["transit_alt_minutes"]),
                    "price_mult": float(offer_record["price_mult"]),
                    "surge_mult": float(offer_record["surge_mult"]),
                    "is_base": bool(offer_record["is_base"]),
                    "reject_counterfactual": bool(offer_record["reject_counterfactual"]),
                    "action": decision.action,
                    "accepted": int(decision.action == "accept"),
                    "confidence": decision.confidence,
                    "reservation_fare": decision.reservation_fare,
                    "p_accept": p_accept,
                    "price_sensitivity": persona.price_sensitivity,
                    "wait_tolerance_minutes": persona.wait_tolerance_minutes,
                    "has_transit_alternative": persona.has_transit_alternative,
                    "income_bracket": persona.income_bracket,
                    "loyalty_tier": persona.loyalty_tier,
                    "cost_usd": cost_delta,
                    "prompt_version": (
                        "rider_agent.v1"
                        if config.kind == "llm" and config.prompt == "v1"
                        else "rider_agent.v2"
                        if config.kind == "llm"
                        else ""
                    ),
                }
            )
    trace = pd.DataFrame(rows)
    save_trace(run_id, config.name, rows)
    if client is not None:
        client.report_cache_stats()
    return trace, client.cost_usd if client is not None else 0.0


def run_evaluation(
    run_id: str | None = None,
    ablations: str = "all",
    n_riders: int = 40,
    offers_per_rider: int = 2,
    seed: int = 7,
    provider: str = "auto",
    trips_path: Path = TRIPS_PATH,
    personas_path: Path = PERSONAS_PATH,
    cache_db: Path | None = LLM_CACHE_DB,
    n_bootstrap: int = 500,
    n_permutations: int = 200,
) -> Path:
    """Run the full fidelity evaluation; returns the report path."""
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    trips = pd.read_parquet(trips_path)
    personas = {p.rider_id: p for p in load_personas(personas_path)}
    offers = generate_offers(trips, n_riders=n_riders, trips_per_rider=offers_per_rider, seed=seed)
    rng = np.random.default_rng(seed)
    market_fares = trips["fare_paid"].to_numpy(dtype=float)
    events = real_events(trips)
    n_sim_riders = int(offers["rider_id"].nunique())
    real_rider_ids = rng.choice(
        events["rider_id"].unique(),
        size=min(n_sim_riders, len(events["rider_id"].unique())),
        replace=False,
    )
    real_feats = sequence_features(
        events[events["rider_id"].isin(real_rider_ids)], market_fares, seed
    )

    backend = resolve_eval_backend(provider)
    configs = select_configs(ablations)
    metrics_by_config: dict[str, dict[str, Any]] = {}
    costs: dict[str, float] = {}
    traces: dict[str, pd.DataFrame] = {}
    scores: dict[str, tuple[list[float], list[float]]] = {}

    for config in configs:
        console.log(f"[bold]running config[/bold] {config.name} (backend={backend.name})")
        trace, cost_usd = run_config(
            config, offers, personas, backend, seed, run_id, cache_db, trips_path
        )
        traces[config.name] = trace
        costs[config.name] = cost_usd
        sim_feats = sequence_features(sim_events(trace), market_fares, seed)
        discriminator = run_discriminator(real_feats, sim_feats, seed=seed, n_bootstrap=n_bootstrap)
        metrics_by_config[config.name] = {
            "discriminator": discriminator,
            "calibration": run_calibration(trace),
            "distributions": run_distribution_tests(trace, events, seed=seed),
            "mechanism": surge_elasticity(trace, n_bootstrap=n_bootstrap, seed=seed),
        }
        scores[config.name] = (
            cast(list[float], discriminator["oof_real_scores"]),
            cast(list[float], discriminator["oof_sim_scores"]),
        )

    table = {
        name: summary_row(metrics_by_config[name], len(traces[name]), costs[name])
        for name in metrics_by_config
    }
    perms = permutation_matrix(scores, n_permutations=n_permutations, seed=seed)
    return render_report(
        run_id=run_id,
        provider=backend.name,
        table=table,
        metrics=metrics_by_config,
        traces=traces,
        permutations=perms,
        offers=offers,
        real_events=events,
        seed=seed,
    )
