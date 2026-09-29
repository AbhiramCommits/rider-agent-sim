"""Typer CLI for the rider simulation pipeline."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from rider_sim.config import DEFAULT_MONTH, DEFAULT_N_RIDERS, DEFAULT_SEED

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="LLM-driven generative rider simulation pipeline.",
)
console = Console()


@app.command()
def fetch(
    month: Annotated[str, typer.Option("--month", help="TLC data month, YYYY-MM")] = DEFAULT_MONTH,
) -> None:
    """Download (idempotently) the TLC HVFHV trip Parquet + zone lookup CSV."""
    from rider_sim.data.fetch import fetch_all

    fetch_all(month)


@app.command()
def build(
    n_riders: Annotated[int, typer.Option("--n-riders", min=1)] = DEFAULT_N_RIDERS,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    month: Annotated[str, typer.Option("--month", help="TLC data month, YYYY-MM")] = DEFAULT_MONTH,
) -> None:
    """Build the semi-synthetic pseudo-rider panel from real TLC trips."""
    from rider_sim.data.build_riders import build_riders

    build_riders(month=month, n_riders=n_riders, seed=seed)


@app.command()
def personas(
    n: Annotated[int, typer.Option("--n", min=1)] = DEFAULT_N_RIDERS,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
) -> None:
    """Fit data-grounded personas from pseudo-rider trip histories."""
    from rider_sim.personas import sample_personas

    sample_personas(n=n, seed=seed)


@app.command()
def simulate(
    n: Annotated[int, typer.Option("--n", min=1, help="personas to simulate")] = 200,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    mode: Annotated[str, typer.Option("--mode", help="auto | llm | statistical")] = "auto",
) -> None:
    """Simulate persona accept/reject choices on counterfactual trip offers."""
    from rider_sim.simulate import simulate as run_sim

    run_sim(n=n, seed=seed, mode=mode)


@app.command()
def evaluate(
    run_id: Annotated[str | None, typer.Option("--run-id", help="run identifier")] = None,
    ablations: Annotated[str, typer.Option("--ablations", help="all | baselines | llm")] = "all",
    n_riders: Annotated[int, typer.Option("--n-riders", min=1)] = 40,
    offers_per_rider: Annotated[int, typer.Option("--offers-per-rider", min=1)] = 2,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    provider: Annotated[
        str, typer.Option("--provider", help="auto | anthropic | openai | offline")
    ] = "auto",
) -> None:
    """Run the fidelity evaluation (traces, discriminator, ablations, report)."""
    from rider_sim.eval.runner import run_evaluation

    report_path = run_evaluation(
        run_id=run_id,
        ablations=ablations,
        n_riders=n_riders,
        offers_per_rider=offers_per_rider,
        seed=seed,
        provider=provider,
    )
    console.print(f"[green]wrote[/green] {report_path}")


if __name__ == "__main__":
    app()
