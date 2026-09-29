"""Screening report rendering: ranked cells, CUPED, power, back-test."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from rider_sim.config import REPORTS_DIR


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _recommend(result: dict[str, Any], backtest_verdict: str | None) -> str:
    primary = result["metrics"]["accept_rate"]
    use_cuped = isfinite(primary["cuped_var_reduction"]) and primary["cuped_var_reduction"] > 0
    ci = primary["cuped_ci"] if use_cuped else primary["ci"]
    lifts = ci[0] > 0
    veto = backtest_verdict == "FAIL"
    if veto:
        return "screen out (back-test FAIL)"
    return "screen in" if lifts else "screen out"


def _cell_table(results: list[dict[str, Any]], backtest: dict[str, Any] | None) -> str:
    headers = [
        "rank",
        "cell",
        "lift (raw)",
        "lift (CUPED)",
        "CI (effective)",
        "SE rider",
        "SE model",
        "var reduction",
        "completed",
        "fare ($)",
        "abandonment",
        "n/arm",
        "days",
        "recommendation",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for rank, result in enumerate(results, start=1):
        primary = result["metrics"]["accept_rate"]
        use_cuped = isfinite(primary["cuped_var_reduction"]) and primary["cuped_var_reduction"] > 0
        ci = primary["cuped_ci"] if use_cuped else primary["ci"]
        power = result["power"]
        n_per_arm = power["n_per_arm"]
        days = power["days"]
        n_arm_cell = "n/a" if n_per_arm is None else f"{n_per_arm:,}"
        if days is None:
            days_cell = "n/a"
        elif days < 1.0:
            days_cell = f"{days:.3f}"
        else:
            days_cell = _fmt(days, 1)
        backtest_verdict = None
        if backtest is not None and result["cell"] in ("surge_2x", "surge_2x_price_up"):
            backtest_verdict = backtest["verdict"]
        completed = result["metrics"]["completed_rides"]["lift"]
        fare = result["metrics"]["mean_fare"]["lift"]
        abandonment = result["metrics"]["abandonment_rate"]["lift"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    result["cell"],
                    _fmt(primary["lift"]),
                    _fmt(primary["lift_cuped"]),
                    f"[{_fmt(ci[0])}, {_fmt(ci[1])}]",
                    _fmt(primary["se_rider"]),
                    _fmt(primary["se_sampling"]),
                    _fmt(primary["cuped_var_reduction"], 2),
                    _fmt(completed),
                    _fmt(fare, 2),
                    _fmt(abandonment),
                    n_arm_cell,
                    days_cell,
                    _recommend(result, backtest_verdict),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_screening_report(
    run_id: str,
    config: str,
    control_rate: float,
    cell_results: list[dict[str, Any]],
    backtest: dict[str, Any] | None,
    decision_rule: str,
    generated_at: str,
    month: str,
) -> Path:
    """Render reports/screening_<run_id>.md; returns the path."""
    sections = [
        "# Product screening",
        "",
        f"- run_id: `{run_id}` (trace config: `{config}`)",
        f"- generated: {generated_at}",
        f"- control accept rate (base cell): {_fmt(control_rate)}",
        f"- traffic assumption: real HVFHV trip volume for {month} (computed from raw data)",
        "",
        "## Decision rule",
        "",
        f"> {decision_rule}",
        "",
        "## Ranked intervention cells (primary metric: accept-rate lift)",
        "",
        _cell_table(cell_results, backtest),
        "",
        "CIs are lift +/- 1.96 x total SE; the effective CI is the CUPED-adjusted "
        "one when CUPED reduces variance, else the raw one. SE rider = population "
        "variance from a rider-level bootstrap; SE model = LLM sampling variance "
        "from a parametric Bernoulli(p_accept) bootstrap of the decision layer. "
        "The two variance components are reported separately so the reader can "
        "see how much noise is the population vs the model. `completed` and "
        "`abandonment` are complements of the accept metric (no post-accept "
        "churn is modeled); `fare` is the mean accepted fare, in dollars.",
        "",
    ]

    if backtest is not None:
        real = backtest["real"]
        sim = backtest["sim"]
        sections += [
            "## Back-test (held-out real behavior)",
            "",
            "Holdout specification (auditable, from `config.BACKTEST_HOLDOUT`):",
            "",
            "```json",
            json.dumps(backtest["holdout"], indent=2),
            "```",
            "",
            f"- real log demand ratio (high vs low surge band shares): "
            f"{_fmt(real['log_ratio'])} "
            f"(shares {_fmt(real['share_high'], 4)} vs {_fmt(real['share_low'], 4)})",
            f"- simulated log accept-rate ratio: {_fmt(sim['log_ratio'])} "
            f"(95% CI [{_fmt(sim['ci'][0])}, {_fmt(sim['ci'][1])}], "
            f"n={sim['n_high']}+{sim['n_low']} offers)",
            f"- signed error (sim - real): {_fmt(backtest['signed_error'])}",
            f"- directional agreement: {backtest['direction_agrees']}",
            f"- simulated CI covers real estimate: {backtest['ci_covers_real']}",
            f"- verdict: **{backtest['verdict']}**",
            "",
            f"> {backtest['note']}",
            "",
        ]
    else:
        sections += [
            "## Back-test",
            "",
            "Not run for this screening (pass `--backtest` to include it).",
            "",
        ]

    sections += [
        "## Power (real A/B test, 80% power, alpha 0.05)",
        "",
        "Sample sizes use the closed-form two-proportion formula on the "
        "screening lift vs the control accept rate; days assume a 50/50 split "
        "of the real daily trip volume.",
        "",
    ]

    report_path = REPORTS_DIR / f"screening_{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections))
    return report_path
