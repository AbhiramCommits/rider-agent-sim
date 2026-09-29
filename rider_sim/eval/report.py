"""Markdown + figure rendering for a fidelity run.

Writes ``reports/fidelity_<run_id>.md`` and figures under
``reports/figures/<run_id>/``. Every number in the report comes from the
computed metrics dict; nothing is re-derived or hardcoded here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rider_sim.config import FIGURES_DIR, REPORTS_DIR
from rider_sim.eval.calibration import labeled_offers
from rider_sim.eval.distributions import (
    _matched_real_sample,
    _purpose_rank,
    _real_accept_rate_by_hour,
    _sim_accept_rate_by_hour,
)
from rider_sim.eval.mechanism import LITERATURE_CITATION, LITERATURE_ELASTICITY_RANGE, _band_rates

_FIGURE_STYLE: dict[str, Any] = {"bbox_inches": "tight", "dpi": 110}


def _img(path: Path) -> str:
    return f"![figure](figures/{path.parent.name}/{path.name})" if path.name else ""


def _figure_dir(run_id: str) -> Path:
    path = FIGURES_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _metrics_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "config",
        "AUC",
        "AUC CI",
        "Brier",
        "KS passed",
        "elasticity",
        "elast CI",
        "elast verdict",
        "$/1k decisions",
        "n decisions",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config"],
                    _fmt(row["discriminator_auc"]),
                    f"[{_fmt(row['auc_ci_low'])}, {_fmt(row['auc_ci_high'])}]",
                    _fmt(row["brier"]),
                    str(row["ks_passed"]) + "/4",
                    _fmt(row["elasticity"]),
                    f"[{_fmt(row['elasticity_ci_low'])}, {_fmt(row['elasticity_ci_high'])}]",
                    row["elasticity_verdict"],
                    _fmt(row["cost_per_1k"], 2),
                    str(row["n_decisions"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _ks_table(distributions: dict[str, Any]) -> str:
    lines = [
        "| marginal | statistic | p-value | Bonferroni alpha | verdict |",
        "|---|---|---|---|---|",
    ]
    for name in ("accepted_fare", "accepted_wait", "accept_rate_by_hour", "purpose_mix"):
        result = distributions[name]
        lines.append(
            f"| {name} | {_fmt(result['statistic'])} | {_fmt(result['p_value'], 6)} "
            f"| {_fmt(result['bonferroni_alpha'], 4)} | "
            f"{'pass' if result['passed'] else 'fail'} |"
        )
    return "\n".join(lines)


def _permutation_table(permutations: dict[str, dict[str, float]]) -> str:
    names = sorted(permutations)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for name in names:
        cells = []
        for other in names:
            if name == other:
                cells.append("-")
            elif other in permutations.get(name, {}):
                cells.append(f"{permutations[name][other]:.3f}")
            else:
                cells.append(f"{permutations[other][name]:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _calibration_figure(trace: pd.DataFrame, calibration: dict[str, Any], run_id: str) -> Path:
    labeled = labeled_offers(trace)
    curves = calibration.get("curves", {})
    if labeled.empty or "overall" not in curves:
        return Path()
    keys = [
        key
        for key in curves
        if key in ("overall",)
        or "price_sensitivity" in key
        or "wait_tolerance" in key
        or "has_transit_alternative" in key
    ]
    if not keys:
        keys = ["overall"]
    n_cols = min(2, len(keys))
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    flat_axes = [axes] if n_rows * n_cols == 1 else list(np.ravel(axes))
    for index, key in enumerate(keys):
        axis = flat_axes[index]
        curve = curves[key]
        axis.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        axis.plot(curve["predicted"], curve["observed"], "o-", label=key)
        axis.set_xlabel("predicted accept probability")
        axis.set_ylabel("observed real accept rate")
        axis.set_title(key)
        axis.legend()
    for axis in flat_axes[len(keys) :]:
        axis.axis("off")
    path = _figure_dir(run_id) / "calibration.png"
    fig.savefig(path, **_FIGURE_STYLE)
    plt.close(fig)
    return path


def _ks_figure(trace: pd.DataFrame, real_events: pd.DataFrame, run_id: str) -> Path:
    sim_accepted = trace[trace["accepted"] == 1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pairs = [
        (axes[0][0], "accepted fare", real_events["quoted_fare"], sim_accepted["quoted_fare"], 0),
        (axes[0][1], "accepted wait (min)", real_events["eta_min"], sim_accepted["eta_min"], 1),
    ]
    for axis, title, real_series, sim_series, offset in pairs:
        real = _matched_real_sample(real_series, len(sim_series), offset)
        sim = sim_series.to_numpy(dtype=float)
        axis.plot(np.sort(real), np.linspace(0, 1, len(real)), label="real")
        axis.plot(np.sort(sim), np.linspace(0, 1, len(sim)), label="sim")
        axis.set_title(title)
        axis.legend()
    purpose_map = _purpose_rank(real_events)
    axis = axes[1][0]
    real_purpose = real_events["purpose"].map(purpose_map)
    sim_purpose = sim_accepted["purpose"].map(purpose_map)
    axis.hist([real_purpose, sim_purpose], bins=len(purpose_map), label=["real", "sim"], alpha=0.6)
    axis.set_xticks(range(len(purpose_map)))
    axis.set_xticklabels(sorted(purpose_map, key=lambda k: purpose_map[k]))
    axis.set_title("trip purpose mix")
    axis.legend()
    axis = axes[1][1]
    real_rates = _real_accept_rate_by_hour(real_events)
    sim_rates = _sim_accept_rate_by_hour(trace)
    hours = sorted(set(real_rates) & set(sim_rates))
    axis.plot(hours, [real_rates[h] for h in hours], "o-", label="real")
    axis.plot(hours, [sim_rates[h] for h in hours], "s-", label="sim")
    axis.set_xlabel("hour")
    axis.set_ylabel("accept rate")
    axis.set_title("accept rate by hour")
    axis.legend()
    path = _figure_dir(run_id) / "distributions.png"
    fig.savefig(path, **_FIGURE_STYLE)
    plt.close(fig)
    return path


def _importances_figure(importances: dict[str, float], run_id: str) -> Path:
    names = list(importances)[:8][::-1]
    values = [importances[name] for name in names]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.barh(names, values)
    axis.set_xlabel("LightGBM gain")
    axis.set_title("discriminator feature importances")
    path = _figure_dir(run_id) / "importances.png"
    fig.savefig(path, **_FIGURE_STYLE)
    plt.close(fig)
    return path


def _elasticity_figure(trace: pd.DataFrame, mechanism: dict[str, Any], run_id: str) -> Path:
    centers, rates = _band_rates(trace)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.loglog(centers, rates, "o", label="band accept rate")
    slope = mechanism["elasticity"]
    if np.isfinite(slope):
        fit = np.exp(np.log(rates.mean()) + slope * (np.log(centers) - np.log(centers.mean())))
        axis.loglog(centers, fit, "--", label=f"fit slope {slope:.2f}")
    lo, hi = LITERATURE_ELASTICITY_RANGE
    axis.set_xlabel("surge multiplier (log)")
    axis.set_ylabel("accept rate (log)")
    axis.set_title(f"surge elasticity: {mechanism['verdict']} (literature {lo}..{hi})")
    axis.legend()
    path = _figure_dir(run_id) / "elasticity.png"
    fig.savefig(path, **_FIGURE_STYLE)
    plt.close(fig)
    return path


def _ablations_figure(table: dict[str, dict[str, Any]], run_id: str) -> Path:
    names = list(table)
    aucs = [table[name]["discriminator_auc"] for name in names]
    errors = [
        (
            table[name]["discriminator_auc"] - table[name]["auc_ci_low"],
            table[name]["auc_ci_high"] - table[name]["discriminator_auc"],
        )
        for name in names
    ]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(names, aucs, yerr=np.asarray(errors).T, capsize=4)
    axis.axhline(0.5, color="k", linestyle="--", label="indistinguishable (AUC=0.5)")
    axis.set_ylabel("discriminator AUC")
    axis.set_title("ablation: real-vs-sim discriminability")
    axis.tick_params(axis="x", rotation=30)
    axis.legend()
    path = _figure_dir(run_id) / "ablations.png"
    fig.savefig(path, **_FIGURE_STYLE)
    plt.close(fig)
    return path


def render_report(
    run_id: str,
    provider: str,
    table: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    traces: dict[str, pd.DataFrame],
    permutations: dict[str, dict[str, float]],
    offers: pd.DataFrame,
    real_events: pd.DataFrame,
    seed: int,
) -> Path:
    """Render the fidelity report and figures; returns the report path."""
    full_name = "full" if "full" in metrics else next(iter(metrics))
    full_metrics = metrics[full_name]
    full_trace = traces[full_name]

    figures = {
        "calibration": _calibration_figure(full_trace, full_metrics["calibration"], run_id),
        "distributions": _ks_figure(full_trace, real_events, run_id),
        "importances": _importances_figure(full_metrics["discriminator"]["importances"], run_id),
        "elasticity": _elasticity_figure(full_trace, full_metrics["mechanism"], run_id),
        "ablations": _ablations_figure(table, run_id),
    }

    rows = [{"config": name, **values} for name, values in table.items()]
    rows.sort(key=lambda r: r["config"])
    mechanism = full_metrics["mechanism"]
    sections = [
        "# Fidelity evaluation",
        "",
        f"- run_id: `{run_id}`",
        f"- provider: `{provider}`",
        f"- seed: {seed}",
        f"- offers generated: {len(offers):,} across {offers['rider_id'].nunique()} riders",
        f"- labeled offers for calibration: {full_metrics['calibration'].get('n_labeled', 0):,}",
        "",
        "## Ablation table",
        "",
        _metrics_table(rows),
        "",
        f"{_img(figures['ablations'])}",
        "",
        "## Discriminator (real vs simulated sequences)",
        "",
        "Grouped K-fold by rider_id; pooled OOF AUC with bootstrap 95% CI. "
        "AUC near 0.5: indistinguishable. AUC near 1.0: trivially separable.",
        "",
        "Top features for the full config:",
        "",
        "```json",
        json.dumps(full_metrics["discriminator"]["importances"], indent=2),
        "```",
        "",
        f"{_img(figures['importances'])}",
        "",
        "## Calibration (Brier + reliability vs real revealed acceptance)",
        "",
        f"- overall Brier (full config): {_fmt(full_metrics['calibration']['brier'])}",
        "",
        f"{_img(figures['calibration'])}",
        "",
        "## Distributions (KS vs real marginals, full config)",
        "",
        _ks_table(full_metrics["distributions"]),
        "",
        "Purpose-mix KS is reported with the caveat that KS is not "
        "distribution-free for categorical data; the category order is computed "
        "from the real purpose frequencies.",
        "",
        f"{_img(figures['distributions'])}",
        "",
        "## Mechanism: surge elasticity vs literature",
        "",
        f"- literature range: {LITERATURE_ELASTICITY_RANGE} ({LITERATURE_CITATION})",
        f"- full config elasticity: {_fmt(mechanism['elasticity'])} "
        f"(95% CI [{_fmt(mechanism['elasticity_ci'][0])}, "
        f"{_fmt(mechanism['elasticity_ci'][1])}])",
        f"- sign PASS: {mechanism['sign_pass']}; magnitude PASS: "
        f"{mechanism['magnitude_pass']}; verdict: **{mechanism['verdict']}**",
        "",
        f"{_img(figures['elasticity'])}",
        "",
        "## Ablation significance (permutation p-values for AUC differences)",
        "",
        _permutation_table(permutations),
        "",
    ]
    report_path = REPORTS_DIR / f"fidelity_{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections))
    return report_path
