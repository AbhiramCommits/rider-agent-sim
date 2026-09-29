"""Ablation configs, per-config metrics, and permutation tests.

Configs: (a) full agent (prompt v1), (b) no memory, (c) persona reduced to
demographics only, (d) no tool use, (e) prompt v1 vs restructured v2,
(f) LogitRiderAgent baseline, (g) RandomRiderAgent baseline.

AUC differences between configs are tested with permutation tests: the two
configs' out-of-fold simulated scores are pooled, config labels are permuted,
and the observed AUC difference is compared against the permutation null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

DEMOGRAPHIC_FIELDS = (
    "rider_id",
    "home_zone",
    "work_zone",
    "income_bracket",
    "loyalty_tier",
    "observed_trip_count",
)

BASELINE_CONFIG_NAMES = ("logit_baseline", "random_baseline")
LLM_CONFIG_NAMES = (
    "full",
    "prompt_v2",
    "no_memory",
    "demographics_only",
    "no_tools",
)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    kind: str  # "llm" | "logit" | "random"
    persona_mode: str = "full"  # "full" | "demographics"
    use_memory: bool = True
    use_tools: bool = True
    prompt: str = "v1"  # "v1" | "v2"


CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig("full", "llm"),
    AblationConfig("prompt_v2", "llm", prompt="v2"),
    AblationConfig("no_memory", "llm", use_memory=False),
    AblationConfig("demographics_only", "llm", persona_mode="demographics"),
    AblationConfig("no_tools", "llm", use_tools=False),
    AblationConfig("logit_baseline", "logit"),
    AblationConfig("random_baseline", "random"),
)


def select_configs(ablations: str) -> tuple[AblationConfig, ...]:
    if ablations == "all":
        return CONFIGS
    if ablations == "baselines":
        return tuple(c for c in CONFIGS if c.kind != "llm")
    if ablations == "llm":
        return tuple(c for c in CONFIGS if c.kind == "llm")
    raise ValueError(f"unknown ablations selection {ablations!r}; expected all|baselines|llm")


def auc_difference_permutation_test(
    real_scores: np.ndarray,
    sim_scores_a: np.ndarray,
    sim_scores_b: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value for |AUC(a) - AUC(b)| vs the same real set."""

    def auc(scores: np.ndarray) -> float:
        labels = np.concatenate([np.zeros(len(real_scores)), np.ones(len(scores))])
        return float(roc_auc_score(labels, np.concatenate([real_scores, scores])))

    observed = abs(auc(sim_scores_a) - auc(sim_scores_b))
    pooled = np.concatenate([sim_scores_a, sim_scores_b])
    n_a = len(sim_scores_a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(len(pooled))
        perm_a = pooled[permuted[:n_a]]
        perm_b = pooled[permuted[n_a:]]
        count += int(abs(auc(perm_a) - auc(perm_b)) >= observed)
    return float((count + 1) / (n_permutations + 1))


def permutation_matrix(
    scores: dict[str, tuple[list[float], list[float]]],
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Pairwise AUC-difference permutation p-values between configs."""
    names = sorted(scores)
    matrix: dict[str, dict[str, float]] = {}
    for i, name_a in enumerate(names):
        row: dict[str, float] = {}
        for name_b in names[i + 1 :]:
            real_scores = np.asarray(scores[name_a][0])
            p_value = auc_difference_permutation_test(
                real_scores,
                np.asarray(scores[name_a][1]),
                np.asarray(scores[name_b][1]),
                n_permutations=n_permutations,
                seed=seed,
            )
            row[name_b] = p_value
        matrix[name_a] = row
    return matrix


def cost_per_1k(cost_usd: float, n_decisions: int) -> float:
    """Cost per 1,000 decisions (computed from total spend and decision count)."""
    if n_decisions <= 0:
        return 0.0
    return cost_usd / n_decisions * 1000.0


def summary_row(metrics: dict[str, Any], n_decisions: int, cost_usd: float) -> dict[str, Any]:
    """One row of the ablation table."""
    return {
        "discriminator_auc": metrics["discriminator"]["auc"],
        "auc_ci_low": metrics["discriminator"]["auc_ci"][0],
        "auc_ci_high": metrics["discriminator"]["auc_ci"][1],
        "brier": metrics["calibration"]["brier"],
        "ks_passed": metrics["distributions"]["n_passed"],
        "elasticity": metrics["mechanism"]["elasticity"],
        "elasticity_ci_low": metrics["mechanism"]["elasticity_ci"][0],
        "elasticity_ci_high": metrics["mechanism"]["elasticity_ci"][1],
        "elasticity_verdict": metrics["mechanism"]["verdict"],
        "cost_per_1k": cost_per_1k(cost_usd, n_decisions),
        "n_decisions": n_decisions,
    }
