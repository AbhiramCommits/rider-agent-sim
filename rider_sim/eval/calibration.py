"""Calibration of simulated accept probabilities against real revealed acceptance.

Ground-truth labels come from the real data: base offers (price/surge
multiplier 1.0) reproduce trips the real rider actually took, so y=1; the
x1.8-fare counterfactuals are labeled y=0 (documented construction -- real
TLC records contain no rejects). Intermediate variants have no honest label
and are excluded.

Predicted accept probability per decision is config-specific:

- LLM configs: confidence when the action is accept, 1 - confidence for
  reject / switch_mode / wait_for_better (documented mapping).
- LogitRiderAgent: the fitted multinomial P(accept).
- RandomRiderAgent: 1 / number-of-actions from its uniform policy.

Reliability curves and Brier scores are computed overall and per segment
(price sensitivity tercile, wait tolerance tercile, transit alternative).
"""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

SEGMENT_COLUMNS = ["price_sensitivity", "wait_tolerance_minutes", "has_transit_alternative"]


def labeled_offers(trace: pd.DataFrame) -> pd.DataFrame:
    """Offers with a real-data-derived accept label (1 for base, 0 for 1.8x)."""
    labeled = trace[trace["is_base"] | trace["reject_counterfactual"]].copy()
    labeled["label"] = labeled["is_base"].astype(int)
    return labeled


def tercile_labels(values: pd.Series) -> pd.Series:
    """Computed tercile (low/mid/high) labels from the trace's own values."""
    low_cut, high_cut = values.quantile([1 / 3, 2 / 3])
    labels = pd.Series("mid", index=values.index)
    labels[values <= low_cut] = "low"
    labels[values > high_cut] = "high"
    return labels


def reliability_curves(
    labeled: pd.DataFrame, n_bins: int = 10
) -> dict[str, dict[str, list[float]]]:
    """Reliability curve (observed vs predicted) overall and per segment."""
    curves: dict[str, dict[str, list[float]]] = {}
    bins = max(2, min(n_bins, len(labeled) // 2))
    prob_true, prob_pred = calibration_curve(
        labeled["label"], labeled["p_accept"], n_bins=bins, strategy="uniform"
    )
    curves["overall"] = {"observed": prob_true.tolist(), "predicted": prob_pred.tolist()}
    for column in SEGMENT_COLUMNS:
        if column == "has_transit_alternative":
            groups = {str(v): g for v, g in labeled.groupby(column)}
        else:
            groups = {str(t): g for t, g in labeled.groupby(tercile_labels(labeled[column]))}
        for name, group in sorted(groups.items()):
            if len(group) < bins * 2:
                continue
            prob_true, prob_pred = calibration_curve(
                group["label"], group["p_accept"], n_bins=bins, strategy="uniform"
            )
            curves[f"{column}={name}"] = {
                "observed": prob_true.tolist(),
                "predicted": prob_pred.tolist(),
            }
    return curves


def run_calibration(trace: pd.DataFrame, n_bins: int = 10) -> dict[str, object]:
    labeled = labeled_offers(trace)
    if len(labeled) < 4:
        return {"brier": float("nan"), "curves": {}, "n_labeled": len(labeled)}
    brier = float(brier_score_loss(labeled["label"], labeled["p_accept"]))
    return {
        "brier": brier,
        "curves": reliability_curves(labeled, n_bins),
        "n_labeled": len(labeled),
    }
