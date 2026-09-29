"""Intervention screening: simulated lift of cells vs control, with honest noise.

For each intervention cell (a price/surge variant of the same underlying real
trips) the lift vs the control cell is computed per rider -- a paired design,
since every cell contains counterfactual offers built from the same base
trips. Each cell's offers are split deterministically (by offer id) into a
**pre period** and an **experiment window**: lifts are computed on the
experiment window, and each rider's pre-period control-cell metric is the
CUPED covariate (a genuine pre-period baseline, so its noise does not leak
into the delta). Uncertainty is decomposed into two components, reported
separately:

- **population (rider) variance**: nonparametric bootstrap over riders,
  resampling paired per-rider deltas.
- **model (LLM sampling) variance**: parametric bootstrap of the decision
  layer -- each decision is redrawn as Bernoulli(p_accept) -- capturing the
  noise a stochastic LLM would add on the same offers.

The CUPED adjustment uses the rider's pre-period control metric as the
covariate and reports the achieved variance reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import norm

Z_975 = float(norm.ppf(0.975))

METRICS = ("accept_rate", "completed_rides", "mean_fare", "abandonment_rate")
PRIMARY = "accept_rate"
PRE_SPLIT = 0.5


@dataclass(frozen=True)
class ScreeningCell:
    name: str
    price_mult: float
    surge_mult: float
    description: str


CONTROL_CELL = ScreeningCell("base", 1.0, 1.0, "current pricing and surge (control)")
INTERVENTION_CELLS: tuple[ScreeningCell, ...] = (
    ScreeningCell("discount_20pct", 0.8, 1.0, "20% fare discount"),
    ScreeningCell("price_up_25pct", 1.25, 1.0, "25% fare increase"),
    ScreeningCell("price_up_80pct", 1.8, 1.0, "80% fare increase"),
    ScreeningCell("surge_2x", 1.0, 2.0, "2x surge multiplier"),
    ScreeningCell("surge_2x_price_up", 1.25, 2.0, "2x surge plus 25% fare"),
)


def cell_rows(trace: pd.DataFrame, cell: ScreeningCell) -> pd.DataFrame:
    return trace[
        (trace["price_mult"] == cell.price_mult) & (trace["surge_mult"] == cell.surge_mult)
    ]


def _split_pre_post(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic pre/post split of one cell's offers per rider."""
    ordered = cell_df.sort_values(["rider_id", "offer_id"])
    pre_parts: list[pd.DataFrame] = []
    post_parts: list[pd.DataFrame] = []
    for _, group in ordered.groupby("rider_id", sort=True):
        n_pre = max(1, int(len(group) * PRE_SPLIT))
        pre_parts.append(group.iloc[:n_pre])
        post_parts.append(group.iloc[n_pre:])
    return pd.concat(pre_parts), pd.concat(post_parts)


def _observed_metrics(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Per-rider observed metrics for one cell (columns = METRICS)."""
    rows: dict[int, dict[str, float]] = {}
    for rider_id, group in cell_df.groupby("rider_id", sort=True):
        accepted = group["accepted"].to_numpy(dtype=bool)
        rate = float(accepted.mean())
        fares = group.loc[accepted, "quoted_fare"].to_numpy(dtype=float)
        rows[int(cast(int, rider_id))] = {
            "accept_rate": rate,
            "completed_rides": float(accepted.sum()),
            "mean_fare": float(fares.mean()) if len(fares) else float("nan"),
            "abandonment_rate": 1.0 - rate,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _draw_metrics(p: np.ndarray, fares: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """One sampling replicate of a rider's cell metrics (Bernoulli(p_accept))."""
    accepted = rng.random(len(p)) < p
    rate = float(accepted.mean())
    accepted_fares = fares[accepted]
    return {
        "accept_rate": rate,
        "completed_rides": float(accepted.sum()),
        "mean_fare": float(accepted_fares.mean()) if len(accepted_fares) else float("nan"),
        "abandonment_rate": 1.0 - rate,
    }


def _metric_deltas(
    cell_post: pd.DataFrame,
    control_post: pd.DataFrame,
    control_pre: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """Per-rider post-window deltas, pre-period control covariates, and
    per-rider decision inputs for the sampling bootstrap."""
    cell_metrics = _observed_metrics(cell_post)
    control_metrics = _observed_metrics(control_post)
    covariate_metrics = _observed_metrics(control_pre)
    riders = sorted(set(cell_metrics.index) & set(control_metrics.index))
    deltas = {
        metric: (cell_metrics.loc[riders, metric] - control_metrics.loc[riders, metric]).to_numpy()
        for metric in METRICS
    }
    covariates = {metric: covariate_metrics.loc[riders, metric].to_numpy() for metric in METRICS}
    grouped_cell = {int(cast(int, r)): g for r, g in cell_post.groupby("rider_id", sort=True)}
    grouped_control = {int(cast(int, r)): g for r, g in control_post.groupby("rider_id", sort=True)}
    inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for rider in riders:
        cell_group = grouped_cell[rider]
        control_group = grouped_control[rider]
        inputs[str(rider)] = (
            cell_group["p_accept"].to_numpy(dtype=float),
            cell_group["quoted_fare"].to_numpy(dtype=float),
            control_group["p_accept"].to_numpy(dtype=float),
            control_group["quoted_fare"].to_numpy(dtype=float),
        )
    return deltas, covariates, inputs


def _cuped(deltas: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float, float]:
    """CUPED adjustment with the pre-period control metric as the covariate."""
    var_x = float(np.var(covariate))
    var_delta = float(np.var(deltas))
    if var_x <= 0 or var_delta <= 0 or len(deltas) < 3:
        return deltas.copy(), 0.0, 0.0
    theta = float(np.cov(deltas, covariate)[0, 1] / var_x)
    adjusted = deltas - theta * (covariate - covariate.mean())
    reduction = 1.0 - float(np.var(adjusted)) / var_delta
    return adjusted, theta, reduction


def screen_cell(
    trace: pd.DataFrame,
    cell: ScreeningCell,
    seed: int = 7,
    n_boot_riders: int = 1000,
    n_boot_samples: int = 200,
) -> dict[str, Any]:
    """Lift of one intervention cell vs control with decomposed uncertainty.

    Returns per-metric dicts with ``lift``, ``se_rider``, ``se_sampling``,
    ``se_total``, ``ci``, ``lift_cuped``, ``cuped_var_reduction``, ``cuped_ci``.
    """
    _cell_pre, cell_post = _split_pre_post(cell_rows(trace, cell))
    control_pre, control_post = _split_pre_post(cell_rows(trace, CONTROL_CELL))
    deltas, covariates, inputs = _metric_deltas(cell_post, control_post, control_pre)
    n_riders = len(deltas[PRIMARY])
    if n_riders < 5:
        raise ValueError(f"cell {cell.name} has only {n_riders} paired riders; need >= 5")
    rng = np.random.default_rng(seed)

    results: dict[str, Any] = {
        "cell": cell.name,
        "description": cell.description,
        "n_riders": n_riders,
        "n_pre": int(len(control_pre) // max(n_riders, 1)),
        "n_post": int(len(control_post) // max(n_riders, 1)),
        "metrics": {},
    }
    for metric in METRICS:
        delta_all = deltas[metric]
        covariate_all = covariates[metric]
        valid = np.isfinite(delta_all) & np.isfinite(covariate_all)
        if valid.sum() < 5:
            results["metrics"][metric] = {
                "lift": float("nan"),
                "se_rider": float("nan"),
                "se_sampling": float("nan"),
                "se_total": float("nan"),
                "ci": (float("nan"), float("nan")),
                "lift_cuped": float("nan"),
                "cuped_var_reduction": float("nan"),
                "cuped_ci": (float("nan"), float("nan")),
            }
            continue
        delta = delta_all[valid]
        covariate = covariate_all[valid]
        rider_list = [rider for rider, ok in zip(inputs, valid, strict=True) if ok]
        lift = float(delta.mean())

        boot_lifts = np.empty(n_boot_riders)
        for b in range(n_boot_riders):
            index = rng.integers(0, len(delta), size=len(delta))
            boot_lifts[b] = delta[index].mean()
        se_rider = float(boot_lifts.std())

        per_rider_var = np.empty(len(rider_list))
        for i, rider in enumerate(rider_list):
            cell_p, cell_fare, control_p, control_fare = inputs[rider]
            draws = np.array(
                [
                    _draw_metrics(cell_p, cell_fare, rng)[metric]
                    - _draw_metrics(control_p, control_fare, rng)[metric]
                    for _ in range(n_boot_samples)
                ]
            )
            per_rider_var[i] = float(np.nanvar(draws)) if np.isfinite(draws).any() else 0.0
        se_sampling = float(np.sqrt(np.nanmean(per_rider_var) / len(rider_list)))
        se_total = float(np.hypot(se_rider, se_sampling))
        ci = (lift - Z_975 * se_total, lift + Z_975 * se_total)

        adjusted, theta, reduction = _cuped(delta, covariate)
        lift_cuped = float(adjusted.mean())
        boot_cuped = np.empty(n_boot_riders)
        for b in range(n_boot_riders):
            index = rng.integers(0, len(adjusted), size=len(adjusted))
            boot_cuped[b] = adjusted[index].mean()
        se_rider_cuped = float(boot_cuped.std())

        per_rider_var_cuped = np.empty(len(rider_list))
        for i, rider in enumerate(rider_list):
            cell_p, cell_fare, control_p, control_fare = inputs[rider]
            draws = np.array(
                [
                    _draw_metrics(cell_p, cell_fare, rng)[metric]
                    - _draw_metrics(control_p, control_fare, rng)[metric]
                    for _ in range(n_boot_samples)
                ]
            )
            adjusted_draws = draws - theta * (covariate[i] - covariate.mean())
            per_rider_var_cuped[i] = (
                float(np.nanvar(adjusted_draws)) if np.isfinite(adjusted_draws).any() else 0.0
            )
        se_sampling_cuped = float(np.sqrt(np.nanmean(per_rider_var_cuped) / len(rider_list)))
        se_total_cuped = float(np.hypot(se_rider_cuped, se_sampling_cuped))
        cuped_ci = (lift_cuped - Z_975 * se_total_cuped, lift_cuped + Z_975 * se_total_cuped)

        results["metrics"][metric] = {
            "lift": lift,
            "se_rider": se_rider,
            "se_sampling": se_sampling,
            "se_total": se_total,
            "ci": ci,
            "lift_cuped": lift_cuped,
            "cuped_var_reduction": reduction,
            "cuped_ci": cuped_ci,
        }
    return results
