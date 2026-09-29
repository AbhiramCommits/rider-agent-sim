"""Real-vs-simulated discriminator.

Decision sequences (one per rider) are featurized identically for real riders
(from ``trips.parquet``, where every trip was accepted) and simulated riders
(from a trace). A LightGBM classifier is trained with grouped K-fold by
rider_id; the pooled out-of-fold AUC is reported with a bootstrap 95% CI.
AUC near 0.5 means simulated sequences are indistinguishable from real ones;
near 1.0 means they are trivially separable. Feature importances name the
failure mode (e.g. "sim accepts too often at surge 2.0").

Sequence features (all computed from the sequence, never hardcoded):

- accept rate
- mean fare-vs-market percentile over accepted events (market = real fares)
- median accepted wait
- per-sequence surge elasticity (logistic slope of accept ~ log surge)
- purpose shares (commute / social / airport)
- action entropy over the 4 actions
- mean and max run length of consecutive accepts
- mean surge of accepted events and log number of decisions
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from rider_sim.agent.policy import ACTIONS
from rider_sim.eval.offers import classify_trip_purpose

FEATURE_NAMES = [
    "accept_rate",
    "fare_pctl_accepted",
    "wait_accepted_median",
    "surge_elasticity",
    "purpose_commute",
    "purpose_social",
    "purpose_airport",
    "action_entropy",
    "run_accept_mean",
    "run_accept_max",
    "surge_accepted_mean",
    "log_n_decisions",
]


def real_events(trips: pd.DataFrame) -> pd.DataFrame:
    """Real decision events: every real trip was accepted, by construction."""
    events = trips[
        [
            "rider_id",
            "fare_paid",
            "surge_ratio",
            "wait_secs",
            "pickup_zone",
            "dropoff_zone",
            "pickup_hour",
            "is_weekend",
        ]
    ].copy()
    events["quoted_fare"] = events["fare_paid"]
    events["surge"] = events["surge_ratio"].clip(1.0, 3.0)
    events["eta_min"] = events["wait_secs"] / 60.0
    events["purpose"] = trips.apply(classify_trip_purpose, axis=1)
    events["action"] = "accept"
    events["accepted"] = 1
    return events[
        [
            "rider_id",
            "quoted_fare",
            "surge",
            "eta_min",
            "purpose",
            "action",
            "accepted",
            "pickup_hour",
        ]
    ]


def sim_events(trace: pd.DataFrame) -> pd.DataFrame:
    return trace[
        ["rider_id", "quoted_fare", "surge", "eta_min", "purpose", "action", "accepted"]
    ].copy()


def _run_lengths(accepted: np.ndarray) -> tuple[float, float]:
    lengths: list[int] = []
    current = 0
    for value in accepted:
        if value:
            current += 1
        else:
            if current:
                lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    if not lengths:
        return 0.0, 0.0
    return float(np.mean(lengths)), float(np.max(lengths))


def _action_entropy(actions: pd.Series) -> float:
    counts = actions.value_counts().reindex(list(ACTIONS), fill_value=0).to_numpy(dtype=float)
    probs = counts / counts.sum()
    mask = probs > 0
    terms = np.zeros_like(probs)
    terms[mask] = probs[mask] * np.log(probs[mask])
    return float(-terms.sum())


def _surge_elasticity(events: pd.DataFrame) -> float:
    """Per-sequence logistic slope of accepted ~ log(surge); 0 when degenerate."""
    surge = np.log(events["surge"].clip(lower=1.0).to_numpy(dtype=float))
    if len(events) < 4 or np.unique(np.round(surge, 3)).size < 2:
        return 0.0
    y = events["accepted"].to_numpy(dtype=float)
    if y.std() == 0:
        return 0.0
    model = LogisticRegression(max_iter=1000, C=1e6)
    model.fit(surge.reshape(-1, 1), y)
    return float(model.coef_[0][0])


def sequence_features(events: pd.DataFrame, market_fares: np.ndarray, seed: int) -> pd.DataFrame:
    """One feature row per rider sequence."""
    rng = np.random.default_rng(seed)
    sorted_fares = np.sort(market_fares)
    n_market = len(sorted_fares)

    def fare_pctl(fare: float) -> float:
        return float(np.searchsorted(sorted_fares, fare, side="right") / n_market)

    rows: list[dict[str, float | int]] = []
    for rider_id, group in events.groupby("rider_id", sort=True):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
        accepted = group["accepted"].to_numpy(dtype=bool)
        accepted_rows = group[accepted]
        purpose_shares = (
            group["purpose"]
            .value_counts(normalize=True)
            .reindex(["commute", "social", "errand", "airport"], fill_value=0.0)
        )
        run_mean, run_max = _run_lengths(accepted)
        fare_pctls = accepted_rows["quoted_fare"].map(fare_pctl)
        rows.append(
            {
                "rider_id": int(cast(int, rider_id)),
                "accept_rate": float(accepted.mean()),
                "fare_pctl_accepted": float(fare_pctls.mean()) if len(fare_pctls) else 0.0,
                "wait_accepted_median": float(accepted_rows["eta_min"].median())
                if len(accepted_rows)
                else 0.0,
                "surge_elasticity": _surge_elasticity(group),
                "purpose_commute": float(purpose_shares["commute"]),
                "purpose_social": float(purpose_shares["social"]),
                "purpose_airport": float(purpose_shares["airport"]),
                "action_entropy": _action_entropy(group["action"]),
                "run_accept_mean": run_mean,
                "run_accept_max": run_max,
                "surge_accepted_mean": float(accepted_rows["surge"].mean())
                if len(accepted_rows)
                else 0.0,
                "log_n_decisions": float(np.log(len(group))),
            }
        )
    return pd.DataFrame(rows)


def run_discriminator(
    real_feats: pd.DataFrame,
    sim_feats: pd.DataFrame,
    n_folds: int = 5,
    n_bootstrap: int = 200,
    seed: int = 0,
    lgbm_params: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Grouped-K-fold LightGBM AUC + bootstrap CI + importances.

    ``real_feats``/``sim_feats`` are the featurized sequences with a
    ``rider_id`` column; groups are (label, rider_id). The bootstrap CI is
    computed over (label, OOF score) pairs.
    """
    real = real_feats.copy()
    sim = sim_feats.copy()
    real["label"] = 0
    sim["label"] = 1
    data = pd.concat([real, sim], ignore_index=True)
    groups = np.asarray(
        [f"{label}:{rider}" for label, rider in zip(data["label"], data["rider_id"], strict=True)]
    )
    x = data[FEATURE_NAMES].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=int)

    params: dict[str, float] = {
        "n_estimators": 100,
        "num_leaves": 15,
        "min_child_samples": max(5, len(data) // 30),
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "random_state": seed,
    }
    if lgbm_params:
        params.update({key: value for key, value in lgbm_params.items()})

    oof = np.zeros(len(data))
    for train_index, test_index in GroupKFold(n_splits=n_folds).split(x, y, groups):
        model = LGBMClassifier(**cast(Any, params))
        model.fit(x[train_index], y[train_index])
        proba = cast(np.ndarray, model.predict_proba(x[test_index]))
        oof[test_index] = proba[:, 1]

    auc = float(roc_auc_score(y, oof))

    # Bootstrap 95% CI over (label, OOF score) pairs. Under the null (real and
    # sim sequences exchangeable, e.g. identical distributions) the OOF scores
    # carry no signal and this CI covers 0.5; when the classifier picks up a
    # real difference the CI sits above 0.5.
    rng = np.random.default_rng(seed)
    boot_aucs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        index = rng.integers(0, len(y), size=len(y))
        boot_aucs[i] = roc_auc_score(y[index], oof[index])
    ci_low, ci_high = float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))

    final_model = LGBMClassifier(**cast(Any, params))
    final_model.fit(x, y)
    gains = dict(
        zip(
            FEATURE_NAMES,
            final_model.booster_.feature_importance(importance_type="gain"),
            strict=False,
        )
    )
    importances = {
        name: float(gain)
        for name, gain in sorted(gains.items(), key=lambda kv: kv[1], reverse=True)
    }
    return {
        "auc": auc,
        "auc_ci": (ci_low, ci_high),
        "n_real": len(real_feats),
        "n_sim": len(sim_feats),
        "oof_sim_scores": oof[data["label"].to_numpy() == 1].tolist(),
        "oof_real_scores": oof[data["label"].to_numpy() == 0].tolist(),
        "importances": importances,
    }
