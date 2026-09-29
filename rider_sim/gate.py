"""Fidelity regression gate.

Compares ``reports/metrics.json`` (latest evaluation run) against the
committed baseline ``eval_baselines.json`` and fails when:

- the discriminator AUC drifts outside the committed tolerance band,
- fewer KS marginals pass than the baseline floor,
- the elasticity sign flips.

Prints a clear metric vs baseline vs tolerance diff on failure. This is a
regression gate on simulation realism, not just on code. Run as
``python -m rider_sim.gate``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from rider_sim.config import REPO_ROOT, REPORTS_DIR

console = Console()

METRICS_PATH = REPORTS_DIR / "metrics.json"
BASELINES_PATH = REPO_ROOT / "eval_baselines.json"

CONFIG_KEY = "full"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    return cast(dict[str, Any], json.loads(path.read_text()))


def check_fidelity(
    metrics: dict[str, Any], baselines: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    """Return (passed, rows) where each row describes one gate check."""
    configs = metrics.get("configs", {})
    if CONFIG_KEY not in configs:
        raise ValueError(f"metrics.json has no {CONFIG_KEY!r} config")
    row = configs[CONFIG_KEY]

    auc = row["discriminator_auc"]
    auc_band = baselines["discriminator_auc_tolerance"]
    auc_low = baselines["discriminator_auc"] - auc_band
    auc_high = baselines["discriminator_auc"] + auc_band

    ks = row["ks_passed"]
    ks_floor = baselines["min_ks_marginals_passed"]

    sign = row["elasticity_sign"]
    expected_sign = baselines["elasticity_sign"]

    rows = [
        {
            "check": "discriminator AUC",
            "baseline": baselines["discriminator_auc"],
            "current": auc,
            "tolerance": f"+/- {auc_band}",
            "passed": auc_low <= auc <= auc_high,
        },
        {
            "check": "KS marginals passed",
            "baseline": f">= {ks_floor}",
            "current": ks,
            "tolerance": f"floor {ks_floor}",
            "passed": ks >= ks_floor,
        },
        {
            "check": "elasticity sign",
            "baseline": expected_sign,
            "current": sign,
            "tolerance": "must match",
            "passed": sign == expected_sign,
        },
    ]
    return all(r["passed"] for r in rows), rows


def main() -> int:
    try:
        metrics = load_json(METRICS_PATH)
        baselines = load_json(BASELINES_PATH)
        passed, rows = check_fidelity(metrics, baselines)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"[red]gate cannot run: {exc}[/red]")
        return 2

    table = Table(title="Fidelity gate")
    table.add_column("check")
    table.add_column("baseline", justify="right")
    table.add_column("current", justify="right")
    table.add_column("tolerance", justify="right")
    table.add_column("status", justify="center")
    for row in rows:
        table.add_row(
            row["check"],
            str(row["baseline"]),
            str(row["current"]),
            str(row["tolerance"]),
            "[green]pass[/green]" if row["passed"] else "[red]FAIL[/red]",
        )
    console.print(table)
    if not passed:
        console.print(
            "[red]fidelity gate FAILED[/red] -- simulation realism drifted from "
            f"the committed baseline in {BASELINES_PATH}. See the diff above."
        )
        return 1
    console.print("[green]fidelity gate passed[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
