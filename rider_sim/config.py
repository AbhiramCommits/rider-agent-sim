"""Central paths and defaults for the rider simulation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

DEFAULT_MONTH = "2024-01"
DEFAULT_N_RIDERS = 2000
DEFAULT_SEED = 7

RIDERS_PATH: Path = PROCESSED_DIR / "riders.parquet"
TRIPS_PATH: Path = PROCESSED_DIR / "trips.parquet"
PERSONAS_PATH: Path = PROCESSED_DIR / "personas.parquet"
SIM_CHOICES_PATH: Path = PROCESSED_DIR / "simulated_choices.parquet"

PROMPTS_DIR: Path = REPO_ROOT / "prompts"
RIDER_AGENT_PROMPT_V1: Path = PROMPTS_DIR / "rider_agent.v1.md"
RIDER_AGENT_PROMPT_V2: Path = PROMPTS_DIR / "rider_agent.v2.md"
LLM_CACHE_DB: Path = PROCESSED_DIR / "llm_cache.duckdb"

TRACES_DIR: Path = PROCESSED_DIR / "traces"
REPORTS_DIR: Path = REPO_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Auditable back-test holdout: the one real behavioral relationship the
# simulation must predict blind. The personas are fit from fares, waits, and
# miles only -- never from surge -- so the real demand response across surge
# bands is genuinely held out. The real effect is the log ratio of trip
# shares in the two surge bands; the sim effect is the log ratio of simulated
# accept rates over trace offers falling in the same bands.
BACKTEST_HOLDOUT: dict[str, Any] = {
    "type": "surge_band_demand_ratio",
    "high_band": [2.0, 3.0],
    "low_band": [1.0, 1.2],
}
