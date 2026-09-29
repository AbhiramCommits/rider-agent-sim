"""Central paths and defaults for the rider simulation pipeline."""

from __future__ import annotations

from pathlib import Path

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
