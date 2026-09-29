.PHONY: install lint test fetch-data build personas simulate evaluate

MONTH ?= 2024-01
N_RIDERS ?= 2000
SEED ?= 7
PERSONAS_N ?= 2000
SIM_N ?= 200
EVAL_ABLATIONS ?= all

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy rider_sim

test:
	uv run pytest

fetch-data:
	uv run python -m rider_sim fetch --month $(MONTH)

build: fetch-data
	uv run python -m rider_sim build --n-riders $(N_RIDERS) --seed $(SEED) --month $(MONTH)

personas: build
	uv run python -m rider_sim personas --n $(PERSONAS_N) --seed $(SEED)

simulate: personas
	uv run python -m rider_sim simulate --n $(SIM_N) --seed $(SEED)

evaluate: personas
	uv run python -m rider_sim evaluate --ablations $(EVAL_ABLATIONS)
