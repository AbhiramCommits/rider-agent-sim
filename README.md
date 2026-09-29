# rider-agent-sim

LLM-driven generative rider simulation for a ride-hailing marketplace,
validated against real NYC TLC behavioral data.

The pipeline downloads one month of real NYC TLC High Volume FHV (Uber/Lyft/Via)
trip records, builds a semi-synthetic panel of 2,000 pseudo-riders whose trips
are real (real zones, times, fares, tips, waits, driver pay), fits each
pseudo-rider a **data-grounded persona** from their own trip history, simulates
their accept/reject choices on counterfactual offers (optionally via an LLM),
and validates the simulated choices against the real behavioral distributions.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # or: make install
cp .env.example .env    # optional; only needed for LLM decision mode
```

Set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` in `.env` to use LLM decision
mode in `simulate`. Without keys, the pipeline falls back to a statistical
decision model and everything still runs.

## Quickstart

```bash
make fetch-data   # download TLC data (idempotent, ~450 MB for 2024-01)
make build        # construct the pseudo-rider panel (2000 riders)
make personas     # fit personas (python -m rider_sim personas --n 2000 --seed 7)
make simulate     # simulate choices on counterfactual offers
make evaluate     # validate simulated choices vs real behavior
```

Or via the CLI directly:

```bash
python -m rider_sim fetch --month 2024-01
python -m rider_sim build --n-riders 2000 --seed 7
python -m rider_sim personas --n 2000 --seed 7
python -m rider_sim simulate --n 200 --seed 7 [--mode auto|llm|statistical]
python -m rider_sim evaluate --run-id demo --ablations all
```

## Data pipeline

### 1. Fetch (`rider_sim/data/fetch.py`)

Downloads two public files from the TLC CloudFront mirror into `data/raw/`,
skipping anything already cached (idempotent; partial downloads are atomic):

- `fhvhv_tripdata_{YYYY-MM}.parquet` — one month of High Volume FHV trip
  records (`request/pickup/dropoff` timestamps, zones, miles, time, fares,
  tips, driver pay, shared flags).
- `taxi_zone_lookup.csv` — zone ID → borough/zone name lookup.

### 2. Pseudo-rider panel (`rider_sim/data/build_riders.py`)

Loads the Parquet with DuckDB, filters to usable trips, and derives real
per-trip observables:

| observable | definition |
| --- | --- |
| `base_passenger_fare` | raw TLC column |
| `tips` | raw TLC column |
| `trip_miles`, `trip_time` | raw TLC columns |
| `wait_secs` | `pickup_datetime - request_datetime`, clamped to `[0, 3600]` |
| `surge_ratio` | `driver_pay / (trip_miles * median_pay_per_mile)` where the median is computed over the whole month |

Every trip is bucketed into a *commute cell* on the four clustering
dimensions (pickup zone, hour-of-day bucket, weekday/weekend, trip-miles
bucket), the cells are clustered with KMeans to form fallback neighborhoods,
and each of `N=2000` pseudo-riders is assigned 5-40 real trips sampled from a
home cell drawn proportionally to real trip volume. Outputs:

- `data/processed/trips.parquet` — every assigned trip + `rider_id` + derived observables
- `data/processed/riders.parquet` — `rider_id`, home/work zone, trip count, home cell

#### Caveat: riders are pseudo-riders, not real people

**Real TLC trip records contain no rider identifier** — every row is an
anonymous trip. The panel is therefore **semi-synthetic**: each individual
trip is real, but the attribution of trips to a specific "rider" is
constructed. Each pseudo-rider owns 5-40 trips that share a coherent home
zone and commute pattern, and every per-trip observable (fares, tips, waits,
surge ratios) is real, but:

- individual-level statements (e.g. "this rider's personal elasticity") are
  modeling artifacts;
- only distributional statements (e.g. "riders who tip less are more price
  sensitive") are grounded in the real data;
- rider IDs do not correspond to any real account and cannot be linked
  across months.

Treat `riders.parquet` as a *sampling device* over the real trip distribution,
not as a recovered panel of individuals. This construction is documented
honestly in the `build_riders` docstring.

### 3. Personas (`rider_sim/personas.py`)

`RiderPersona` is a Pydantic model with `rider_id`, `home_zone`, `work_zone`,
`trip_purpose_mix`, `price_sensitivity`, `wait_tolerance_minutes`,
`income_bracket`, `has_transit_alternative`, `loyalty_tier`,
`observed_trip_count`, `median_fare_paid`, `median_wait_experienced`.

`sample_personas(n, seed)` fits each persona **from that pseudo-rider's own
empirical trip stats** — nothing is hand-invented:

| trait | derivation |
| --- | --- |
| `price_sensitivity` | `1 - (0.6 * fare-per-mile percentile + 0.4 * tip rate)` — revealed preference |
| `wait_tolerance_minutes` | `max(p75 accepted wait, 1.25 * median accepted wait)` — a lower bound, since every real trip was accepted |
| `income_bracket` | quartile of the rider's median fare paid (spend proxy) |
| `has_transit_alternative` | median trip < 2.5 mi *and* price sensitivity ≥ 0.55 (derived heuristic) |
| `loyalty_tier` | trip frequency (≥25 platinum, ≥15 gold, ≥8 silver, else member) |
| `trip_purpose_mix` | zone/hour classification of their real trips (airport zones 1/132/138, night = social, weekday peaks = commute, rest = errand) |

Persists to `data/processed/personas.parquet`.

### 4. Simulation (`rider_sim/simulate.py`)

Generates counterfactual offers from each persona's *own real trips* (real
zone pair, hour, miles, fare, wait) with price (×0.8/×1.0/×1.25) and wait
(×0.5/×1.0/×2.0) multipliers. Each offer is judged by an LLM (Anthropic
Claude, OpenAI fallback) when a key is set, else by a trait-calibrated logit.
Outputs `data/processed/simulated_choices.parquet`.

### 5. Fidelity evaluation (`rider_sim/eval/`)

The scientific core. `python -m rider_sim evaluate --run-id <id> --ablations all`
runs every ablation config over a counterfactual offer grid derived from real
trips, stores decision traces under `data/processed/traces/<run_id>/`, and
renders `reports/fidelity_<run_id>.md` plus figures. Every number is computed
from the data and traces; nothing is hardcoded.

- **Discriminator** — featurizes real and simulated decision sequences
  (accept rate, fare-vs-market percentile, accepted wait, surge elasticity,
  purpose mix, action entropy, run lengths), trains a LightGBM classifier with
  grouped K-fold by rider_id, and reports AUC with a bootstrap 95% CI. AUC
  near 0.5 = indistinguishable; near 1.0 = trivially separable. Feature
  importances name the failure mode.
- **Calibration** — reliability curves and Brier score of predicted accept
  probability vs real revealed acceptance (base offers are trips real riders
  took; ×1.8-fare counterfactuals are labeled rejects), segmented by price
  sensitivity tercile, wait tolerance tercile, and transit alternative.
- **Distributions** — two-sample KS tests of accepted fare, accepted wait,
  accept rate by hour, and purpose mix against the real marginals, with
  Bonferroni-adjusted verdicts.
- **Mechanism** — log-log elasticity of accept rate w.r.t. surge with a
  bootstrap CI, checked for sign and magnitude against the literature range
  below.
- **Ablations** — full agent, no memory, demographics-only persona, no tool
  use, prompt v1 vs v2, logit baseline, random baseline; one table of
  config × {AUC, Brier, KS passed, elasticity verdict, cost per 1k decisions},
  with permutation tests for AUC differences.

Without API keys, the evaluator falls back to a deterministic offline backend
(pipeline validation only); set `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` for real
LLM configs. Reruns are free via the DuckDB LLM response cache.

### Literature benchmark

The mechanism check compares simulated surge elasticity against:

> Cohen, Hahn, Hall, Levitt & Metcalfe (2016), *Using Big Data to Estimate
> Consumer Surplus: The Case of Uber*, NBER Working Paper 22627. Own-price
> elasticity of demand for Uber rides: roughly **-0.4 to -0.6**.

The simulated accept-rate elasticity must be negative (sign PASS) and its
bootstrap CI must overlap this range (magnitude PASS).

## Development

```bash
make lint   # ruff check + format check + mypy (strict)
make test   # pytest
```

Tests cover persona schema validity, seed-deterministic sampling, and the
evaluation pipeline with known ground truth (identical distributions →
AUC CI covers 0.5; shifted sims → KS rejects; synthetic elasticities → the
mechanism verdict is correct).
All pipeline steps are deterministic for a fixed seed.
