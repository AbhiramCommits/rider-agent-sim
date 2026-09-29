# Product screening

- run_id: `final-offline` (trace config: `full`)
- generated: 2026-09-28T22:15:45
- control accept rate (base cell): 0.963
- traffic assumption: real HVFHV trip volume for 2024-01 (computed from raw data)

## Decision rule

> Screen in iff the CUPED-adjusted accept-rate lift's lower 95% CI bound is > 0 (raw CI used when CUPED does not reduce variance), and -- for cells with a back-tested counterpart -- the back-test does not contradict the direction (verdict != FAIL).

## Ranked intervention cells (primary metric: accept-rate lift)

| rank | cell | lift (raw) | lift (CUPED) | CI (effective) | SE rider | SE model | var reduction | completed | fare ($) | abandonment | n/arm | days | recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | discount_20pct | 0.000 | 0.000 | [-0.079, 0.079] | 0.000 | 0.040 | 0.00 | 0.000 | -5.19 | 0.000 | n/a | n/a | screen out |
| 2 | price_up_25pct | 0.000 | 0.000 | [-0.079, 0.079] | 0.000 | 0.040 | 0.00 | 0.000 | 6.48 | 0.000 | n/a | n/a | screen out |
| 3 | price_up_80pct | 0.000 | 0.000 | [-0.079, 0.079] | 0.000 | 0.040 | 0.00 | 0.000 | 20.75 | 0.000 | n/a | n/a | screen out |
| 4 | surge_2x | -0.455 | -0.455 | [-0.561, -0.349] | 0.033 | 0.041 | 0.00 | -0.455 | 0.00 | 0.455 | 14 | 0.000 | screen out |
| 5 | surge_2x_price_up | -0.455 | -0.455 | [-0.561, -0.349] | 0.033 | 0.041 | 0.00 | -0.455 | 5.74 | 0.455 | 14 | 0.000 | screen out |

CIs are lift +/- 1.96 x total SE; the effective CI is the CUPED-adjusted one when CUPED reduces variance, else the raw one. SE rider = population variance from a rider-level bootstrap; SE model = LLM sampling variance from a parametric Bernoulli(p_accept) bootstrap of the decision layer. The two variance components are reported separately so the reader can see how much noise is the population vs the model. `completed` and `abandonment` are complements of the accept metric (no post-accept churn is modeled); `fare` is the mean accepted fare, in dollars.

## Back-test (held-out real behavior)

Holdout specification (auditable, from `config.BACKTEST_HOLDOUT`):

```json
{
  "type": "surge_band_demand_ratio",
  "high_band": [
    2.0,
    3.0
  ],
  "low_band": [
    1.0,
    1.2
  ]
}
```

- real log demand ratio (high vs low surge band shares): -1.360 (shares 0.0478 vs 0.1864)
- simulated log accept-rate ratio: -0.517 (95% CI [-0.586, -0.445], n=770+1192 offers)
- signed error (sim - real): 0.844
- directional agreement: True
- simulated CI covers real estimate: False
- verdict: **PARTIAL**

> The real ratio reflects the equilibrium allocation of rides (demand and supply together), not a controlled experiment; exact agreement is not expected -- the back-test is a directional sanity gate.

## Power (real A/B test, 80% power, alpha 0.05)

Sample sizes use the closed-form two-proportion formula on the screening lift vs the control accept rate; days assume a 50/50 split of the real daily trip volume.
