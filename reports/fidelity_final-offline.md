# Fidelity evaluation

- run_id: `final-offline`
- provider: `offline`
- seed: 7
- offers generated: 2,400 across 200 riders
- labeled offers for calibration: 800

## Ablation table

| config | AUC | AUC CI | Brier | KS passed | elasticity | elast CI | elast verdict | $/1k decisions | n decisions |
|---|---|---|---|---|---|---|---|---|---|
| demographics_only | 1.000 | [0.999, 1.000] | 0.343 | 1/4 | -0.651 | [-0.814, -0.537] | PASS | 0.00 | 2400 |
| full | 1.000 | [1.000, 1.000] | 0.340 | 1/4 | -0.734 | [-0.861, -0.631] | FAIL | 0.00 | 2400 |
| logit_baseline | 0.998 | [0.994, 1.000] | 0.120 | 1/4 | -1.013 | [-1.362, -0.671] | FAIL | 0.00 | 2400 |
| no_memory | 1.000 | [1.000, 1.000] | 0.340 | 1/4 | -0.734 | [-0.861, -0.631] | FAIL | 0.00 | 2400 |
| no_tools | 1.000 | [1.000, 1.000] | 0.340 | 1/4 | -0.734 | [-0.861, -0.631] | FAIL | 0.00 | 2400 |
| prompt_v2 | 1.000 | [1.000, 1.000] | 0.340 | 1/4 | -0.734 | [-0.861, -0.631] | FAIL | 0.00 | 2400 |
| random_baseline | 1.000 | [1.000, 1.000] | 0.312 | 2/4 | -0.187 | [-0.455, 0.074] | PASS | 0.00 | 2400 |

![figure](figures/final-offline/ablations.png)

## Discriminator (real vs simulated sequences)

Grouped K-fold by rider_id; pooled OOF AUC with bootstrap 95% CI. AUC near 0.5: indistinguishable. AUC near 1.0: trivially separable.

Top features for the full config:

```json
{
  "run_accept_mean": 1702.5415644664317,
  "log_n_decisions": 609.2287339121103,
  "surge_accepted_mean": 183.6132528019807,
  "accept_rate": 146.61837662011385,
  "action_entropy": 127.11823435872793,
  "run_accept_max": 111.95752761140466,
  "wait_accepted_median": 8.571515383198857,
  "surge_elasticity": 3.047969937324524,
  "fare_pctl_accepted": 0.0,
  "purpose_commute": 0.0,
  "purpose_social": 0.0,
  "purpose_airport": 0.0
}
```

![figure](figures/final-offline/importances.png)

## Calibration (Brier + reliability vs real revealed acceptance)

- overall Brier (full config): 0.340

![figure](figures/final-offline/calibration.png)

## Distributions (KS vs real marginals, full config)

| marginal | statistic | p-value | Bonferroni alpha | verdict |
|---|---|---|---|---|
| accepted_fare | 0.100 | 0.000000 | 0.0125 | fail |
| accepted_wait | 0.093 | 0.000000 | 0.0125 | fail |
| accept_rate_by_hour | 1.000 | 0.000000 | 0.0125 | fail |
| purpose_mix | 0.029 | 0.360508 | 0.0125 | pass |

Purpose-mix KS is reported with the caveat that KS is not distribution-free for categorical data; the category order is computed from the real purpose frequencies.

![figure](figures/final-offline/distributions.png)

## Mechanism: surge elasticity vs literature

- literature range: (-0.6, -0.4) (Cohen, Hahn, Hall, Levitt & Metcalfe (2016), 'Using Big Data to Estimate Consumer Surplus: The Case of Uber', NBER Working Paper 22627)
- full config elasticity: -0.734 (95% CI [-0.861, -0.631])
- sign PASS: True; magnitude PASS: False; verdict: **FAIL**

![figure](figures/final-offline/elasticity.png)

## Ablation significance (permutation p-values for AUC differences)

| | demographics_only | full | logit_baseline | no_memory | no_tools | prompt_v2 | random_baseline |
|---|---|---|---|---|---|---|---|
| demographics_only | - | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 | 0.881 |
| full | 1.000 | - | 0.886 | 1.000 | 1.000 | 1.000 | 1.000 |
| logit_baseline | 0.930 | 0.886 | - | 0.368 | 0.368 | 0.368 | 0.468 |
| no_memory | 1.000 | 1.000 | 0.368 | - | 1.000 | 1.000 | 1.000 |
| no_tools | 1.000 | 1.000 | 0.368 | 1.000 | - | 1.000 | 1.000 |
| prompt_v2 | 1.000 | 1.000 | 0.368 | 1.000 | 1.000 | - | 1.000 |
| random_baseline | 0.881 | 1.000 | 0.468 | 1.000 | 1.000 | 1.000 | - |
