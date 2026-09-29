# Methods: estimators and assumptions

This document specifies the estimators used by the fidelity evaluation and
screening layers, with the assumptions each relies on. Notation: a *rider*
is a pseudo-rider (see below); an *offer* is one counterfactual ride offer; a
*decision* is one (offer, action, confidence, predicted-accept-probability)
tuple stored in the trace.

## 0. The pseudo-rider panel (assumption carried everywhere)

Real TLC trips carry no rider ID. The panel is constructed by assigning
5-40 real trips that share a commute cell (pickup zone, hour-of-day bucket,
weekday/weekend, trip-miles bucket) to each pseudo-rider. **Assumption:**
trips are exchangeable within a commute cell; pseudo-riders are a sampling
device over the real trip distribution, not recovered individuals. All
downstream estimators inherit this caveat: they are valid for the panel's
trip distribution, not for named real people.

## 1. Discriminator (real vs simulated sequences)

**Featurization.** Each rider sequence is summarized by 12 computed features:
accept rate, mean fare-vs-market percentile of accepted offers (market = ECDF
of real fares), median accepted wait, a per-sequence surge elasticity
(logistic slope of accepted on log-surge, 0 when degenerate), purpose shares
(commute/social/airport), action entropy over the 4-action support, mean and
max accepted run lengths, mean surge of accepted offers, and log decision
count.

**Estimation.** LightGBM classifier, grouped K-fold (5 folds, groups =
(label, rider_id), so no rider appears in both train and test folds). The
reported AUC is the pooled out-of-fold AUC; the 95% CI is a percentile
bootstrap over (label, OOF score) pairs. **Assumptions:** (i) sequences are
the experimental unit; (ii) bootstrap replicates are iid draws of OOF pairs;
(iii) an AUC near 0.5 means indistinguishable (the null is exchangeability of
the real/sim labels -- this is what the identical-distributions test
exercises). Feature importances are gain importances of the final model
trained on all data; they name failure modes, they do not infer causality.

## 2. Calibration (Brier + reliability curves)

Ground-truth labels come from the real data: base offers (price/surge
multiplier 1.0) reproduce trips the real rider took, so y = 1; the x1.8-fare
counterfactuals are labeled y = 0. Intermediate variants have no honest label
and are excluded. **Assumption:** the x1.8 counterfactual would have been
rejected -- documented as a construction, since real TLC records contain no
rejects. Predicted accept probability per config: LLM configs map the
decision to p = confidence if accept, else 1 - confidence; the logit baseline
uses its fitted P(accept); the random baseline uses 1/4 (its uniform policy).
The Brier score is mean((p - y)^2) over labeled offers; reliability curves
are sklearn `calibration_curve` bins (uniform strategy, bin count derived
from sample size). Segments are computed terciles of price sensitivity and
wait tolerance over the trace's riders, plus the transit-alternative flag.

## 3. Distributional KS tests

Four marginals, each a two-sample Kolmogorov-Smirnov test (scipy): accepted
fare, accepted wait, accept rate by hour (per-hour rates; the real rate is
1.0 in every hour because every real trip was accepted -- computed, not
assumed), and purpose mix (per-event purpose codes, category order computed
from the real purpose frequencies). Real samples are matched in size to the
simulated sample (seeded draws) so power is comparable across marginals.
Verdicts use Bonferroni alpha = 0.05/4. **Assumption:** KS is exact only for
continuous distributions; the purpose-mix marginal is categorical and is
reported with that caveat. **Assumption:** matched-size subsampling of the
real data does not distort the marginal.

## 4. Mechanism: surge elasticity

The trace is binned into surge bands (quantile edges, at most 8); band accept
rates are Laplace-smoothed (rate = (k + 0.5)/(n + 1)); the elasticity is the
OLS slope of log-rate on log-band-center. The 95% CI is a percentile
bootstrap that resamples decision rows and refits the slope. Sign PASS
requires a negative slope; magnitude PASS requires CI overlap with the
literature range [-0.6, -0.4] (Cohen et al., 2016, NBER WP 22627, documented
in the README). **Assumptions:** (i) accept rate is a demand quantity so its
log-log slope is a demand elasticity; (ii) the band aggregate captures the
policy's surge response; (iii) the literature range is a valid external
anchor for own-price demand elasticity.

## 5. Ablations and the permutation test

Configs: full (prompt v1), prompt v2, no memory, demographics-only persona,
no tool use, logit baseline, random baseline -- identical offer sets. For
each pair of configs A and B, the AUC difference is tested by a two-sided
permutation test: the two configs' simulated OOF scores are pooled, config
labels are permuted, and the AUC difference recomputed against the shared
real scores; p = (count(perm >= observed) + 1)/(n_perm + 1).
**Assumptions:** (i) permutation units are sequences; (ii) the real score set
is fixed across configs (true in the runner).

## 6. Screening: cell lifts and variance decomposition

Each intervention cell is a price/surge variant of the same base trips;
control is the base cell. For each rider the paired delta is computed on the
experiment window (deterministic pre/post split of each cell's offers by
offer id). Two variance components, reported separately:

- population: nonparametric bootstrap over riders (resample paired deltas),
- model: parametric bootstrap of the decision layer -- each decision is
  redrawn as Bernoulli(p_accept) -- capturing LLM sampling noise.

Total SE = sqrt(SE_rider^2 + SE_sampling^2); CI = lift +/- 1.96 * SE_total.
**Assumptions:** (i) the paired design removes base-trip heterogeneity;
(ii) decision draws are conditionally independent given p_accept; (iii) the
components are independent.

## 7. CUPED

Covariate x_i = rider's pre-period control metric (accept rate, and the same
metric for secondaries). theta = cov(delta, x)/var(x); adjusted delta =
delta - theta (x - x_bar); variance reduction = 1 - var(adjusted)/var(delta),
and the adjusted CI is computed with the same two-component bootstrap with
theta fixed. **Assumptions:** the pre-period control metric is a valid
pre-treatment covariate (no treatment effect in the pre period -- true by
construction, since pre-period offers are identical across cells) and theta
is estimated without material sampling error in the covariate.

## 8. Power (real A/B sizing)

Per-arm sample size is the closed-form two-proportion formula with z-scores
from the normal quantile function at the requested alpha/power. Days =
n_per_arm / (daily_offers * traffic_share), where daily_offers is the mean
daily trip count of the raw TLC month (computed via DuckDB) and
traffic_share = 0.5. **Assumptions:** (i) the simulated lift transfers to a
real experiment; (ii) the test unit is one ride offer; (iii) two-sided
two-proportion approximation is adequate.

## 9. Back-test

Holdout (auditable in `config.BACKTEST_HOLDOUT`): the real demand response
across two surge bands [1.0, 1.2) vs [2.0, 3.0), expressed as the log ratio
of real trip-count shares. The simulated counterpart is the log ratio of
Laplace-smoothed accept rates over trace offers in the same bands, with a
rider-level bootstrap CI. Verdicts: PASS (direction agrees and CI covers the
real value), PARTIAL (direction agrees, CI misses), FAIL (direction
disagrees). **Assumptions / caveats:** the real ratio reflects the
equilibrium allocation (demand and supply together), not a controlled
experiment; exact agreement is not expected -- the back-test is a directional
sanity gate, and a documented miss is a result. Surge never enters persona
fitting, so the relationship is genuinely held out.
