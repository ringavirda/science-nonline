# Ensemble — overlapping-window robust aggregation

> Numeric batch method (a *composition* of EDA / LSI). Source:
> [`methods/_ensemble.py`](../../packages/dtfit/src/dtfit/methods/_ensemble.py).
> Invoke via `ensemble_fit(x, y, expr, var, ...)`. Promoted from the experimental
> adaptations (#3) after the validation suite showed a consistent win on
> outlier-contaminated data.

Fitting one model to the whole record gives a single estimate with **full
exposure to outliers**: a handful of corrupted samples pull the least-squares fit
off the true parameters. `ensemble_fit` instead fits the model on many
**overlapping subwindows** and aggregates the per-window coefficients robustly —
this is *bagging over the time axis*.

## How it works

For a record of $n$ samples, slide a window of width $w$ with stride
$s = \frac{n}{M}(1-\text{overlap})$, fitting the model on each window with the
chosen base method (EDA by default, or LSI). This yields a set of per-window
coefficient vectors $\{\hat\theta_1,\dots,\hat\theta_K\}$, aggregated as

$$
\hat\theta \;=\; \operatorname*{median}_k \hat\theta_k,
\qquad
\sigma_j \;=\; \operatorname{std}_k\big(\hat\theta_{k,j}\big),
$$

The **median** is the robustness mechanism: a window that happens to contain
outliers produces a deviant $\hat\theta_k$, and the coordinate-wise median
**rejects** it as long as most windows are clean. The inter-window standard
deviation $\sigma$ is a cheap empirical uncertainty band (it populates the result
covariance, so `stderr()` and `predict(return_std=True)` work).

This rests on the same property the [scale backends](scaling.md) use — that a fit
restricted to a sub-interval is still a valid (if higher-variance) estimate of the
*same* global parameters — combined with the breakdown-resistance of the median.

## Algorithm

1. **Window** the record into $M$ overlapping spans (`n_windows`, `overlap`).
2. **Fit** the model on each window with the base `method` (`"eda"`/`"lsi"`),
   forwarding `p0` and any extra fitter `kwargs` (e.g. `bounds`). A window whose
   fit fails (e.g. too few clean points) is simply **skipped**.
3. **Aggregate** the surviving per-window coefficients by `median` (default) or
   `mean`; compute the inter-window spread.
4. **Return** an [`EnsembleResult`](../api/fitting.md#ensembleresult) — a full
   [`FittingResult`](../api/types.md) (named `params`, `predict`, `to_dict`) that
   also carries the raw `members` and their `spread`.

## When to use it

**Use the ensemble for outlier-contaminated data.** It is the recommended tool
when a fraction of samples are spikes / dropouts / sensor glitches, for two
reasons:

- it needs **no tuning** — unlike [`fit_eda(loss="soft_l1", f_scale=…)`](eda.md),
  whose robust loss acts on *window-area* residuals and only bites when `f_scale`
  is matched to the (small) residual scale, the median aggregation is
  parameter-free;
- it **stays stable** where the robust loss can diverge (a poorly-scaled robust
  loss returned a 1000%+ error on a logistic in the validation runs; the ensemble
  did not).

**Do not** reach for it on clean (Gaussian-noise) data: a whole-record fit (or the
shape-routed [`auto_estimate`](auto.md)) is more accurate there. The ensemble
trades a little accuracy on clean data for breakdown resistance under
contamination — a **specialised tool, not the default path**.

## Measured robustness

Recovery error (max relative parameter error, median over 6 noise draws) on the
[validation corpus](../../packages/dtfit/tests/accuracy/) with **4 % of samples
replaced by 8σ spikes**, plain whole-record EDA vs the ensemble:

| family | plain EDA | **ensemble** |
|---|---|---|
| exponential | 0.125 | **0.027** |
| exp_decay | 0.117 | **0.052** |
| power_law | 0.282 | **0.053** |
| michaelis_menten | 0.153 | **0.067** |
| first_order | 0.111 | **0.024** |
| *pooled median* | 0.143 | **0.070** |
| *pooled mean* | 1.56 | **0.18** |

The ensemble roughly halves the typical error and cuts the *mean* error ~8× —
the latter because, unlike a single fit, it never blows up on a bad window. This
is gated in
[`tests/validation/test_outlier_robustness.py`](../../packages/dtfit/tests/validation/test_outlier_robustness.py).

## Where it is best applied

**Use the ensemble for:** outlier-/glitch-contaminated records where you still
want a point estimate and a cheap uncertainty band, with no robust-loss tuning.
Pick `method="eda"` (default, fastest/most robust base) for transient/saturating
shapes, `method="lsi"` when the base fit benefits from the spectral criterion.

**Prefer instead:** a single [`fit_eda`](eda.md) / [`fit_lsi`](lsi.md) (or
[`auto_estimate`](auto.md)) on clean data; the recursive
[`EDAFilter`](equal_areas_filter.md) for *streaming* drift rather than batch
contamination.
