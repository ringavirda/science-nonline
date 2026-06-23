# API: streaming / online estimators

Online estimators that ingest **one sample at a time** at bounded per-update cost,
for control loops and live streams. Symbolic work (model + Jacobian) is done once
at construction; each update is pure NumPy. Concept:
[../guides/methods-explained.md#streaming](Guides-Methods-Explained#streaming);
math: [../methods/equal_areas_filter.md](Methods-Equal-Areas-Filter).

- [`EACFilter`](#edafilter) -- area-measurement filter (cheaper; monotone/saturating)
- [`LSIFilter`](#lsifilter) -- spectrum-measurement filter (oscillatory plants)
- [`FilterBank`](#filterbank) -- many streams in lockstep
- [`FusedChiSquareDetector`](#fused) -- pooled multi-stream fault detection

Each filter is the **streaming twin of a batch method**: `EACFilter` <->
[`fit_eac`](API-Fitting#fit_eac), `LSIFilter` <-> [`fit_lsi`](API-Fitting#fit_lsi).

---

<a name="edafilter"></a>
## `EACFilter`

```python
EACFilter(expr, var, *, p0=None, window_size=50, q_diag=None, r=1.0,
          alpha=0.001, cusum_k=0.5, cusum_h=5.0, n_sub=1,
          adapt_r=False, drift_reset="full", drift_inflation=100.0)
```

Kalman-style recursive estimator whose **measurement is the area innovation**
(data minus model, integrated over a sliding window). Tracks time-varying
parameters and detects drift.

**Constructor arguments**

| name | default | meaning |
|---|---|---|
| `expr`, `var` | -- | model expression and main variable, e.g. `"A*sin(w*t)"` |
| `p0` | `None` | initial parameter estimate (defaults to ones) |
| `window_size` | `50` | sliding-window length for the area integration (smoothing vs responsiveness) |
| `q_diag` | `None` | per-parameter process-noise variances; larger => faster drift. Defaults to 0.01 each |
| `r` | `1.0` | measurement-noise variance of the area innovation |
| `alpha` | `0.001` | significance level for the NIS sudden-jump test |
| `cusum_k` | `0.5` | CUSUM slack in innovation sigma -- smallest sustained shift to ignore; `inf` disables CUSUM |
| `cusum_h` | `5.0` | CUSUM decision threshold; larger => fewer false alarms, slower detection |
| `n_sub` | `1` | sub-window area measurements per step; `>1` gives a *vector* measurement (better observability for coupled multi-parameter models) |
| `adapt_r` | `False` | adapt `R` online from an EWMA of the squared innovation (Mehra-style); pair with `n_sub>1` |
| `drift_reset` | `"full"` | on a detected drift, `"full"` resets covariance + clears the window; `"inflate"` multiplies covariance by `drift_inflation` and keeps the estimate (gentler) |
| `drift_inflation` | `100.0` | covariance inflation factor for `drift_reset="inflate"` |

**Methods & attributes**

- `partial_fit(t_new, y_new) -> self` -- ingest one sample; returns early until the
  window fills. `update` is an alias (recursive-filter naming).
- `predict(x) -> ndarray` -- evaluate the current model at `x`.
- `params_ -> dict[str, float]` -- current parameter estimate.
- `inflate(factor=None) -> None` -- manually inflate the covariance (the re-arm hook).
- `n_drifts_` -- drifts detected so far; `drift_flag_` -- `True` only on the exact
  step a drift fires; `last_drift_direction_` -- `+1` up / `-1` down / `0` none;
  `last_residual_` -- most recent one-step innovation (NaN until the window fills).

```python
from dtfit import EACFilter
flt = EACFilter("a*exp(b*t)", "t", window_size=40)
for t, y in stream:
    flt.partial_fit(t, y)
    if flt.drift_flag_:
        print("regime change at", t, "dir", flt.last_drift_direction_)
print(flt.params_)
```

---

<a name="lsifilter"></a>
## `LSIFilter`

```python
LSIFilter(expr, var, *, p0=None, window_size=50, order=5, q_diag=None, r=1.0,
          alpha=0.001, cusum_k=0.5, cusum_h=5.0,
          adapt_r=False, drift_reset="full", drift_inflation=100.0)
```

Drop-in sibling of `EACFilter` with the same `partial_fit` / `predict` / `params_`
API, but the measurement is the window's **Legendre spectrum** (the first
`order+1` coefficients) instead of a single area. The spectrum captures
oscillations the area criterion partly cancels, so **use `LSIFilter` for
oscillatory plants**.

Differs from `EACFilter` only in:

| name | default | meaning |
|---|---|---|
| `order` | `5` | Legendre spectral order; measurement is the first `order+1` coefficients (richer observability, larger measurement vector). Clamped so `order+1 <= window_size` |
| `r` | `1.0` | *base* measurement-noise variance; per-coefficient variance is `r*(2j+1)` (the LSI orthonormal weighting, down-weighting noisier high orders) |

(No `n_sub` -- the spectral order plays that role.)

---

<a name="filterbank"></a>
## `FilterBank`

```python
FilterBank(filters)
FilterBank.from_model(expr, var, n_streams, *, filter_cls=EACFilter, **kwargs)
```

A bank of `K` independent streaming filters updated in lockstep -- e.g. the x/y/z
axes of a trajectory, or a sensor array. Build it from a model with
`from_model` (forwards `**kwargs` to each filter's constructor; `filter_cls` is
`EACFilter` or `LSIFilter`).

**Methods & attributes**

- `partial_fit(t, y, *, n_jobs=1) -> self` -- ingest one sample **per stream**
  (`t` shared scalar or per-stream array; `y` length `K`). `n_jobs>1` fans the
  updates across a thread pool (wins when `K` and the window are large).
- `run(t_seq, Y, *, n_jobs=1, track=False) -> dict` -- drive every stream over a
  whole block (`Y` is `(n_steps, K)`); each worker thread runs a disjoint subset
  of streams to completion (no per-step barrier -- the throughput primitive).
  Returns `{"params": (K, n_params), "n_drifts": (K,), ["track": (n_steps, K)]}`.
- `params_array()` / `params_` / `drift_flags_` -- collected per-stream state.
- `predict(x)` -- per-stream predictions.
- `fused_detector(**kwargs) -> FusedChiSquareDetector` -- attach the pooled detector.
- `len(bank)`, `bank[i]` -- size and indexing.

```python
from dtfit import FilterBank, LSIFilter
bank = FilterBank.from_model("A*sin(w*t)", "t", 3, filter_cls=LSIFilter, window_size=64)
out = bank.run(t_seq, Y, n_jobs=3)        # Y is (n_steps, 3)
```

---

<a name="fused"></a>
## `FusedChiSquareDetector`

```python
FusedChiSquareDetector(bank, *, alpha=1e-4, inflate=4.0, ewma=0.9,
                       warmup=None, cooldown=None)
```

Pools a [`FilterBank`](#filterbank)'s one-step innovations into a **fused** fault
test. A fault that moves *every* stream (a shared structural change) is weak in any
single stream but strong in the sum across streams. Each filter's residual is
normalized by an online EWMA of its variance and the squares are summed into a
`chi2(K)` statistic; exceeding the `alpha`-level threshold flags a fault and
(optionally) re-arms each filter via `inflate` so the bank re-adapts.

| name | default | meaning |
|---|---|---|
| `bank` | -- | the `FilterBank` to drive (filters must expose `last_residual_`, `W`, `inflate` -- both stock filters do) |
| `alpha` | `1e-4` | per-step false-alarm probability; threshold is `chi2.ppf(1-alpha, df=K)` |
| `inflate` | `4.0` | covariance re-arm factor on detection (`<=1` disables the re-arm; flag still raised) |
| `ewma` | `0.9` | decay for the per-stream innovation-variance estimate |
| `warmup` | `None` | steps before detecting (default `3 x window`) |
| `cooldown` | `None` | steps to suppress detection after a flag (default one `window`) |

- `update(t, y) -> bool` -- ingest one sample per stream; returns `True` iff this
  step raises a fault. `statistic_` (current chi^2), `flag_`, `flags_` (flagged step
  indices), `n_flags_`, `predict(x)`.

```python
bank = FilterBank.from_model(model, "t", n_axes, filter_cls=LSIFilter)
det = bank.fused_detector(alpha=1e-4, inflate=4.0)
for i, (t, y) in enumerate(stream):       # y length K
    if det.update(t, y):
        handle_fault(i, det.statistic_)
```
