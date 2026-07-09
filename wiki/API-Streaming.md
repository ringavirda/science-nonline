# API: streaming / online estimators

Online estimators that ingest **one sample at a time** at bounded per-update cost,
for control loops and live streams. Symbolic work (model + Jacobian) is done once
at construction; each update is pure NumPy. Concept:
[../guides/methods-explained.md#streaming](Guides-Methods-Explained#streaming);
math: [../methods/equal_areas_filter.md](Methods-Equal-Areas-Filter).

- [`EACFilter`](#eacfilter) -- area-measurement filter (cheaper; monotone/saturating)
- [`LSIFilter`](#lsifilter) -- spectrum-measurement filter (oscillatory plants)
- [`FilterBank`](#filterbank) -- many streams in lockstep
- [`FusedChiSquareDetector`](#fused) -- pooled multi-stream fault detection

(The inverse-covariance `InformationFilter` fusion primitive is **no longer part
of the stable streaming surface** -- it was demoted to `dtfit-experimental`;
import it as `from dtfit_experimental import InformationFilter`.)

Each filter is the **streaming twin of a batch method**: `EACFilter` <->
[`fit_eac`](API-Fitting#fit_eac), `LSIFilter` <-> [`fit_lsi`](API-Fitting#fit_lsi).

---

<a name="eacfilter"></a>
## `EACFilter`

```python
EACFilter(expr, var, *, regressors=None, param_names=None, p0=None,
          window_size=50, min_window=None, adaptive_window=False,
          window_tol=0.02, q_diag=None, r=1.0, alpha=0.001, cusum_k=0.5,
          cusum_h=5.0, n_sub=1, adapt_r=False, robust=False, huber_c=3.0,
          drift_reset="full", drift_inflation=100.0)
```

Kalman-style recursive estimator whose **measurement is the area innovation**
(data minus model, integrated over a sliding window). Tracks time-varying
parameters and detects drift.

**Quick-start presets** -- most callers should start from a preset classmethod
and only pass overrides, rather than tuning the full knob set below:

- `EACFilter.tracking(expr, var, **overrides)` -- responsive auto-sized window for
  drifting parameters. Equivalent to `EACFilter(expr, var, adaptive_window=True,
  ...)`; any keyword in `overrides` wins over the preset.
- `EACFilter.robust(expr, var, **overrides)` -- outlier/anomaly-resilient gains.
  Equivalent to `EACFilter(expr, var, robust=True, adapt_r=True,
  drift_reset="inflate", ...)`; `overrides` win.

**Constructor arguments**

| name | default | meaning |
|---|---|---|
| `expr`, `var` | -- | model and main variable, e.g. `"A*sin(w*t)"`. `expr` may be a SymPy-expression **string**, a `sympy.Expr`, or a plain Python **callable** `f(t, *params)`; a string/expression may also reference [external regressors](#regressors). A callable is evaluated numerically (no symbolic form) — but has no closed-form time derivatives, so [`coast`](#coast) / [`coast_cov`](#coast_cov) are **unavailable** for it and external regressors are not supported. `var` is a label only for a callable |
| `regressors` | `None` | name(s) of [external-regressor](#regressors) channels appearing in `expr` (everything else free is a parameter); when given, each `partial_fit` / `predict` supplies the regressor value(s) for that sample. **Symbolic models only** (a callable rejects regressors) |
| `param_names` | `None` | for a **callable** model, the parameter names in **signature order** (those after the leading `t`); introspected from the callable's signature when omitted. Ignored for a symbolic model, whose parameters come from the expression |
| `p0` | `None` | initial parameter estimate (defaults to ones); ordered like `params_` — sorted names for a symbolic model, signature order for a callable |
| `window_size` | `50` | target (**maximum**) sliding-window length for the area integration; the window **grows** from `min_window` up to this size as samples arrive, then slides. Larger => smoother/more rigid, smaller => more responsive |
| `min_window` | `None` | smallest window at which the filter starts producing an estimate (the area measurement is accumulative, so it does not idle until `window_size` samples arrive). Defaults to **half** `window_size` (a scalar area over few noisy points is unreliable); with `adaptive_window` a small floor so the window can collapse here on a drift. Clamped to `[2*n_sub, window_size]` |
| `adaptive_window` | `False` | size the window **automatically from the data**, using `window_size` as the maximum: grow from `min_window` while the state covariance is still shrinking, collapse back to `min_window` on a detected drift. Best-effort for the area filter -- it cannot size a *global*-parameter model (a polynomial's intercept) well; for robust auto-sizing use [`LSIFilter`](#lsifilter) |
| `window_tol` | `0.02` | covariance-reduction threshold for `adaptive_window` -- the window stops growing once successive updates shrink `trace(P)` by less than this fraction (default 2%) |
| `q_diag` | `None` | per-parameter process-noise variances; larger => faster drift. Defaults to 0.01 each |
| `r` | `1.0` | measurement-noise variance of the area innovation |
| `alpha` | `0.001` | significance level for the NIS sudden-jump test |
| `cusum_k` | `0.5` | CUSUM slack in innovation sigma -- smallest sustained shift to ignore; `inf` disables CUSUM |
| `cusum_h` | `5.0` | CUSUM decision threshold; larger => fewer false alarms, slower detection |
| `n_sub` | `1` | sub-window area measurements per step; `>1` gives a *vector* measurement (better observability for coupled multi-parameter models) |
| `adapt_r` | `False` | adapt `R` online from an EWMA of the squared innovation (Mehra-style); pair with `n_sub>1` |
| `robust` | `False` | gate each update by the normalized innovation -- a window whose per-dof Mahalanobis innovation exceeds `huber_c` has its measurement-noise inflated (gain shrunk) by a Huber weight, so an outlier-corrupted window cannot yank the estimate. The drift detector still sees the raw innovation, so a genuine regime shift is still detected |
| `huber_c` | `3.0` | robust gate threshold in innovation standard deviations (per dof); ~3 keeps clean windows unweighted |
| `drift_reset` | `"full"` | on a detected drift, `"full"` resets covariance + clears the window; `"inflate"` multiplies covariance by `drift_inflation` and keeps the estimate (gentler). Validated at construction — any other value raises `ValueError` |
| `drift_inflation` | `100.0` | covariance inflation factor for `drift_reset="inflate"` |

**Methods & attributes**

- `partial_fit(t_new, y_new, regressors=None) -> self` -- ingest one sample; returns
  early until the window reaches `min_window`. `update` is an alias
  (recursive-filter naming). Pass [`regressors=`](#regressors) iff the model
  declares external regressors. A **non-finite sample** (NaN/inf in `t`, `y`, or a
  regressor) is skipped at entry with a `RuntimeWarning` — it never enters the
  window and the filter resumes on the next good sample (before v0.2 one NaN
  silently stalled updates for up to a full window).
- `predict(x, regressors=None) -> ndarray` -- evaluate the current model at `x`.
- [`coast(x, *, order=1)`](#coast) -- dead-reckon *past the fitted window* (gap
  extrapolation).
- [`coast_cov(x, *, order=1)`](#coast_cov) -- predictive variance of `coast(x)`;
  the band **grows with gap length** (confidence decays while coasting), so a gap
  dead-reckon can be fused as `coast(x) +/- sqrt(coast_cov(x))`.
- [`predict_cov(x, regressors=None)`](#predict_cov) -- predictive variance of the
  model **output** at `x` (the delta method); gives the `predict(x) +/-
  sqrt(predict_cov(x))` band.
- `params_ -> dict[str, float]` -- current parameter estimate.
- `param_cov_ -> ndarray` -- the running parameter covariance `P` (the Kalman
  state covariance), shape `(n_params, n_params)` -- the streaming analogue of
  [`FittingResult.cov`](API-Types). Large early on, it contracts as the parameters
  become identified and re-inflates on a detected drift.
- `stderr_ -> dict[str, float]` -- per-parameter running standard errors (`sqrt`
  of the `param_cov_` diagonal), the online twin of
  [`FittingResult.stderr`](API-Types) -- an uncertainty band on the streamed
  estimate (embedded control, fault detection).
- `inflate(factor=None) -> None` -- manually inflate the covariance (the re-arm hook).
- `n_drifts_` -- drifts detected so far; `drift_flag_` -- `True` only on the exact
  step a drift fires; `last_drift_direction_` -- `+1` up / `-1` down / `0` none;
  `last_residual_` -- most recent one-step innovation (NaN until the window fills).

Both `param_cov_` and `stderr_` are shared by [`LSIFilter`](#lsifilter) (same
Kalman-state surface).

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
LSIFilter(expr, var, *, regressors=None, param_names=None, p0=None,
          window_size=50, min_window=None, adaptive_window=False,
          window_tol=0.001, order=5, q_diag=None, r=1.0, alpha=0.001,
          cusum_k=0.5, cusum_h=5.0, adapt_r=False, adapt_noise=False,
          robust=False, huber_c=3.0, drift_reset="full", drift_inflation=100.0)
```

Drop-in sibling of `EACFilter` with the same `partial_fit` / `predict` / `params_`
API, but the measurement is the window's **Legendre spectrum** (the first
`order+1` coefficients) instead of a single area. The spectrum captures
oscillations the area criterion partly cancels, so **use `LSIFilter` for
oscillatory plants**.

**Quick-start presets** -- as with `EACFilter`, prefer a preset classmethod over
the full knob set:

- `LSIFilter.tracking(expr, var, **overrides)` -- responsive auto-sized window.
  Equivalent to `LSIFilter(expr, var, adaptive_window=True, ...)`; `overrides` win.
- `LSIFilter.robust(expr, var, **overrides)` -- outlier/anomaly-resilient gains.
  Equivalent to `LSIFilter(expr, var, robust=True, adapt_noise=True,
  drift_reset="inflate", ...)`; `overrides` win.

It shares `EACFilter`'s `regressors`, `param_names`, `p0`, `window_size`
(target/**maximum**, window grows from `min_window`), `min_window`,
`adaptive_window`, `q_diag`, `r`, `alpha`, `cusum_k`, `cusum_h`, `adapt_r`,
`robust`, `huber_c`, `drift_reset` and `drift_inflation` arguments — including the
**callable-model** support (`expr` may be a `f(t, *params)` callable, with
`param_names` for a non-introspectable signature; `coast`/`coast_cov` and external
regressors then unavailable) — and the same [`coast`](#coast) /
[`predict_cov`](#predict_cov) / [external-regressor](#regressors) methods. It
differs in:

| name | default | meaning |
|---|---|---|
| `window_tol` | `0.001` | relative-movement threshold for `adaptive_window` (**not** 0.02 as for EAC): the window stops growing once successive updates move the estimate by less than this (EWMA of `|dp|/|p|`, default 0.1%). Its adaptive window is *bidirectional* -- it also **shrinks** when the forecast residual becomes systematically autocorrelated (a lagging fit) -- and, unlike EAC's, is safe for global-parameter models |
| `order` | `5` | Legendre spectral order; measurement is the first `order+1` coefficients (richer observability, larger measurement vector). Clamped so `order+1 <= window_size`. `min_window` defaults to `order + 2` |
| `r` | `1.0` | *base* measurement-noise variance; per-coefficient variance is `r*(2j+1)` (the LSI orthonormal weighting, down-weighting noisier high orders) |
| `adapt_noise` | `False` | set the measurement-noise covariance **entirely** from the data (`R_diag = v * diag(proj @ proj.T)`, `v` an online EWMA of the residual variance) -- the statistically correct self-tuning noise, overriding `r`. Damps the gain on noisy streams, frees it on clean ones; pairs naturally with `robust=True` |

(No `n_sub` -- the spectral order plays that role.)

---

<a name="coast"></a>
## Shared: `coast(x, *, order=1, regressors=None)` -- dead-reckoning past the window

Both filters (via the shared base) expose `coast`, which **extrapolates beyond
the fitted window by dead-reckoning** instead of evaluating the model off its
support. `predict(x)` evaluates the fitted `f(x)` directly -- exact inside the
window the parameters were identified on, but a higher-order model *diverges* once
`x` runs past it (a fitted cubic's `c3*x**3` blows up), so a measurement gap (no
`partial_fit` while `x` advances) turns a good fit into an unbounded extrapolation.

**Symbolic models only.** `coast` (and [`coast_cov`](#coast_cov)) need closed-form
time-derivatives, so a **callable**-backed filter raises `NotImplementedError` —
construct the filter from an expression string to use coasting.

`coast` anchors at the last in-window sample `a = self._t[-1]` and propagates a
Taylor expansion from there:

- `order=1` (default) -- position + velocity `f(a) + f'(a)*(x - a)`
  (constant-velocity / frozen rate); bounded for any model, the safe default.
- `order=2` -- also `+ 0.5*f''(a)*(x - a)**2` (constant acceleration).

It **reduces to `predict` at and before the anchor** (`x <= a`), so it is a
drop-in for the whole track: exact where the window supports `x`, bounded
dead-reckoning past it.

**With [external regressors](#regressors)**, pass `regressors=` the *future*
regressor value(s) at `x` (e.g. an IMU-propagated motion basis). The model is
split into its **extrapolable** part (terms containing a regressor, rolled forward
with the supplied future regressor) and its **nuisance** drift (time-only terms,
which would blow up if extrapolated); only the nuisance drift is dead-reckoned.
This rolls a *fused* model forward using the sensed regressor instead of a crude
finite difference -- the honest way to coast through a GPS/IMU dropout. Without
`regressors`, a regressor model raises `NotImplementedError` (the future regressor
is unknown); call `predict` there instead.

```python
# fused drift + IMU model: coast forward with the caller-propagated accel
future = flt.coast(t_gap, order=1, regressors={"acc": imu_accel_at_t_gap})
```

```python
from dtfit import EACFilter
flt = EACFilter("a + b*t + c*t**2", "t", window_size=40)
for t, y in stream:            # ... fit up to the last received sample
    flt.partial_fit(t, y)
future = flt.coast(t_gap, order=2)   # constant-acceleration coast across a dropout
```

See [Domain: embedded control -- sample dropout](Domain-Embedded-Control) for where
this matters.

---

<a name="predict_cov"></a>
## Shared: `predict_cov(x, regressors=None)` -- output variance for fusion

`predict_cov` returns the **predictive variance of the model output** at `x`,
propagated from the parameter covariance by the delta method:
`Var[f(x)] = J(x)^T P J(x)` where `J(x) = df/dparams`. Where [`stderr_`](#eacfilter)
gives the uncertainty of each *parameter*, `predict_cov` maps that covariance into
*output* space, so the streamed estimate carries a calibrated one-sigma band:

```python
mu = flt.predict(x)
sigma = np.sqrt(flt.predict_cov(x))
band = (mu - sigma, mu + sigma)          # predict(x) +/- sqrt(predict_cov(x))
```

That is what lets a downstream consumer **fuse** the dtfit output (as a
pseudo-measurement with a known variance) or gate on its confidence. It is nearly
free (the parameter Jacobian is already compiled). Note this is the variance from
the *estimate's* uncertainty only -- add the measurement-noise floor separately for
a full predictive interval. With external regressors, `regressors=` supplies their
value(s) at `x` (as for `predict`). Returns an array shaped like `x`.

---

<a name="coast_cov"></a>
## Shared: `coast_cov(x, *, order=1)` -- gap-growing uncertainty band

`coast_cov` is to [`coast`](#coast) what [`predict_cov`](#predict_cov) is to
`predict`: the predictive variance of the dead-reckoned value across a measurement
gap. The coasted value `c(x) = f(a) + f'(a)*dt [+ 1/2 f''(a)*dt^2]` (anchor
`a = last sample`, `dt = x - a`) is a function of the parameters, so its variance
is `J_c(x)^T P J_c(x)` with the coast Jacobian
`J_c[k] = df/dp_k(a) + d f'/dp_k(a)*dt [+ 1/2 d f''/dp_k(a)*dt^2]`. The `dt`/`dt^2`
factors make the band **grow with gap length** -- confidence correctly decays the
longer the filter coasts without a measurement:

```python
mu = flt.coast(x)                       # dead-reckon through the gap
sigma = np.sqrt(flt.coast_cov(x))       # widening band
band = (mu - sigma, mu + sigma)         # honest, gap-growing pseudo-measurement
```

At and before the anchor (`x <= a`) it returns `predict_cov`, so it is a drop-in
for the whole track. Match its `order` to the `coast` call. Not defined for models
with external regressors or for a **callable** model (as `coast`) — both raise
`NotImplementedError`.

---

<a name="regressors"></a>
## Shared: external regressors (exogenous side-channels)

Both filters accept **external regressors** -- measured exogenous signals the model
depends on in addition to `t` (e.g. an IMU-derived motion basis fused into a GPS
track). Declare them at construction and the model becomes `f(t, regressors,
params)`; everything else free in `expr` is still a fitted parameter, and the model
is still scored by the *same* area / Legendre-spectrum measurement:

```python
# expr references the regressor name(s); `S` here is a measured side-channel.
flt = LSIFilter("c0 + c1*t + S", "t", regressors="S")
for t, y, s in stream:
    flt.partial_fit(t, y, regressors={"S": s})   # mapping ...
    # ... or an array/sequence ordered like `regressors`:
    # flt.partial_fit(t, y, regressors=[s])
mu = flt.predict(t_query, regressors={"S": s_query})   # or (len(x), n_reg) array
```

- `regressors=` in the constructor is a name or sequence of names appearing in
  `expr`.
- `partial_fit(t, y, regressors=...)` and `predict(x, regressors=...)` /
  `predict_cov(x, regressors=...)` each take a `{name: value}` **mapping** or a
  value **sequence/array** ordered like the declared `regressors` (an
  `(len(x), n_reg)` array for `predict`). Required iff the model declares
  regressors.
- [`coast`](#coast) is **excluded** for regressor models (the regressor's value
  across a gap is unknown) and raises `NotImplementedError`.

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
- `run(t_seq, Y, *, n_jobs=1, track=False, backend="thread") -> dict` -- drive
  every stream over a whole block (`Y` is `(n_steps, K)`); each worker runs a
  disjoint subset of streams to completion (no per-step barrier -- the throughput
  primitive). Returns
  `{"params": (K, n_params), "n_drifts": (K,), ["track": (n_steps, K)]}`.
  `backend="thread"` (default) fans the streams across worker threads, which only
  overlaps the GIL-released native kernel work -- so for the Python-level
  recursive-filter loop it is usually no faster than serial. `backend="process"`
  instead runs disjoint stream subsets in **separate interpreters** (each with its
  own GIL), so the per-sample Python work runs genuinely concurrently; it wins on
  large workloads (many streams x long records) where the compute dwarfs the
  spawn/pickling overhead. The process backend needs `n_jobs>1` and a bank built
  via `from_model` (so the filters are reconstructable in the workers); it falls
  back to threads otherwise.
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

---

## Moved: `InformationFilter`

The inverse-covariance (information-form) linear estimator that used to live here
has been **demoted to `dtfit-experimental`** -- it is an experimental fusion
primitive (additive/associative information fusion) that has not cleared the
promotion gate, and it is no longer exported from `dtfit`. Import it from the
experimental package instead:

```python
from dtfit_experimental import InformationFilter
```

It shares no code with the nonlinear dtfit filters above; see the experimental
package for its API.
