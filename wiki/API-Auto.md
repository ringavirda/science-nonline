# API: one-call entry points

Two high-level "just fit it" functions distilled from the domain studies, whose
main finding was that **picking the structurally-correct model/estimator variant
is the biggest lever** -- not the solver. These compose only validated, stable
pieces behind a single call.

- [`auto_estimate`](#auto_estimate) -- recover physical parameters, routing by signal shape
- [`auto_forecast`](#auto_forecast) -- structured fit-then-extrapolate forecast, with safety guards

---

<a name="auto_estimate"></a>
## `auto_estimate`

```python
auto_estimate(x, y, expr, var, *,
              shape="auto", freq_param=None,
              p0=None, bounds=None) -> FittingResult
```

Recover the parameters of `expr` by routing to the estimator that fits the
signal's *shape* (the parameter-estimation study's merged selector).

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `x`, `y` | array | -- | observed samples |
| `expr`, `var` | str | -- | model expression and main variable |
| `shape` | str | `"auto"` | which variant to use (see table below) |
| `freq_param` | str \| None | `None` | angular-frequency parameter name; forwarded to the LSI oscillatory recipe and **implies an oscillatory shape** |
| `p0` | array \| None | `None` | initial guess |
| `bounds` | list[(lo, hi)] \| None | `None` | per-parameter bounds |

**`shape` routing**

| `shape` | routes to |
|---|---|
| `"auto"` | detect a cycle (FFT power share > 0.10) -> oscillatory; else `"bulk"`. `freq_param` given => oscillatory |
| `"oscillatory"` | [`fit_lsi`](API-Fitting#fit_lsi) with the oscillatory recipe |
| `"transient"` / `"peak"` | [`fit_eac_adaptive`](API-Fitting#fit_eac_adaptive) (curvature windows) |
| `"robust"` | [`fit_eac`](API-Fitting#fit_eac) with `loss="soft_l1"` |
| `"bulk"` | fit both LSI and EAC, keep the lower in-sample RMSE |

**Returns** the `FittingResult` from the selected estimator.

```python
from dtfit import auto_estimate
res = auto_estimate(x, y, "a*atan(w*x)", "x", shape="transient")
res = auto_estimate(x, y, "A*sin(w*x + p)", "x", freq_param="w")   # oscillatory
res = auto_estimate(x, y, "a*exp(b*x)", "x")                       # auto > bulk
```

Honest ceiling: on clean bulk shapes `auto_estimate` *matches* but does not beat a
well-initialized NLLS.

---

<a name="auto_forecast"></a>
## `auto_forecast`

```python
auto_forecast(x, y, horizon, *,
              model="auto", period=None,
              seasonal=True, season_strength=0.05) -> np.ndarray
```

A structured fit-then-extrapolate forecaster: route the model class, fit it, and
extrapolate `horizon` steps past `x` on its uniform grid. Two safety guards keep
it honest.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `x`, `y` | array | -- | observed series (`x` (near-)uniformly sampled) |
| `horizon` | int | -- | number of future steps to forecast |
| `model` | str | `"auto"` | `"auto"` (route by structure) or one of `"logistic"`, `"linear"`, `"poly"`, `"linear_seasonal"`, `"random_walk"` |
| `period` | float \| None | `None` | known seasonal period (in samples) for the seasonal fit |
| `seasonal` | bool | `True` | consider a seasonal model under `"auto"` |
| `season_strength` | float | `0.05` | minimum detected cycle strength to pick a seasonal model |

**Routing (`model="auto"`)**: saturating positive growth -> `logistic`; a detected
cycle -> `linear_seasonal` (linear + sine); otherwise a quadratic level (`poly`).

**The guards**

- **No-structure guard** -- if the structured model can't get near naive
  persistence on a held-out tail of the *training* data (the near-random-walk
  signature), it falls back to persisting the last value.
- **Divergence guard** -- a runaway quadratic (extrapolating far outside the data
  range) is dropped to a linear fit.

**Returns** the length-`horizon` forecast (values at the extrapolated x grid).

```python
from dtfit import auto_forecast
future = auto_forecast(x, y, horizon=30)               # auto-routed + guarded
future = auto_forecast(x, y, horizon=30, model="logistic")
```

Honest ceiling: near-random-walk series fall back to persistence -- by design, not
defect (see [../guides/README.md](Guides) Sec.6).
