# API: stochastic-series characterization

Fit dtfit's deterministic fitters to the **deterministic functionals** of a
*random* series -- its autocovariance, spectrum, aggregated variance and
trend/cycle -- to characterize it, forecast it, generate it, and track it online.
A martingale path has no `y = f(t)` to fit; its functionals do, and their forms
(damped exponentials/cosines, power laws) are exactly the shapes dtfit excels at.

Concept and validation: [methods/stochastic.md](Methods-Stochastic) and the
[stochastic-series domain report](Domain-Stochastic-Series). This page is the
exact call reference.

- [`fit_stochastic`](#fit_stochastic) / [`StochasticModel`](#model) -- the batch
  solution (characterize + forecast + generate)
- [`Stochastic`](#stochastic) -- the same in the model `.fit()` convention
- [`StochasticFilter`](#filter) -- the streaming online twin
- [estimators](#estimators) -- the individual functional routes
- [`FORECASTERS`](#forecasters) -- the forecaster names

```python
# the whole surface
from dtfit import fit_stochastic, StochasticModel, StochasticFilter, Stochastic
from dtfit import stochastic            # submodule namespace
from dtfit.stochastic import (sample_acf, hurst_aggvar, hurst_spectral,
                              ar1_reversion, garch_persistence, cycle_period,
                              decompose_trend_cycle, FORECASTERS)
from dtfit.models import Stochastic     # also here (catalog convention)
```

---

<a name="fit_stochastic"></a>
## `fit_stochastic`

```python
fit_stochastic(y, t=None, *, period=None, max_harmonics=4, forecaster="auto",
               trend_t=3.0, cycle_strength=0.08, min_cycles=2.5,
               lm_hurst=0.68, mr_phi=0.15, vol_persist=0.60) -> StochasticModel
```

Characterize an arbitrary series across every route at once and return a single
coherent [`StochasticModel`](#model). The routes are composed in the order the
second-order theory dictates, each behind a **significance gate**:

1. **unit-root gate** -- a vendored, statsmodels-free augmented Dickey-Fuller test
   (reproduces `statsmodels.adfuller(regression="ct", autolag="AIC")` to machine
   precision in pure NumPy). An I(1) level (random walk) is *differenced* and
   reported as such, not given a spurious trend / cycle / long memory.
2. **deterministic mean** -- an LSI trend (kept only if `|t| > trend_t` *and* it
   explains real variance) and a multi-harmonic Fourier seasonal/cycle (kept only
   on a genuine repeating spectral peak: `> cycle_strength` of the power, repeating
   `>= min_cycles` times; period detected or supplied via `period=`).
3. **whiten** with an AR(1), then test **long memory on the innovations**
   (`H > lm_hurst`) so a near-unit-root AR(1) is not mislabelled.
4. **mean reversion** (`mr_phi < phi < 0.99`, lag-1 ACF significant) and
   **volatility clustering** (persistence `> vol_persist` on the whitened residual).

A series with no gate open is reported as `regime="white noise / random walk"`.

**Forecasting is RMSE-optimal backtest model selection** -- a regime-informed
candidate set (random walk, drift, mean reversion, a curvature-aware dtfit LSI
trend, two multi-harmonic seasonal continuations) is rolling-origin backtested and
the best kept; the choice is in `model.forecaster_name`.

| argument | default | meaning |
|---|---|---|
| `y` | -- | the series (1-D) |
| `t` | `None` | time index; defaults to `0..n-1` (uniform spacing) |
| `period` | `None` | seasonal period to use, else detected from the spectrum |
| `max_harmonics` | `4` | cap on the Fourier harmonics (count chosen by BIC) |
| `forecaster` | `"auto"` | `"auto"` backtest-selects; a name from [`FORECASTERS`](#forecasters) forces one; a callable `(train, h) -> array` is used directly; a list of names/callables is a custom candidate set |
| `trend_t`, `cycle_strength`, `min_cycles`, `lm_hurst`, `mr_phi`, `vol_persist` | -- | the detection gates (above). Tuned defaults; override per series |

```python
import numpy as np
from dtfit import fit_stochastic

t = np.arange(600.0)
y = 0.02 * t + 3 * np.sin(2 * np.pi * t / 50) + np.random.default_rng(0).normal(0, 1, 600)
m = fit_stochastic(y)
print(m.regime)              # 'trend+seasonal'
print(m.summary())           # the detected components + parameters
yhat = m.forecast(40)        # backtest-selected forecast
```

---

<a name="model"></a>
## `StochasticModel`

The unified second-order characterization returned by `fit_stochastic`. A
dataclass of detected components + recovered parameters, plus a forecaster and a
generator.

**Fields** (each detection is behind a gate, so white noise yields none):

| field | meaning |
|---|---|
| `regime` | the primary regime label (`"trend+seasonal"`, `"mean-reverting"`, `"long-memory"`, `"random walk + drift"`, `"white noise / random walk"`, ...) |
| `components` | tuple of detected components (`("trend", "seasonal")`, `("none",)`, ...) |
| `forecaster_name` | the backtest-selected forecaster |
| `trend_slope`, `has_trend` | linear trend |
| `cycle_period`, `cycle_amp`, `has_cycle`, `n_harmonics`, `seasonal` | cycle / seasonal |
| `hurst`, `has_long_memory` | long memory (`d = hurst - 0.5`) |
| `ar1_phi`, `has_mean_reversion` | AR(1) mean reversion |
| `vol_persistence`, `has_vol_clustering` | GARCH-type volatility persistence |
| `sigma`, `sigma_walk` | one-step innovation std / random-walk (first-difference) scale |
| `n`, `level` | length / mean |

**Methods**

- `forecast(h, *, return_conf_int=False, alpha=0.05)` -- forecast `h` steps with
  the selected forecaster. With `return_conf_int` returns `(point, lower, upper)`
  whose band growth matches the forecaster (bounded for mean reversion, `~h^(2H)`
  for long memory, `~sqrt(h)` for a random walk / drift).
- `simulate(n=None, *, seed=None, rng=None) -> ndarray` -- draw a **fresh
  realization** from the detected components: the deterministic mean plus a
  residual matched to the regime (AR(1) / ARFIMA long memory / GARCH / integrated
  walk / white noise). Re-fitting a simulated path recovers the same regime -- the
  honest test that the model is a faithful generator.
- `fingerprint() -> dict` -- the detected structure as a flat `{name: value}` map
  (for tables).
- `summary() -> str` -- a human-readable multi-line summary.

```python
pt, lo, hi = m.forecast(40, return_conf_int=True)   # forecast + 95% band
sim = m.simulate(600, seed=1)                        # a new series, same structure
m.fingerprint()                                      # {'regime': ..., 'trend slope': ..., ...}
```

---

<a name="stochastic"></a>
## `Stochastic`

```python
Stochastic(*, period=None, max_harmonics=4, forecaster="auto", **gates)
Stochastic.fit(x, y=None) -> StochasticModel
```

The stochastic-series model in the catalog `.fit()` convention -- the same
ergonomics as the deterministic families (`dtfit.models.logistic().fit(x, y)`),
but it characterizes a random *series* instead of fitting a `y = f(x)` curve. A
thin wrapper over `fit_stochastic`; constructor arguments mirror it. `**gates`
forwards the detection-gate overrides.

`fit` accepts `fit(series)` (uniform unit time) or `fit(t, series)` (explicit time
index). It returns the fitted [`StochasticModel`](#model), also stored on
`.model_`. It is **not** a `Model` subclass (it has no sympy expression) and is
**not** in the AIC `CATALOG` -- it just shares the calling convention. Available as
`dtfit.Stochastic` and `dtfit.models.Stochastic`.

```python
from dtfit import Stochastic
m = Stochastic().fit(series)                 # -> a fitted StochasticModel
m = Stochastic(period=12, forecaster="trend+seasonal").fit(t, series)
print(m.regime, m.forecaster_name)
```

---

<a name="filter"></a>
## `StochasticFilter`

```python
StochasticFilter(nlags=24, halflife=150.0, warmup=80, settle=500, z_thresh=4.0)
```

The **per-input streaming twin** of `fit_stochastic`'s second-order stage -- the
stochastic counterpart of [`EACFilter` / `LSIFilter`](API-Streaming). It maintains
EWMA autocovariances of the level and of `|level - mean|` in `O(K)` per sample (the
running ACF), and reads the parameters in **closed form** using dtfit's principles
in streaming form: the **EAC equal-areas criterion** for the AR(1) persistence and
the volatility persistence, and the **AR(2) characteristic roots** for the cycle --
no per-sample optimization, no batch fit. A two-timescale fused statistic flags
structural breaks (a persistence jump, a volatility switch), once per change, at a
low false-alarm rate. Flat memory, bounded per-sample cost.

> **Scope.** This is the online twin of the *second-order* stage only. Long memory
> (the spectral Hurst) and the unit-root gate are inherently batch (a periodogram /
> a regression over the whole record), so they are not tracked here.

| argument | default | meaning |
|---|---|---|
| `nlags` | `24` | autocovariance lags maintained (the memory footprint) |
| `halflife` | `150.0` | EWMA half-life (samples) -- how fast the characterization adapts |
| `warmup` | `80` | samples before the detector is active |
| `settle` | `500` | samples after `warmup` to calibrate the detector's in-control baseline before it may flag |
| `z_thresh` | `4.0` | fused-statistic threshold (sigmas) for a flag |

**Methods & attributes**

- `update(x) -> self` -- ingest one sample. `partial_fit(xs)` ingests a batch.
- `params_ -> dict` -- current `{level, sigma, ar1_phi, cycle_period, vol_persistence, n}`.
- `snapshot() -> dict` -- `params_` plus a coarse online `regime` label.
- `predict(h) -> ndarray` -- forecast `h` steps by AR(1) mean reversion at the snapshot.
- `n_flags_` -- structural breaks detected so far; `flag_times_` -- a bounded ring
  of recent break sample indices; `last_flag_` -- the most recent break index.

```python
from dtfit import StochasticFilter
f = StochasticFilter(halflife=200)
for x in stream:
    f.update(x)
    if f.last_flag_ == f.params_["n"]:
        print("regime change at sample", f.last_flag_, "->", f.snapshot()["regime"])
```

---

<a name="estimators"></a>
## Estimators -- the individual functional routes

Each recovers a stochastic-model parameter by feeding a deterministic functional of
the series to [`fit_lsi`](API-Fitting#fit_lsi) / [`fit_eac`](API-Fitting#fit_eac).
`method="lsi"` (default) / `"eac"` pick the engine; `"ols"` / `"acf1"` are plain
baselines.

| function | recovers | functional fit |
|---|---|---|
| `hurst_spectral(x, *, n_freq=None, method="lsi")` | Hurst `H`, `d` | LSI slope of the low-frequency log-periodogram (GPH) |
| `hurst_aggvar(x, *, n_scales=14, min_block=2, method="lsi")` | Hurst `H`, `d` | power-law fit of the aggregated-variance curve |
| `ar1_reversion(x, *, nlags=None, method="lsi")` | AR(1) `phi`, `tau`, `halflife` | exponential fit to the ACF |
| `garch_persistence(returns, *, nlags=None, method="lsi", use="square")` | persistence `alpha+beta`, `tau` | exponential fit to the ACF of `\|returns\|` (`use="abs"`) or squared returns |
| `cycle_period(x, *, nlags=None)` | cycle `period`, `w`, `damping` | damped-cosine fit to the ACF (oscillatory recipe) |
| `decompose_trend_cycle(t, y, *, trend_deg=1, with_cycle=True)` | `slope`, `period`, `amp`, fitted `trend`/`cycle`/`residual`, a `forecast(h, dt)` closure | LSI trend + LSI cycle, leaving a stochastic residual |
| `sample_acf(x, nlags) -> ndarray` | the biased sample autocorrelation `rho[0..nlags]` (the shared functional, FFT-based) |

```python
from dtfit.stochastic import hurst_spectral, ar1_reversion
hurst_spectral(returns)["H"]        # long-memory exponent
ar1_reversion(spread)["phi"]        # mean-reversion speed
```

---

<a name="forecasters"></a>
## `FORECASTERS`

The built-in forecaster names accepted by `fit_stochastic(..., forecaster=...)`:

```python
FORECASTERS == ("random walk", "drift", "mean-reversion", "trend",
                "seasonal", "trend+seasonal")
```

Force one with a string, plug your own with a callable `(train, h) -> array`, or
pass a list of candidates to backtest-select among. `"seasonal"` /
`"trend+seasonal"` require a known `period` (detected or via `period=`).
