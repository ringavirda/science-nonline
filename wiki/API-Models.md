# API: the model framework

Pick a *shape*, not a formula. `dtfit.models` is a catalog of named model
families that read their own starting guesses off the data, compose with `+`, and
can be ranked automatically. Background: the domain studies found that **choosing
the structurally-correct model is the whole game** -- this package makes that
choice ergonomic.

- [`Model`](#model) -- a model family (expression + self-seeder + fit)
- [`suggest_models`](#suggest_models) / [`Suggestion`](#suggestion) -- fit & rank candidates
- [the catalog](#catalog) -- the built-in families
- [`Stochastic`](#stochastic) -- the stochastic-series model in the same `.fit()` convention

```python
from dtfit import models
fit = models.logistic().fit(x, y)                    # self-seeded
fit = (models.linear() + models.sine()).fit(x, y)    # compose trend + cycle
for s in suggest_models(x, y)[:3]:
    print(s.name, s.r2, s.aic)

m = models.Stochastic().fit(series)                  # a random series -> StochasticModel
```

---

<a name="model"></a>
## `Model`

```python
Model(expr, var="x", *, name="", shape="bulk", category="general",
      freq_param=None, seeder=None)
```

A model family: the expression plus a routing `shape`, an optional
angular-frequency parameter, and an optional data-driven `seeder` that produces
`{name: (p0, lo, hi)}`. You normally get `Model` instances from the
[catalog](#catalog) rather than constructing them.

**Key attributes:** `expr`, `var`, `name`, `shape` (`"bulk"` / `"oscillatory"` /
`"transient"` / `"peak"` / `"composite"` -- decides the estimator under
`method="auto"`), `category`, `freq_param`, `params` (the parameter-name tuple).

### `fit(x, y, *, method="auto", p0=None, bounds=None) -> FittingResult`
Fit this family to the data, **self-seeding** `p0`/`bounds` from the model's
seeder unless you override them.

| `method` | engine | bounds forwarded? |
|---|---|---|
| `"auto"` (default) | routes by `shape` through [`auto_estimate`](API-Auto#auto_estimate) | yes |
| `"lsi"` | [`fit_lsi`](API-Fitting#fit_lsi) (passing `freq_param`) | yes |
| `"eac"` | [`fit_eac`](API-Fitting#fit_eac) | yes (as `(lower, upper)` tuples) |
| `"adaptive"` | [`fit_eac`](API-Fitting#fit_eac) with `window_mode="curvature"` | **no -- bounds are dropped** |

> **Bounds on the `"adaptive"` path.** Unlike `"eac"` (and every other method),
> `method="adaptive"` forwards only `p0` to [`fit_eac`](API-Fitting#fit_eac) -- the
> seeded (or explicitly passed) `bounds` are **silently ignored**, so the
> curvature-window fit is unconstrained. If you need the fit constrained, use
> `method="eac"` (uniform windows, honours bounds) or apply the bounds another way.

### `seed(x, y) -> dict`
The data-driven `{name: (p0, lo, hi)}` seed map (empty if the family has no
seeder).

### `__add__` (composition with `+`)
`model_a + model_b` builds a new additive model (e.g. `trend + seasonal`).
Colliding parameter names in the right operand are renamed, and the seeders
compose so the combined model is **still self-seeding**: the second component is
seeded on the detrended residual of the first. Both operands must share the same
`var`.

```python
trend_plus_cycle = models.linear() + models.sine()
fit = trend_plus_cycle.fit(x, y)
```

---

<a name="suggest_models"></a>
## `suggest_models`

```python
suggest_models(x, y, candidates=None, *, method="auto", top=None,
               include=None, exclude=None) -> list[Suggestion]
```

Fit candidate families to `(x, y)` and rank them **best-first by AIC**.

| arg | default | meaning |
|---|---|---|
| `candidates` | `None` | models to try; default is a **shape-based shortlist** of the catalog (oscillatory data skips peak/monotone families, etc.; ambiguous data falls back to the whole catalog so the true family is never dropped) |
| `method` | `"auto"` | fitting method passed to each model |
| `top` | `None` | if given, return only the best `top` |
| `include` | `None` | keep only candidates whose **name** (e.g. `"logistic"`) or **category** (e.g. `"decay"`, `"oscillatory"`) is in this list -- restrict the search to families you believe plausible |
| `exclude` | `None` | drop candidates whose name or category is in this list -- prune families you know don't apply (e.g. `exclude=["oscillatory"]` on a monotone series) without post-filtering. Applied **after** `include` |

Candidates whose fit fails (or yields non-finite R^2) are skipped. Each result is a
[`Suggestion`](#suggestion).

```python
for s in suggest_models(x, y, top=3):
    print(f"{s.name:20s} r2={s.r2:.4f} aic={s.aic:.1f}")

# you know the data is not oscillatory -> prune those families up front
suggest_models(x, y, exclude=["oscillatory"])
# or restrict to the exponential growth/decay families only
suggest_models(x, y, include=["growth", "decay"])
```

<a name="suggestion"></a>
### `Suggestion`

One ranked candidate. Attributes: `name`, `model` ([`Model`](#model)), `result`
([`FittingResult`](API-Types)), `report` (the full [`fit_report`](API-Diagnostics)
dict). Convenience properties: `aic`, `bic`, `r2`.

---

<a name="catalog"></a>
## The catalog

`from dtfit import models` exposes one constructor per family (each returns a
self-seeding [`Model`](#model)), grouped by `category`. `models.CATALOG` is the
full registry and `models.all_models()` returns one instance of each.

| category | families (constructors) |
|---|---|
| **trend** | `linear`, `quadratic`, `cubic`, `power_law`, `logarithmic`, `sqrt_law` |
| **growth** | `exponential`, `exp_growth_offset` |
| **decay / relaxation** | `exp_decay`, `exp_decay_offset`, `first_order`, `biexponential`, `stretched_exponential` |
| **sigmoid** | `logistic`, `gompertz`, `weibull_cdf`, `tanh_step` |
| **saturating / rational** | `michaelis_menten`, `hill` |
| **peak** | `gaussian`, `lorentzian`, `double_gaussian` |
| **oscillatory** | `sine`, `damped_oscillation`, `fourier_series` |

```python
from dtfit import models

models.logistic()            # L/(1 + exp(-k*(x - x0))), shape="bulk"/sigmoid, self-seeds L,k,x0
models.gaussian()            # a peak family; curvature-routed
models.damped_oscillation()  # oscillatory, carries a freq_param

[m.name for m in models.all_models()]   # every family
```

Each constructor accepts no required arguments and reads its parameter ranges off
the data at `fit` time, so the typical use is a one-liner:
`models.<family>().fit(x, y)`.

---

<a name="stochastic"></a>
## `Stochastic`

```python
Stochastic(*, period=None, max_harmonics=4, forecaster="auto", **gates)
Stochastic.fit(x, y=None) -> StochasticModel
```

The stochastic-series model in the catalog `.fit()` convention. Unlike the families
above it does **not** fit a `y = f(x)` curve -- it characterizes a *random series*
across the stochastic routes (trend / cycle / long memory / mean reversion /
volatility) behind significance gates -- but it is driven the same way. `fit`
accepts `fit(series)` or `fit(t, series)` and returns a `StochasticModel` (also on
`.model_`) that forecasts and generates. It is not a `Model` subclass (no sympy
expression) and is not in the AIC catalog -- it just shares the convention; full
reference in [stochastic.md](API-Stochastic). Available as `dtfit.Stochastic` and
`dtfit.models.Stochastic`.

```python
from dtfit import Stochastic
m = Stochastic().fit(series)        # -> a fitted StochasticModel
print(m.regime); m.forecast(12); m.simulate(200)
```

---

<a name="accuracy"></a>
## Accuracy and known limits

Every family above is exercised by an **accuracy corpus**
([`packages/dtfit/tests/accuracy/`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/tests/accuracy)): each
is fit on a known-ground-truth signal across a noise sweep, through the realistic
self-seeded `models.<family>().fit(x, y)` path, and required to stay competitive
with the `scipy.curve_fit` gold standard. A checked-in golden baseline guards
against silent accuracy drift. So the one-liner is validated for **all** families
-- but two honest limits are worth knowing.

### Parameter recovery vs. curve quality

For most families the parameters are **identifiable**: the fit recovers the true
values (relative error a few percent at 5 % noise). Three families are only weakly
identifiable -- many parameter sets reproduce the same curve, so a noisy fit lands
on *a* curve that matches, not on the true parameters:

| family | why parameters are weakly identifiable | judge by |
|---|---|---|
| `biexponential` | a sum of exponentials trades rate against amplitude (a classic ill-conditioned inverse problem) | curve quality (R^2); recovery degrades from ~10 % noise up |
| `double_gaussian` | overlapping peaks can swap labels / trade width for amplitude | curve quality (R^2) |
| `fourier_series` | individual harmonic amplitudes/phases are weakly constrained | curve quality (R^2) |

For these, trust the **fitted curve** (`R^2`, prediction) rather than the
individual coefficients, and prefer extra data / lower noise / bounds if you need
the parameters themselves.

### Cycles need the oscillatory recipe

Recovering a frequency requires the **oscillatory recipe** (smoothing off, order
raised to resolve the cycle, FFT-seeded frequency). The self-seeding path applies
it automatically -- `models.sine().fit(x, y)`, `(linear() + sine()).fit(...)`, or
[`auto_estimate`](API-Auto#auto_estimate) all route oscillatory families through
it. The **bare** [`NonlineRegressor("...sin...", method="lsi")`](API-Estimator) does
*not* (its default smoothing + low order erase the cycle); pass `freq_param=` or
use the model/`auto_estimate` path instead. See the
[LSI oscillatory recipe](Methods-LSI#the-oscillatory-recipe).

### `suggest_models` coverage

The default shortlist spans the whole catalog (the shape detector is permissive:
a strongly monotone series keeps the trend/growth/decay/sigmoid/saturating
families, a peak keeps the peak families, and ambiguous data falls back to
everything, so the true family is never silently dropped). `fourier_series` is the
one exception -- a parametric factory offered separately, not in the default sweep,
so a periodic signal surfaces its fundamental `sine`; add `fourier_series()`
explicitly to the `candidates` if you need the harmonic model.
