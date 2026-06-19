# API: the model framework

Pick a *shape*, not a formula. `dtfit.models` is a catalog of named model
families that read their own starting guesses off the data, compose with `+`, and
can be ranked automatically. Background: the domain studies found that **choosing
the structurally-correct model is the whole game** — this package makes that
choice ergonomic.

- [`Model`](#model) — a model family (expression + self-seeder + fit)
- [`suggest_models`](#suggest_models) / [`Suggestion`](#suggestion) — fit & rank candidates
- [the catalog](#catalog) — the built-in families

```python
from dtfit import models
fit = models.logistic().fit(x, y)                    # self-seeded
fit = (models.linear() + models.sine()).fit(x, y)    # compose trend + cycle
for s in suggest_models(x, y)[:3]:
    print(s.name, s.r2, s.aic)
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
`"transient"` / `"peak"` / `"composite"` — decides the estimator under
`method="auto"`), `category`, `freq_param`, `params` (the parameter-name tuple).

### `fit(x, y, *, method="auto", p0=None, bounds=None) -> FittingResult`
Fit this family to the data, **self-seeding** `p0`/`bounds` from the model's
seeder unless you override them.

| `method` | engine |
|---|---|
| `"auto"` (default) | routes by `shape` through [`auto_estimate`](auto.md#auto_estimate) |
| `"lsi"` | [`fit_lsi`](fitting.md#fit_lsi) (passing `freq_param`) |
| `"eda"` | [`fit_eda`](fitting.md#fit_eda) |
| `"adaptive"` | [`fit_eda_adaptive`](fitting.md#fit_eda_adaptive) |

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
suggest_models(x, y, candidates=None, *, method="auto", top=None) -> list[Suggestion]
```

Fit candidate families to `(x, y)` and rank them **best-first by AIC**.

| arg | default | meaning |
|---|---|---|
| `candidates` | `None` | models to try; default is a **shape-based shortlist** of the catalog (oscillatory data skips peak/monotone families, etc.; ambiguous data falls back to the whole catalog so the true family is never dropped) |
| `method` | `"auto"` | fitting method passed to each model |
| `top` | `None` | if given, return only the best `top` |

Candidates whose fit fails (or yields non-finite R²) are skipped. Each result is a
[`Suggestion`](#suggestion).

```python
for s in suggest_models(x, y, top=3):
    print(f"{s.name:20s} r2={s.r2:.4f} aic={s.aic:.1f}")
```

<a name="suggestion"></a>
### `Suggestion`

One ranked candidate. Attributes: `name`, `model` ([`Model`](#model)), `result`
([`FittingResult`](types.md)), `report` (the full [`fit_report`](diagnostics.md)
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
