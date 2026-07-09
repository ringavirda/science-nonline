# API: `NonlineRegressor`

A scikit-learn-compatible estimator wrapping the LSI / EAC / DSB batch methods.
It exposes the standard `fit` / `predict` / `score` API (plus
`get_params`/`set_params` from `BaseEstimator`), so it composes with
`sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`.

```python
NonlineRegressor(expr="a0 + a1*x", var="x", param_names=None, method="lsi",
                 k_star=5, alpha=0.0, filter_data=False, bounds=None,
                 active_ratio=1.0, poly_degree=None, p0=None, random_state=0,
                 robust=False, huber_c=3.0, nan_policy="raise",
                 loss="linear", window_mode="uniform")
```

## Constructor arguments

| name | applies to | default | meaning |
|---|---|---|---|
| `expr` | all | `"a0 + a1*x"` | model, e.g. `"a0 + a1*x + a2*exp(a3*x)"` — a SymPy-expression string, a `sympy.Expr`, or a plain Python **callable** `f(x, *params)` (resolved via [`resolve_model`](API-Fitting#also-exported-from-dtfitmethods)). A callable is only supported on the `"lsi"` / `"eac"` routes — `"dsb"` needs a symbolic spectrum and raises at fit time for a callable. Effectively required -- the affine default exists only so `NonlineRegressor()` is constructible with no arguments (the sklearn `clone` / meta-estimator contract); supply your own model |
| `var` | all | `"x"` | main variable name (the single input feature); a label only for a callable model |
| `param_names` | LSI+EAC | `None` | parameter names for a **callable** model, in signature order (the parameters after the leading `x`); introspected from the callable's signature when omitted, and only required when that signature cannot be introspected (a `*args` model). For a symbolic model, optional and validated against the parsed names. Stored verbatim and forwarded to the fitter |
| `method` | -- | `"lsi"` | `"lsi"`, `"eac"`, or `"dsb"` |
| `k_star` | LSI | `5` | number of spectral discretes to match |
| `alpha` | LSI | `0.0` | discrete weight decay `exp(-alpha*i)` (default now matches `fit_lsi`; was `0.2` before v0.2) |
| `filter_data` | LSI | `False` | opt-in Savitzky-Golay pre-filter (matches `fit_lsi`'s v0.2 default) |
| `bounds` | LSI+EAC | `None` | per-parameter bounds; same forms as the fitters (pair list, partial `{name: (lo, hi)}` dict, scipy tuple); enables a global search on the LSI route |
| `active_ratio` | EAC | `1.0` | leading fraction of data used for window placement (matches `fit_eac`'s v0.2 default; was `0.8`) |
| `poly_degree` | DSB | `None` | polynomial degree for the required pre-fit; `None` selects it by BIC |
| `p0` | all | `None` | initial guess: positional or `{name: value}` dict, passed through to the fitter |
| `random_state` | LSI | `0` | seed for the deterministic global / differential-evolution search used when `bounds` are given, so a bounded fit is reproducible under `GridSearchCV` / `clone`; `None` uses the global RNG |
| `robust` | LSI+EAC | `False` | robust-integral IRLS lever, forwarded to the fitter |
| `huber_c` | LSI+EAC | `3.0` | winsorization threshold for `robust=True` |
| `nan_policy` | LSI+EAC | `"raise"` | `"omit"` lets the fitter drop non-finite `(x, y)` pairs (accepted in `X`/`y` on this route) |
| `loss` | EAC | `"linear"` | least-squares loss (`"soft_l1"`/`"cauchy"`/`"huber"`) |
| `window_mode` | EAC | `"uniform"` | `"uniform"` or `"curvature"` window placement |

For DSB, the estimator runs the polynomial pre-fit for you (BIC degree, floored at
`n_params - 1`) -- you don't supply `coeffs_poly` as you would to bare
[`fit_dsb`](API-Fitting#fit_dsb).

## Fitted attributes (after `fit`)

| attribute | meaning |
|---|---|
| `coef_` | fitted coefficients (ordered by sorted parameter name) |
| `model_` | callable model at the fitted coefficients |
| `result_` | the full [`FittingResult`](API-Types) — `cov`, `stderr()`, `confidence_intervals()`, `converged`, and the v0.3 fit-quality stats [`rsquared` / `aic` / `bic`](API-Types#fit-quality-diagnostics-v03) (from the LSI / EAC routes) are all reachable from the sklearn route |
| `n_features_in_` | number of input features (always 1) |

Samples are sorted by `x` before dispatch (the integral fitters need a monotone
axis), so fit results are sample-order invariant. Fitted estimators pickle
cleanly (the lambdified `model_` is rebuilt from `result_` on unpickle), so
`joblib`-parallel cross-validation works.

## Methods

### `fit(X, y, sample_weight=None) -> self`
Fit the model. `X` is a 1-D feature vector or a single-column 2-D array (a bare
1-D vector is promoted to one column; DataFrames keep their column name).
**Exactly one feature** is supported -- multi-feature `X` raises `ValueError`.

`sample_weight` (the scikit-learn convention, v0.3) is optional per-sample weights,
translated to the integral fitters' per-sample `sigma = 1 / sqrt(sample_weight)`
and forwarded with `absolute_sigma=False` (relative weights), so a down-weighted
sample pulls the fit less without being dropped. Only the `"lsi"` / `"eac"` routes
support it; **`method="dsb"` has no per-sample weighting and raises** when a weight
is given. A weight of `0` effectively ignores its sample (a huge `sigma`); negative
or all-zero weights raise. A **callable** model likewise fits only on `"lsi"` /
`"eac"` — `method="dsb"` raises for a callable at fit time.

### `predict(X) -> ndarray`
Evaluate the fitted model. Broadcasts a constant model to `X`'s shape.

### `score(X, y) -> float`
Inherited from `RegressorMixin` -- the R^2 of the prediction.

## Examples

Plain use:

```python
import numpy as np
from dtfit import NonlineRegressor

x = np.linspace(0, 4, 200)
y = 1.4 * np.exp(0.8 * x) + np.random.default_rng(0).normal(0, 0.15, x.size)

reg = NonlineRegressor("a*exp(b*x)", "x", method="lsi").fit(x, y)
print(reg.coef_)          # [a, b]
print(reg.score(x, y))    # R^2
```

In a cross-validation / grid search:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

cross_val_score(NonlineRegressor("a*exp(b*x)", "x"), x, y, cv=4)

GridSearchCV(
    NonlineRegressor("a*exp(b*x)", "x"),
    {"method": ["lsi", "eac"], "k_star": [4, 6, 8]},
).fit(x.reshape(-1, 1), y)
```

In a pipeline (e.g. with a scaler):

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
make_pipeline(StandardScaler(), NonlineRegressor("a*exp(b*x)", "x")).fit(x.reshape(-1, 1), y)
```

## Notes

- `get_params` / `set_params` expose every constructor argument, so all of them
  are tunable by `GridSearchCV` (`method`, `k_star`, `active_ratio`, ...).
- For uncertainty (`stderr`, confidence intervals, prediction bands), use
  `reg.result_` — the full [`FittingResult`](API-Types) is stored on fit as of
  v0.2 (previously only `coef_`/`model_` survived).
