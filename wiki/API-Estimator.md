# API: `NonlineRegressor`

A scikit-learn-compatible estimator wrapping the LSI / EAC / DSB batch methods.
It exposes the standard `fit` / `predict` / `score` API (plus
`get_params`/`set_params` from `BaseEstimator`), so it composes with
`sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`.

```python
NonlineRegressor(expr, var="x", method="lsi", k_star=5, alpha=0.2,
                 filter_data=True, bounds=None, active_ratio=0.8,
                 poly_degree=None, p0=None)
```

## Constructor arguments

| name | applies to | default | meaning |
|---|---|---|---|
| `expr` | all | -- | model expression, e.g. `"a0 + a1*x + a2*exp(a3*x)"` |
| `var` | all | `"x"` | main variable name (the single input feature) |
| `method` | -- | `"lsi"` | `"lsi"`, `"eac"`, or `"dsb"` |
| `k_star` | LSI | `5` | number of spectral discretes to match |
| `alpha` | LSI | `0.2` | discrete weight decay `exp(-alpha*i)` |
| `filter_data` | LSI | `True` | Savitzky-Golay pre-filter |
| `bounds` | LSI | `None` | per-parameter `(min, max)` bounds; enables a global search |
| `active_ratio` | EAC | `0.8` | leading fraction of data used for window placement |
| `poly_degree` | DSB | `None` | polynomial degree for the required pre-fit; `None` selects it by BIC |
| `p0` | all | `None` | initial guess for the parameters |

For DSB, the estimator runs the polynomial pre-fit for you (BIC degree, floored at
`n_params - 1`) -- you don't supply `coeffs_poly` as you would to bare
[`fit_dsb`](API-Fitting#fit_dsb).

## Fitted attributes (after `fit`)

| attribute | meaning |
|---|---|
| `coef_` | fitted coefficients (ordered by sorted parameter name) |
| `model_` | callable model at the fitted coefficients |
| `n_features_in_` | number of input features (always 1) |

## Methods

### `fit(X, y) -> self`
Fit the model. `X` is a 1-D feature vector or a single-column 2-D array (a bare
1-D vector is promoted to one column; DataFrames keep their column name).
**Exactly one feature** is supported -- multi-feature `X` raises `ValueError`.

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
- For uncertainty (`stderr`, confidence intervals, prediction bands), call the
  underlying [`fit_lsi`](API-Fitting#fit_lsi) / [`fit_eac`](API-Fitting#fit_eac)
  directly and use the returned [`FittingResult`](API-Types); the sklearn estimator
  surfaces only `coef_`/`model_` to stay API-compatible.
