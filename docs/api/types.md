# API: `FittingResult`

The self-describing result returned by every batch fitter. It carries the fitted
parameters, their uncertainty, the model expression, and a callable — enough to
predict, quantify uncertainty, serialize, and redeploy.

```python
FittingResult(coeffs, cov=None, expr=None, var=None, names=(), model=None)
```

You rarely construct one yourself; the fitters return it. (`scale` fitters return
the lighter [`BatchFittingResult`](scaling.md#batchfittingresult), which has the
same `model`/`predict` but no uncertainty methods.)

## Attributes

| attribute | type | meaning |
|---|---|---|
| `coeffs` | ndarray | fitted coefficients, ordered by sorted parameter name |
| `cov` | ndarray \| None | parameter covariance (`n×n`) when the method produced one from an overdetermined system; else `None`. Diagonal square-roots are the standard errors |
| `expr` | str \| None | the model expression (enables serialization & error bands) |
| `var` | str \| None | the main variable name |
| `names` | tuple[str] | parameter names aligned with `coeffs` |
| `model` | callable | the fitted model `f(x) → y` (lambdified lazily from `expr`+`coeffs` if not precomputed) |

## Methods & properties

### `params -> dict[str, float]`
Fitted parameters as a `{name: value}` mapping (the most common accessor).

```python
res.params          # {'a': 3.001, 'w': 1.498}
```

### `predict(x, *, return_std=False)`
Evaluate the fitted model at `x`. With `return_std=True` also returns the 1-sigma
prediction standard deviation propagated from the parameter covariance (delta
method) — needs `cov` and `expr`.

```python
y_hat = res.predict(x)
y_hat, sigma = res.predict(x, return_std=True)   # with uncertainty band
```

### `stderr() -> dict[str, float]`
Per-parameter standard errors (√ of the covariance diagonal). Raises if `cov` is
`None`.

### `confidence_intervals(level=0.95) -> dict[str, (lo, hi)]`
Per-parameter confidence intervals (normal approximation) at the given level.

### `summary() -> str`
A short human-readable text summary — parameters `±` standard errors when a
covariance is available.

```python
print(res.summary())
# FittingResult: a*atan(w*x)
#   a = 3.00123 +/- 0.0142
#   w = 1.49831 +/- 0.0119
```

### `to_dict() -> dict` / `from_dict(d) -> FittingResult`
JSON-friendly round-trip for storage/deployment — captures `expr`, `var`,
`names`, `coeffs`, `cov`. Requires `expr`/`var` (a result holding only a
precomputed callable cannot be serialized).

```python
import json
blob = json.dumps(res.to_dict())
res2 = FittingResult.from_dict(json.loads(blob))   # rebuilt, ready to predict
```

## Notes

- The `model` callable broadcasts scalars: a constant model still returns an array
  matching `x`.
- `cov` is `None` for exactly- or under-determined fits (e.g. EDA with
  `n_windows == n_params`); uncertainty methods then raise with a clear message.
- Parameter **order is by sorted name**, consistently across the whole library —
  so `coeffs`, `names`, and `params` always agree.
