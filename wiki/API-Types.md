# API: `FittingResult`

## What it is

`FittingResult` is the object every batch fitter hands back. It is deliberately
**self-describing**: instead of returning a bare array of numbers, it bundles the
fitted **coefficients** together with everything needed to *interpret* them -- the
parameter **names**, the model **expression** and **variable**, an estimate of the
parameters' **uncertainty** (a covariance matrix), and a ready-to-call **model
function**. That bundle is enough to do four things without any extra context:

1. **Read the parameters by name** -- `res.params` -> `{'a': 3.0, 'w': 1.5}`.
2. **Predict** -- `res.predict(x)`, optionally with an error band.
3. **Quantify uncertainty** -- standard errors and confidence intervals.
4. **Serialize & redeploy** -- round-trip to a plain dict (JSON) and back.

The design goal is that a fit you computed today can be saved, shipped, reloaded
elsewhere, and still know what it represents -- no separate metadata to carry
around.

```python
FittingResult(coeffs, cov=None, expr=None, var=None, names=(), model=None)
```

You rarely construct one yourself; the fitters return it. `FittingResult` is
picklable, so it survives a process-pool round trip; the `scale` fitters return it
too. [`BatchFittingResult`](API-Scaling#batchfittingresult) is now a **deprecated
back-compat alias** of `FittingResult`, kept for compatibility.

## Attributes

| attribute | type | meaning |
|---|---|---|
| `coeffs` | ndarray | fitted coefficients, ordered by sorted parameter name |
| `cov` | ndarray \| None | parameter covariance (`nxn`) when the method produced one from an overdetermined system; else `None`. Diagonal square-roots are the standard errors |
| `expr` | str \| None | the model expression (enables serialization & error bands) |
| `var` | str \| None | the main variable name |
| `names` | tuple[str] | parameter names aligned with `coeffs` |
| `model` | callable | the fitted model `f(x) > y` (lambdified lazily from `expr`+`coeffs` if not precomputed) |

## Methods & properties

### `params -> dict[str, float]`
Fitted parameters as a `{name: value}` mapping (the most common accessor).

```python
res.params          # {'a': 3.001, 'w': 1.498}
```

### `predict(x, *, return_std=False)`
Evaluate the fitted model at `x`. With `return_std=True` it also returns a 1-sigma
prediction band.

**How the band is computed.** The uncertainty in the *parameters* (the covariance
`cov`) propagates into uncertainty in the *prediction* via the **delta method**:
the model is locally linearized at each `x` (numerically, by perturbing each
parameter), and the parameter covariance is pushed through that linearization to
give a per-point variance. So the band is wide where the model is sensitive to the
uncertain parameters and narrow where it isn't. This needs both `cov` and `expr`.

```python
y_hat = res.predict(x)
y_hat, sigma = res.predict(x, return_std=True)   # 1sigma band from parameter covariance
```

### `stderr() -> dict[str, float]`
Per-parameter standard errors (sqrt of the covariance diagonal). Raises if `cov` is
`None`.

### `confidence_intervals(level=0.95) -> dict[str, (lo, hi)]`
Per-parameter confidence intervals (normal approximation) at the given level.

### `summary() -> str`
A short human-readable text summary -- parameters `+/-` standard errors when a
covariance is available.

```python
print(res.summary())
# FittingResult: a*atan(w*x)
#   a = 3.00123 +/- 0.0142
#   w = 1.49831 +/- 0.0119
```

### `to_dict() -> dict` / `from_dict(d) -> FittingResult`
JSON-friendly round-trip for storage/deployment. `to_dict()` captures everything
needed to rebuild the model -- `expr`, `var`, `names`, `coeffs`, `cov` -- as plain
Python types, so it serializes cleanly to JSON; `from_dict()` reconstructs a fully
functional `FittingResult` (the callable model is re-lambdified lazily on first
use). This is the recommended way to **persist a fit** or **ship it to another
process/service**. Requires `expr`/`var` -- a result holding only a precomputed
callable (no expression) cannot be serialized, since there'd be nothing to rebuild
from.

```python
import json
blob = json.dumps(res.to_dict())
res2 = FittingResult.from_dict(json.loads(blob))   # rebuilt, ready to predict
```

## Notes

- The `model` callable broadcasts scalars: a constant model still returns an array
  matching `x`.
- `cov` is `None` for exactly- or under-determined fits (e.g. EAC with
  `n_windows == n_params`); uncertainty methods then raise with a clear message.
- Parameter **order is by sorted name**, consistently across the whole library --
  so `coeffs`, `names`, and `params` always agree.
