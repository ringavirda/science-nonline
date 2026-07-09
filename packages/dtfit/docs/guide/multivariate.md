# Multivariate data

**dtfit is one-dimensional by design.** Both fitting criteria are built on a
single scalar axis: LSI projects the signal onto Legendre polynomials over an
interval `[x₀, xₙ]`, and EAC integrates areas under windows on that same axis.
There is no meaningful "integral criterion" over a scattered cloud of points in
two or more input dimensions, so **multivariate `X` (several predictors) is not
supported** — every entry point raises a clear error rather than silently
producing a wrong fit:

```python
import numpy as np, dtfit as dt
X = np.random.default_rng(0).normal(size=(200, 2))   # two predictors
y = np.random.default_rng(1).normal(size=200)
dt.fit_lsi(X, y, "a*x", "x")
# ValueError: data_x and data_y must be 1-D; got shapes (200, 2) and (200,).
# dtfit's integral criteria (LSI/EAC) are one-dimensional, so multivariate X
# (several predictors) is not supported. ...
```

This is a hard scope limit, not a passing validation quirk — treat it as "use a
general nonlinear-regression tool for genuinely multivariate models."

## The case people usually mean: a sum of components on **one** axis

Most of the time a user reaching for "multivariate" actually has a **single
time/space axis** whose signal is a *sum of components* — a trend plus a cycle,
a baseline plus a peak, two decays. That is still 1-D, and dtfit fits it
directly by **composing 1-D models with `+`** (same axis, parameters
auto-renamed, each component seeded on the previous one's residual):

```python
from dtfit import models
# y(t) = a0 + a1*t + A*sin(w*t + p)  -- trend + cycle, both in t
fit = (models.linear() + models.sine()).fit(t, y)
print(fit.params)          # a0, a1, A, w, p in one fit
```

`models.linear() + models.sine()` is `a0 + a1*t + A*sin(w*t)` **in the one
variable `t`** — it is *not* `g(t) + h(u)` over two different inputs. If your
"two variables" are really one axis and its components, this is the answer.

## Genuinely separable multi-predictor models

If you truly have distinct predictors and an additively-separable model
`y = g₁(x₁) + g₂(x₂)`, dtfit will not fit it in one call (each fitter takes one
`x`). Two honest options:

1. **Backfitting** — alternate 1-D dtfit fits on the partial residuals until they
   stabilize. Each step is a normal 1-D `fit_lsi` / `fit_eac`, so you keep
   dtfit's seeding, robustness and uncertainty per component:

   ```python
   g2 = np.zeros_like(y)
   for _ in range(10):                       # a few passes converge for smooth g
       f1 = dt.fit_eac(x1, y - g2, "a*x1 + b", "x1")   # g1 on the x1 residual
       g1 = np.asarray(f1.model(x1))
       f2 = dt.fit_eac(x2, y - g1, "c*sin(w*x2)", "x2")  # g2 on the x2 residual
       g2 = np.asarray(f2.model(x2))
   ```

2. **A general nonlinear least-squares tool** — for a non-separable
   `f(x₁, x₂, …; θ)`, use `scipy.optimize.curve_fit` / `lmfit`, which are
   dimension-agnostic. dtfit is complementary to them (see
   [dtfit vs SciPy](../comparison.md)); its edge is 1-D streaming, out-of-memory
   map-reduce, embedded deployment, and robustness without tuning — none of which
   apply to a scattered multivariate cloud.

A future dtfit release may add a general nD nonlinear-least-squares engine that
reuses the model/seeder/`suggest_models`/uncertainty ergonomics while leaving the
integral criteria 1-D — but that is deliberately a separate method, not a
reinterpretation of LSI/EAC.
