# dtfit API reference

Complete reference for the **public** `dtfit` API — every name exported from the
top-level package and the `dtfit.diagnostics` submodule. Internals (anything
under a `_`-prefixed module) are not part of the public contract.

For the *ideas* behind these functions read [../guides/](../guides/); for the
*math* read [../methods/](../methods/). This reference is for looking up exact
signatures, arguments, return types, and behavior.

> **Looking for the experimental adaptations** (`fit_lsi_basis`, `fit_joint`,
> `boosted_fit`)? Those live in the separate `dtfit-experimental` package — see
> [../experimental/adaptations-api.md](../experimental/adaptations-api.md).
> (The overlapping-window ensemble `ensemble_fit` has been promoted — it's in
> [fitting.md](fitting.md#ensemble_fit).) This page covers the **stable** `dtfit` API.

## Conventions

- All fitters take 1-D NumPy arrays `x`, `y` and a model as a **sympy-style
  expression string** plus the name of its main variable, e.g.
  `fit_lsi(x, y, "a*exp(b*x)", "x")`.
- **Parameters are the free symbols** of the expression (everything except the
  variable), and results are ordered by **sorted parameter name** — a stable
  layout used everywhere. So `"a*exp(b*x)"` has parameters `[a, b]` in that order.
- Batch fitters return a [`FittingResult`](types.md); streaming filters expose
  `partial_fit` / `predict` / `params_`.
- Keyword-only arguments appear after `*` in the signatures below.

## The public surface, by area

| Area | Names | Page |
|---|---|---|
| **Batch fitting** | `fit_lsi`, `fit_eda`, `fit_eda_adaptive`, `fit_dsb`, `find_degree`, `fft_frequency_seed` | [fitting.md](fitting.md) |
| **Result type** | `FittingResult` | [types.md](types.md) |
| **sklearn estimator** | `NonlineRegressor` | [estimator.md](estimator.md) |
| **One-call entry points** | `auto_estimate`, `auto_forecast` | [auto.md](auto.md) |
| **Model framework** | `models`, `Model`, `suggest_models` (+ catalog families) | [models.md](models.md) |
| **Streaming / online** | `EDAFilter`, `LSIFilter`, `FilterBank`, `FusedChiSquareDetector` | [streaming.md](streaming.md) |
| **Scaling backends** | `fit_many`, `FittingProblem`, `BatchFittingResult`, `PartitionedLSI`, `PartitionedEDA`, `PartitionedBatchLSI`, `fit_lsi_batched`, `project_spectra` | [scaling.md](scaling.md) |
| **Diagnostics** | `fit_report`, `residual_diagnostics`, `FitDisplay`, `ResidualsDisplay` | [diagnostics.md](diagnostics.md) |
| **Logging** | `enable_logging`, `logger` | [below](#logging) |

## Import map

```python
# batch fitting
from dtfit import (fit_lsi, fit_eda, fit_eda_adaptive, fit_dsb,
                   find_degree, fft_frequency_seed, FittingResult)

# high-level entry points
from dtfit import auto_estimate, auto_forecast

# model framework
from dtfit import models, Model, suggest_models

# sklearn estimator
from dtfit import NonlineRegressor

# streaming
from dtfit import EDAFilter, LSIFilter, FilterBank, FusedChiSquareDetector

# scaling
from dtfit import (fit_many, FittingProblem, BatchFittingResult,
                   PartitionedLSI, PartitionedEDA, PartitionedBatchLSI,
                   fit_lsi_batched, project_spectra)

# diagnostics (submodule, not top-level — sklearn convention)
from dtfit.diagnostics import (fit_report, residual_diagnostics,
                               FitDisplay, ResidualsDisplay)

# logging
from dtfit import enable_logging, logger
```

<a name="logging"></a>
## Logging

`dtfit` is silent by default. Opt in to see what the solvers are doing:

```python
from dtfit import enable_logging
enable_logging()              # INFO-level chatter from the methods
enable_logging(level="DEBUG") # more detail
```

- **`enable_logging(level="INFO")`** — attach a handler to the library logger and
  set its level. Call once at startup.
- **`logger`** — the underlying `logging.Logger` (`"dtfit"`), if you want to wire
  it into your own logging configuration instead.

Internally the methods emit progress through a small `echo` helper (window counts,
selected polynomial degree, fitted coefficients); none of it prints unless you
enable logging.

## A 30-second example

```python
import numpy as np
from dtfit import fit_lsi
from dtfit.diagnostics import fit_report

x = np.linspace(0, 4, 200)
y = 0.5 + 2.0 * np.exp(0.5 * x) + np.random.default_rng(0).normal(0, 0.2, x.size)

res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x", k_star=6)
print(res.summary())                 # parameters ± standard errors
print(fit_report(res, x, y)["r2"])   # goodness of fit
y_hat = res.predict(x)               # evaluate the fitted model
```
