# dtfit

**D**ifferential-**t**ransformation **fit**ting: nonlinear smoothing and
forecasting on time-series / Big Data, using methods built in the scheme of
differential (non-Taylor) transformations. Developed as part of a PhD
dissertation on mathematical models of nonlinear smoothing and prediction.

## Installation

```bash
python -m venv .venv        # create an isolated environment
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate

# from the repo root (this package lives at packages/dtfit):
pip install -e packages/dtfit            # core
pip install -e 'packages/dtfit[viz]'     # + matplotlib plotting helpers
pip install -e 'packages/dtfit[dev]'     # + test/lint tooling

# Optional: the experimental adaptations + experiment suite (a separate
# distribution that depends on dtfit; never shipped in the dtfit wheel):
pip install -e packages/dtfit-experimental             # dtfit_experimental
pip install -e 'packages/dtfit-experimental[bench]'    # + the suite's baselines/plotting
```

A plain `venv` with PyPI wheels is the reference environment and works
identically on Windows and Linux. (Conda is not recommended on Windows: its
MKL-linked numpy only loads its LAPACK DLLs when the env is *activated*, so
tools that call the interpreter directly — VS Code, `pytest` — crash with a
delay-load error.)

### Optional compiled kernels (faster fitting)

The integral-based methods have an optional C backend (`dtfit._native`) for
their hot numeric loops — composite-Simpson window integrals and Gauss-Legendre
projections. Build it with clang (needs LLVM and, on Windows, the Visual Studio
Build Tools C++ workload):

```bash
python build_native.py            # compile into src/dtfit/
python build_native.py --clean    # remove the build artifacts
```

It is entirely optional: without it the package falls back to NumPy/SciPy with
identical results (`dtfit._kernels.HAVE_NATIVE` reports the active backend).
Building it speeds up the area-based methods substantially — the streaming
`EDAFilter` by roughly 6–13× and batch `EDA` by ~3× — while the
already-vectorized Legendre/LSI paths are largely unchanged.

## Quick start

```python
import numpy as np
import dtfit as dt

x = np.linspace(0, 10, 400)
y = ...  # your observations

# Batch fit, numeric (no polynomial pre-fit needed):
result = dt.fit_eda(x, y, "a*atan(w*x)", "x")
print(result.params)

# ...or through the scikit-learn compatible estimator:
reg = dt.NonlineRegressor("a0 + a1*x + a2*exp(a3*x)", "x", method="lsi")
reg.fit(x, y)
y_hat = reg.predict(x)
```

### Real-time / streaming

```python
flt = dt.EDAFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50)
for t, y in stream:           # bounded-cost per-sample update
    flt.partial_fit(t, y)
print(flt.params_)            # tracks time-varying parameters
```

### Picking a model (the model framework)

Most of fitting is *choosing the right structure*. `dtfit.models` is a catalog of
named families that **seed their own `p0`/`bounds` from the data**, compose with
`+`, and can be ranked for you:

```python
from dtfit import models, suggest_models

fit = models.logistic().fit(x, y)            # self-seeded; no p0/bounds to guess
fit = (models.linear() + models.sine()).fit(x, y)   # trend + cycle

for s in suggest_models(x, y)[:3]:           # infer the model from a scored shortlist
    print(s.name, s.r2, s.aic)
```

### Uncertainty & serialization

A `FittingResult` is self-describing — named parameters, uncertainty, and a
JSON-friendly round-trip:

```python
r = dt.fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1, 1])
r.params                       # {'a': ..., 'b': ...}
r.stderr(); r.confidence_intervals(0.95)
y_hat, y_std = r.predict(x, return_std=True)   # prediction band
dt.FittingResult.from_dict(r.to_dict())        # save / ship a fitted model
```

### Diagnostics & visualization

`dtfit.diagnostics` is **fit-aware** (it takes a `FittingResult`, not bare
arrays) and does *not* reimplement `sklearn.metrics` — use those / `scipy.stats`
for plain scalar metrics. It adds what's specific to evaluating a DT fit:
information criteria for model comparison and residual-structure tests, plus the
`*Display` plot helpers (which never call `plt.show()`).

```python
from dtfit.diagnostics import fit_report, residual_diagnostics, FitDisplay

print(fit_report(r, x, y))            # n, rmse, r2, aic, bic, durbin_watson, params±se
print(residual_diagnostics(r, x, y)) # autocorrelation / normality of residuals

FitDisplay.from_estimator(reg, x, y)  # data + fitted curve (needs the viz extra)
```

## Methods

- **LSI** (`method="lsi"`) — least-squares integral; numeric integral-OLS in the
  differential-transformation scheme (successor to DSBI).
- **EDA** (`method="eda"`) — equal differential areas / equal areas; numeric,
  integration-based and noise-robust (successor to DSBE).
- **EDAFilter** — recursive/online EDA with NIS drift detection for
  real-time tracking.
- **DSB** (`method="dsb"`) — symbolic differential spectra balance; kept as the
  analytical reference (requires a polynomial fit first in the pipeline).
- **PartitionedLSI / PartitionedEDA** — map-reduce LSI/EDA: because the
  empirical spectrum and the window areas are *integrals*, they are additive
  over a partition of the domain, so a stream of any length is fitted in one
  pass with O(order) state, and partitions reduce (sum) across workers. Use for
  big-data / distributed fitting (see the `dtfit-experimental`
  `cases/02_big_data_streaming` experiment).

### Scaling out

- `dtfit.fit_many(problems, n_jobs=-1)` fans many independent fits across cores
  (process or threading backend); the compiled kernels release the GIL, so
  thread pools accelerate the hot numeric loops.
- `dtfit.streaming.FilterBank` runs a bank of independent streaming filters
  (one per channel / satellite / axis) for multi-stream real-time tracking.
- `dtfit.fit_lsi_batched(x, Y, ...)` (and `dtfit.PartitionedBatchLSI` for the
  fused streaming variant) fits many channels that share a grid by
  expressing the LSI projection as one GEMM `S = Dᵀ·(w⊙Y)`, dispatched through a
  pluggable array backend (`numpy`/BLAS, or `cupy`/`torch` on a GPU; install the
  GPU extra, e.g. `pip install cupy-cuda13x`). Batching amortizes dispatch on the
  CPU (up to ~300× a per-channel loop) and runs on cuBLAS when data is
  device-resident. Because the projection is a low-arithmetic-intensity
  reduction, the GPU pays off only for resident / many-channel work, not a single
  streaming pass over host data — measured on an RTX 5080 in the
  `cases/08_gpu_batched_projection` experiment (fp32 resident ~16× CPU and
  bandwidth-saturated; streamed is PCIe-bound ≈ CPU).

Further experimental adaptations (pluggable orthogonal bases, robust
overlapping-window ensembles, joint multi-channel fits, stage-wise boosting,
adaptive windows) live in the separate **`dtfit-experimental`** package
(`dtfit_experimental`); their cross-application evaluation is in its experiment
suite ([../dtfit-experimental/src/dtfit_experimental/experiments/cases/REPORTS.md](../dtfit-experimental/src/dtfit_experimental/experiments/cases/REPORTS.md)).

Each method's mathematical grounding (in differential / non-Taylor
transformations), full algorithm, optimizations, guards, applicability, usage
figures and comparison tables are documented in the
[project wiki](https://github.com/ringavirda/science-nonline/wiki/Methods).

Core dependencies: numpy, scipy, sympy, scikit-learn. Plotting helpers require
the optional `viz` extra (matplotlib).
