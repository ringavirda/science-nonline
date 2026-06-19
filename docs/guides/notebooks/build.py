"""Build & execute the dtfit guide notebooks.

Regenerates every ``*.ipynb`` in this folder from the cell definitions below,
executing each so figures and outputs are embedded. Run from anywhere:

    python docs/guides/notebooks/build.py            # build + execute all
    python docs/guides/notebooks/build.py 01 05      # only notebooks whose name starts 01/05

Requires the doc-build extras: ``pip install nbformat nbclient ipykernel``
plus ``dtfit[viz]`` (matplotlib). Execution uses whichever kernel is passed via
``--kernel`` (default: the ``dtfit-docs`` kernelspec registered with
``python -m ipykernel install --user --name dtfit-docs``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
KERNEL = "dtfit-docs"


def md(text: str) -> tuple[str, str]:
    return ("md", text.strip("\n"))


def code(src: str) -> tuple[str, str]:
    return ("code", src.strip("\n"))


# Standard preamble dropped at the top of every notebook.
PREAMBLE = code(
    """
%matplotlib inline
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Fitting at extreme parameter trials can overflow exp() harmlessly; keep the
# guide output clean.
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
rng = np.random.default_rng(0)
"""
)


# --------------------------------------------------------------------------- #
# 01 - Quickstart
# --------------------------------------------------------------------------- #
NB_01 = [
    md(
        """
# dtfit - Quickstart

**dtfit** fits models that are *nonlinear in their parameters* - exponential,
transcendental, oscillatory, mixed - for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations (from the
author's PhD work).

You bring a model as a small **sympy expression string** (e.g. `"a*exp(b*t)"`)
and your data; dtfit recovers the parameters. The public API spans batch
fitters, a self-seeding model catalog, a scikit-learn estimator, streaming
trackers, and scale-out helpers.

```
pip install dtfit            # core
pip install "dtfit[viz]"     # + matplotlib for the plots in these guides
```

This notebook gets you to a first fit in a minute. The rest of the series:

| # | Guide |
|---|-------|
| 02 | Fitting methods - LSI / EDA / DSB |
| 03 | Model catalog, `suggest_models`, `auto_estimate` / `auto_forecast` |
| 04 | scikit-learn `NonlineRegressor` |
| 05 | Streaming / online trackers |
| 06 | Scaling - parallel, batched, partitioned |
| 07 | Diagnostics, visualization & serialization |
"""
    ),
    PREAMBLE,
    md(
        """
## Your first fit - `fit_lsi`

Pick a model expression (sympy syntax) and the variable name. Everything in the
expression except the variable is a free parameter. Here: `a` and `b`.
"""
    ),
    code(
        """
from dtfit import fit_lsi

x = np.linspace(0, 3, 200)
y = 1.4 * np.exp(0.8 * x) + rng.normal(0, 0.15, x.size)

res = fit_lsi(x, y, "a*exp(b*t)", "t")
print(res.summary())
"""
    ),
    md(
        """
The fit comes back as a **`FittingResult`** - self-describing (model expression,
parameter names, coefficients, covariance) and ready to predict.
"""
    ),
    code(
        """
print("params:", res.params)        # {name: value}
print("coeffs:", res.coeffs)        # ordered by parameter name

xs = np.linspace(x.min(), x.max(), 400)
plt.scatter(x, y, s=10, color="0.6", label="data")
plt.plot(xs, res.predict(xs), "r-", lw=2, label="fit")
plt.plot(xs, 1.4 * np.exp(0.8 * xs), "k--", lw=1, label="truth")
plt.legend(); plt.title("fit_lsi:  a * exp(b * t)"); plt.show()
"""
    ),
    md(
        """
## Uncertainty

When the system is overdetermined the result carries a parameter **covariance**,
so you get standard errors, confidence intervals, and prediction bands
(propagated by the delta method).
"""
    ),
    code(
        """
print("std errors :", res.stderr())
print("95% CI     :", res.confidence_intervals())

y_hat, y_sd = res.predict(xs, return_std=True)
plt.scatter(x, y, s=8, color="0.6")
plt.plot(xs, y_hat, "r-", label="prediction")
plt.fill_between(xs, y_hat - 2 * y_sd, y_hat + 2 * y_sd,
                 color="r", alpha=0.2, label="+/- 2 sigma")
plt.legend(); plt.title("prediction with uncertainty band"); plt.show()
"""
    ),
    md(
        """
## Don't want to choose the estimator? Let dtfit route it

`auto_estimate` picks the estimator variant from the signal's *shape*; the model
catalog and `suggest_models` can even rank candidate families for you
(notebook 03).
"""
    ),
    code(
        """
from dtfit import auto_estimate

res2 = auto_estimate(x, y, "a*exp(b*t)", "t")
print(res2.summary())
"""
    ),
]


# --------------------------------------------------------------------------- #
# 02 - Fitting methods
# --------------------------------------------------------------------------- #
NB_02 = [
    md(
        """
# Fitting methods - LSI, EDA, DSB

Three differential-transformation batch fitters, each with a different
*measurement* of "fit":

- **LSI** (`fit_lsi`) - integral least-squares in a reconditioned **Legendre
  spectrum**. The general default; strong on smooth bulk shapes and (with the
  oscillatory recipe) cycles.
- **EDA** (`fit_eda`, `fit_eda_adaptive`) - **equal-areas** integral matching
  over windows. Strong on transients, peaks and saturating shapes; supports
  robust losses for outliers.
- **DSB** (`fit_dsb`) - symbolic **differential-spectra balance** against a
  polynomial pre-fit; an analytical reference method.

All three return a `FittingResult`.
"""
    ),
    PREAMBLE,
    md("## LSI - `fit_lsi`\n\n`k_star` sets the Legendre spectral order matched against the data."),
    code(
        """
from dtfit import fit_lsi

x = np.linspace(0, 4, 300)
y = 0.5 + 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.2, x.size)

res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x", k_star=6)
print(res.summary())

plt.scatter(x, y, s=8, color="0.6", label="data")
plt.plot(x, res.predict(x), "r-", lw=2, label="LSI fit")
plt.legend(); plt.title("LSI: offset + exponential"); plt.show()
"""
    ),
    md(
        """
### The oscillatory recipe

A smoothed, low-order spectral fit *erases* cycles. For oscillatory models pass
`freq_param` (its initial guess is seeded from the data's FFT peak via
`fft_frequency_seed`) - this implies the oscillatory recipe (no smoothing, order
raised to resolve the cycle).
"""
    ),
    code(
        """
from dtfit import fft_frequency_seed

xs = np.linspace(0, 10, 400)
y = 2.0 * np.sin(1.7 * xs + 0.5) + rng.normal(0, 0.10, xs.size)

print("FFT frequency seed:", round(fft_frequency_seed(xs, y), 4))
res = fit_lsi(xs, y, "A*sin(w*x + p)", "x", freq_param="w")
print("recovered:", {k: round(v, 3) for k, v in res.params.items()})

plt.scatter(xs, y, s=8, color="0.6")
plt.plot(xs, res.predict(xs), "r-", lw=2)
plt.title("LSI oscillatory recipe:  A sin(w x + p)"); plt.show()
"""
    ),
    md(
        """
## EDA - `fit_eda` / `fit_eda_adaptive`

Equal-areas matches *integrals* over windows, so it naturally averages over
sparse outliers; the `loss="soft_l1"` option adds extra protection under heavier
contamination. The **adaptive** variant places window edges by curvature -
narrow where the signal bends, wide where it is smooth - which suits localized
transients and peaks.
"""
    ),
    code(
        """
from dtfit import fit_eda

x = np.linspace(0, 5, 250)
y = 3.0 * np.arctan(1.5 * x) + rng.normal(0, 0.1, x.size)   # truth: a=3, w=1.5
y_out = y.copy()
idx = rng.choice(x.size, 12, replace=False)
y_out[idx] += rng.normal(0, 3.0, 12)                        # scattered outliers

res = fit_eda(x, y_out, "a*atan(w*x)", "x", loss="soft_l1")
print("truth    : {'a': 3.0, 'w': 1.5}")
print("recovered:", {k: round(v, 3) for k, v in res.params.items()})

plt.scatter(x, y_out, s=10, color="0.6", label="data + outliers")
plt.plot(x, res.predict(x), "r-", lw=2, label="EDA (soft_l1)")
plt.plot(x, 3.0 * np.arctan(1.5 * x), "k--", lw=1, label="truth")
plt.legend(); plt.title("EDA is robust to sparse outliers"); plt.show()
"""
    ),
    code(
        """
from dtfit import fit_eda_adaptive

x = np.linspace(0, 6, 300)
y = 5.0 * x * np.exp(-1.2 * x) + rng.normal(0, 0.03, x.size)   # rise-then-decay peak

res = fit_eda_adaptive(x, y, "a*x*exp(-b*x)", "x", window_mode="curvature")
print("params:", {k: round(v, 3) for k, v in res.params.items()})

plt.scatter(x, y, s=8, color="0.6")
plt.plot(x, res.predict(x), "r-", lw=2)
plt.title("curvature-adaptive EDA on a transient peak"); plt.show()
"""
    ),
    md(
        """
## DSB - `fit_dsb`

DSB equates the model's Maclaurin spectrum to a polynomial pre-fit's, order by
order. Build the **ascending** polynomial coefficients (the data's Taylor
spectrum) with `find_degree` + `np.polyfit`, then balance. (`NonlineRegressor`
with `method="dsb"` runs this pre-fit for you - notebook 04.)
"""
    ),
    code(
        """
from dtfit import fit_dsb, find_degree

x = np.linspace(0, 1.5, 200)
y = 1.5 * np.exp(1.1 * x) + rng.normal(0, 0.02, x.size)    # truth: a=1.5, b=1.1

deg = find_degree(x, y)                  # BIC-selected polynomial degree
pc = np.polyfit(x, y, deg)[::-1]         # ascending coeffs = data Maclaurin spectrum
res = fit_dsb(pc, "a*exp(b*x)", "x")
print("polynomial degree:", deg)
print(res.summary())                     # recovers a~1.5, b~1.1

plt.scatter(x, y, s=8, color="0.6")
plt.plot(x, res.predict(x), "r-", lw=2)
plt.title("DSB symbolic differential-spectra balance"); plt.show()
"""
    ),
    md(
        """
## Comparing fits

`fit_report` turns any `FittingResult` into r^2 / RMSE / AIC / BIC (notebook 07).
"""
    ),
    code(
        """
from dtfit import fit_eda
from dtfit.diagnostics import fit_report

x = np.linspace(0, 3, 200)
y = 1.4 * np.exp(0.8 * x) + rng.normal(0, 0.15, x.size)
for name, r in [("LSI", fit_lsi(x, y, "a*exp(b*x)", "x")),
                ("EDA", fit_eda(x, y, "a*exp(b*x)", "x"))]:
    rep = fit_report(r, x, y)
    print(f"{name}:  r2={rep['r2']:.4f}   rmse={rep['rmse']:.4f}   aic={rep['aic']:.1f}")
"""
    ),
]


# --------------------------------------------------------------------------- #
# 03 - Models & automatic fitting
# --------------------------------------------------------------------------- #
NB_03 = [
    md(
        """
# Models & automatic fitting

The studies behind dtfit concluded that *picking the structurally-correct model
is the whole game*. `dtfit.models` makes that ergonomic: a catalog of named,
**self-seeding** families (they read `p0` / `bounds` off the data), composition
with `+`, and a recommender that ranks families by AIC.
"""
    ),
    PREAMBLE,
    md("## The catalog\n\n`CATALOG` maps a name to a family factory, grouped by `category`."),
    code(
        """
from collections import defaultdict
from dtfit.models import CATALOG

by_cat = defaultdict(list)
for name, factory in CATALOG.items():
    by_cat[factory().category].append(name)
for cat, names in by_cat.items():
    print(f"{cat:11s}: {', '.join(names)}")
"""
    ),
    md(
        """
## Self-seeding fit - no `p0` needed

Each family reads sensible initial values and bounds off the data, then fits
through the stable engines (routing via `auto_estimate`).
"""
    ),
    code(
        """
from dtfit import models

x = np.linspace(0, 10, 200)
y = 8.0 / (1 + np.exp(-0.9 * (x - 5))) + rng.normal(0, 0.2, x.size)

fit = models.logistic().fit(x, y)        # no p0 / bounds supplied
print(fit.summary())

plt.scatter(x, y, s=8, color="0.6")
plt.plot(x, fit.predict(x), "r-", lw=2)
plt.title("models.logistic() - self-seeded"); plt.show()
"""
    ),
    md(
        """
## Composition with `+`

Add families to build structure (trend + cycle, a sum of peaks). The second
component is seeded on the residual of the first, so the cycle's frequency and
amplitude are read off the detrended data.
"""
    ),
    code(
        """
x = np.linspace(0, 12, 240)
y = (1.0 + 0.6 * x) + 2.5 * np.sin(1.3 * x) + rng.normal(0, 0.2, x.size)

model = models.linear() + models.sine()
print(model)
fit = model.fit(x, y)

plt.scatter(x, y, s=8, color="0.6")
plt.plot(x, fit.predict(x), "r-", lw=2)
plt.title("models.linear() + models.sine()"); plt.show()
"""
    ),
    md(
        """
## Which family? - `suggest_models`

Fits a shape-based shortlist of the catalog and ranks them best-first by AIC.
Each `Suggestion` carries the fitted `Model`, its `FittingResult` and the full
`report`.
"""
    ),
    code(
        """
from dtfit import suggest_models

x = np.linspace(0, 8, 200)
y = 3.0 * np.exp(-((x - 4.0) ** 2) / (2 * 0.8 ** 2)) + rng.normal(0, 0.04, x.size)

ranked = suggest_models(x, y, top=5)
for s in ranked:
    print(f"{s.name:24s}  r2={s.r2:.4f}   aic={s.aic:8.1f}")

best = ranked[0]
plt.scatter(x, y, s=8, color="0.6", label="data")
plt.plot(x, best.result.predict(x), "r-", lw=2, label=f"best: {best.name}")
plt.legend(); plt.title("suggest_models ranks the catalog by AIC"); plt.show()
"""
    ),
    md(
        """
## `auto_estimate` - route by shape

Detects oscillation, transients, peaks and outliers and dispatches to the
estimator variant the parameter-estimation study validated for each.
"""
    ),
    code(
        """
from dtfit import auto_estimate

x = np.linspace(0, 10, 300)
y = 1.5 * np.sin(2.1 * x) + rng.normal(0, 0.1, x.size)
res = auto_estimate(x, y, "A*sin(w*x)", "x", freq_param="w")   # oscillatory route
print("recovered:", {k: round(v, 3) for k, v in res.params.items()})
"""
    ),
    md(
        """
## `auto_forecast` - fit then extrapolate

Routes the model class (saturating growth -> logistic, a detected cycle ->
linear+seasonal, else a quadratic level) with guards that fall back to
persistence on a near-random-walk series and drop a runaway quadratic to linear.
"""
    ),
    code(
        """
from dtfit import auto_forecast

t = np.arange(120)
series = 10.0 / (1 + np.exp(-0.12 * (t - 45))) + rng.normal(0, 0.015, t.size)
cut, h = 90, 30
fc = auto_forecast(t[:cut], series[:cut], horizon=h)   # routes to logistic
print("forecast end:", round(fc[-1], 2), " actual end:", round(series[cut + h - 1], 2))

plt.plot(t[:cut], series[:cut], label="history")
plt.plot(np.arange(cut, cut + h), fc, "r-", lw=2, label="forecast")
plt.plot(np.arange(cut, 120), series[cut:120], "k--", lw=1, label="actual")
plt.legend(); plt.title("auto_forecast - saturating growth routes to logistic"); plt.show()
"""
    ),
]


# --------------------------------------------------------------------------- #
# 04 - scikit-learn estimator
# --------------------------------------------------------------------------- #
NB_04 = [
    md(
        """
# scikit-learn estimator - `NonlineRegressor`

`NonlineRegressor` wraps the LSI / EDA / DSB methods behind the standard
estimator API (`fit` / `predict` / `score`), so it composes with `Pipeline`,
`GridSearchCV` and `cross_val_score`. It takes a single input feature (the
model's variable).
"""
    ),
    PREAMBLE,
    md("## Fit / predict / score"),
    code(
        """
from dtfit import NonlineRegressor

X = np.linspace(0, 3, 200).reshape(-1, 1)
y = 1.4 * np.exp(0.8 * X.ravel()) + rng.normal(0, 0.15, X.shape[0])

reg = NonlineRegressor("a*exp(b*x)", "x", method="lsi").fit(X, y)
print("coef_ :", np.round(reg.coef_, 4))
print("R2    :", round(reg.score(X, y), 4))

xs = np.linspace(0, 3, 300).reshape(-1, 1)
plt.scatter(X, y, s=8, color="0.6", label="data")
plt.plot(xs.ravel(), reg.predict(xs), "r-", lw=2, label="NonlineRegressor")
plt.legend(); plt.show()
"""
    ),
    md(
        """
## Hyperparameter search with `GridSearchCV`

Treat the method and spectral order as hyperparameters; sklearn does the rest.
"""
    ),
    code(
        """
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([("fit", NonlineRegressor("a0 + a1*exp(a2*x)", "x"))])
grid = GridSearchCV(
    pipe,
    {"fit__method": ["lsi", "eda"], "fit__k_star": [4, 6]},
    cv=3, scoring="r2",
)
grid.fit(X, y)
print("best params:", grid.best_params_)
print("best CV R2 :", round(grid.best_score_, 4))
"""
    ),
    md("## Cross-validation"),
    code(
        """
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    NonlineRegressor("a*exp(b*x)", "x"), X, y, cv=4, scoring="r2"
)
print("per-fold R2:", np.round(scores, 3))
print("mean R2    :", round(scores.mean(), 4))
"""
    ),
]


# --------------------------------------------------------------------------- #
# 05 - Streaming / online trackers
# --------------------------------------------------------------------------- #
NB_05 = [
    md(
        """
# Streaming / online trackers

These estimators ingest one sample at a time with bounded per-update cost
(`partial_fit(t, y)`), for control loops and big-data streams. Each filter is
the streaming twin of a batch method and carries built-in **drift detection**.

- `EDAFilter` - streaming equal-areas (twin of `fit_eda`).
- `LSIFilter` - streaming Legendre spectrum (twin of `fit_lsi`); same API plus
  an `order=` knob.
- `FilterBank` - many independent streams updated in lockstep.
- `FusedChiSquareDetector` - pools a bank's innovations into one fault test.
"""
    ),
    PREAMBLE,
    md(
        """
## `EDAFilter` - track a drifting parameter

The exponential rate `b` jumps mid-stream; the filter re-adapts (its drift test
detects the change and re-arms the covariance).
"""
    ),
    code(
        """
from dtfit import EDAFilter

T = 500
t = np.linspace(0, 8, T)
b_true = np.where(t < 4, 0.30, 0.55)
y = np.exp(b_true * t) + rng.normal(0, 0.05, T)

flt = EDAFilter("exp(b*t)", "t", p0=[0.2], window_size=40, q_diag=[1e-4], r=0.5)
track = []
for ti, yi in zip(t, y):
    flt.partial_fit(ti, yi)
    track.append(flt.p[0])

plt.plot(t, b_true, "k--", label="true b")
plt.plot(t, track, "r-", lw=1.3, label="EDAFilter estimate")
plt.ylim(0, 0.8); plt.legend(); plt.title("online tracking with a mid-stream step")
plt.xlabel("t"); plt.ylabel("b"); plt.show()
"""
    ),
    md(
        """
## `FilterBank` - many streams at once

Build `K` identically-configured filters for one model and drive them over a
block of samples. `run` returns final per-stream parameters and drift counts.
"""
    ),
    code(
        """
from dtfit import FilterBank

K = 4
t = np.linspace(0, 20, 400)
b_true = np.array([0.30, 0.50, 0.70, 0.90])
Y = np.column_stack([np.exp(b * t) + rng.normal(0, 0.05, t.size) for b in b_true])

bank = FilterBank.from_model("a*exp(b*t)", "t", K,
                             p0=[1.0, 0.4], window_size=40,
                             q_diag=[1e-4, 1e-3], r=0.3)
out = bank.run(t, Y, n_jobs=1)        # n_jobs>1 fans streams across threads
print("recovered b:", np.round(out["params"][:, 1], 3))
print("true b     :", b_true)
print("drift counts:", out["n_drifts"])
"""
    ),
    md(
        """
## `FusedChiSquareDetector` - catch a shared fault

A change that hits **every** stream (here an amplitude collapse at t = 20) is
weak in any single innovation but strong in the pooled `chi2(K)` statistic. The
detector ingests samples into the bank and flags the step where it fires.
"""
    ),
    code(
        """
from dtfit import FilterBank

K = 3
t = np.linspace(0, 40, 600)
amp = np.where(t < 20, 1.0, 0.5)                    # shared fault at t = 20
phases = (0.0, 0.7, 1.4)
Y = np.column_stack([amp * np.sin(1.2 * t + p) + rng.normal(0, 0.05, t.size)
                     for p in phases])

bank = FilterBank.from_model("A*sin(1.2*t + p)", "t", K,
                             p0=[1.0, 0.0], window_size=40)
det = bank.fused_detector(alpha=1e-4)
fired = [t[s] for s in range(t.size) if det.update(t[s], Y[s])]

print("flags raised :", det.n_flags_)
print("first flag at : t =", round(fired[0], 1) if fired else None, "(fault at t=20)")

plt.plot(t, Y[:, 0], lw=0.8, label="stream 0")
for f in fired:
    plt.axvline(f, color="r", ls="--", alpha=0.6)
plt.axvline(20, color="k", ls=":", label="true fault")
plt.legend(); plt.title("fused fault detection across streams"); plt.show()
"""
    ),
]


# --------------------------------------------------------------------------- #
# 06 - Scaling
# --------------------------------------------------------------------------- #
NB_06 = [
    md(
        """
# Scaling - parallel, batched, partitioned

Per-problem independence makes dtfit embarrassingly parallel. Three tools:

- `fit_many` (+ `FittingProblem`, `BatchFittingResult`) - fan independent fits
  across CPU cores.
- `fit_lsi_batched` / `project_spectra` - many channels' projections in one
  GEMM, on a pluggable backend (`numpy` / `cupy` / `torch`).
- `PartitionedLSI` / `PartitionedEDA` / `PartitionedBatchLSI` - one-pass /
  distributed (map-reduce) estimators for streams too big for memory.
"""
    ),
    PREAMBLE,
    md(
        """
## `fit_many` - independent fits across cores

Each `FittingProblem` is a self-contained, picklable spec. A failed fit is
captured as `error` rather than aborting the batch. Use `n_jobs=-1` for all cores.
"""
    ),
    code(
        """
from dtfit import fit_many, FittingProblem

problems = []
for i, b in enumerate([0.4, 0.6, 0.8, 1.0, 1.2]):
    x = np.linspace(0, 3, 200)
    y = (1 + 0.2 * i) * np.exp(b * x) + rng.normal(0, 0.05, x.size)
    problems.append(FittingProblem(
        x=x, y=y, expr="a*exp(b*t)", var="t",
        method="lsi", kwargs={"p0": [1.0, 1.0]}, label=f"ch{i}"))

results = fit_many(problems, n_jobs=1)
for r in results:
    msg = r.error if r.error else f"coeffs={np.round(r.coeffs, 3)}"
    print(f"{r.label}: {msg}")
"""
    ),
    md(
        """
## `project_spectra` / `fit_lsi_batched` - one GEMM for many channels

All channels share a grid `x`; their empirical spectra are computed together,
then each small spectral-match solve runs on the host.
"""
    ),
    code(
        """
from dtfit import project_spectra, fit_lsi_batched

x = np.linspace(0, 3, 300)
b_true = [0.4, 0.6, 0.8, 1.0]
Y = np.column_stack([np.exp(b * x) + rng.normal(0, 0.03, x.size) for b in b_true])

spectra = project_spectra(x, Y, order=6)            # (B, n_coef), one GEMM
print("spectra shape:", spectra.shape)

fits = fit_lsi_batched(x, Y, "a*exp(b*t)", "t", order=6, p0=[1.0, 1.0])
print("recovered b:", [round(f.params["b"], 3) for f in fits])
print("true b     :", b_true)
"""
    ),
    md(
        """
## `PartitionedLSI` - one pass, fixed memory

Fold chunks of a stream into an additive projection accumulator, then fit once
at the end. Consecutive `update` calls are made *exactly* additive (equal to a
single whole-domain projection).
"""
    ),
    code(
        """
from dtfit import PartitionedLSI

acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 5), order=6)
for x_chunk in np.array_split(np.linspace(0, 5, 5000), 10):
    y_chunk = 1.3 * np.exp(0.7 * x_chunk) + rng.normal(0, 0.05, x_chunk.size)
    acc.update(x_chunk, y_chunk)

res = acc.fit(p0=[1.0, 1.0])
print("one-pass fit:", {k: round(v, 3) for k, v in res.params.items()},
      f"(n_samples={acc.n_samples})")
"""
    ),
    md(
        """
## Map-reduce with `merge`

Workers each accumulate over a shard, then the partials are reduced with
`merge` - the distributed estimator.
"""
    ),
    code(
        """
def shard_accumulator(x_shard):
    a = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 5), order=6)
    y = 1.3 * np.exp(0.7 * x_shard) + rng.normal(0, 0.05, x_shard.size)
    return a.update(x_shard, y)

shards = np.array_split(np.linspace(0, 5, 5000), 4)
partials = [shard_accumulator(s) for s in shards]      # the "map" (parallelizable)
reduced = partials[0]
for a in partials[1:]:
    reduced = reduced.merge(a)                          # the "reduce"

print("map-reduce fit:", {k: round(v, 3) for k, v in reduced.fit(p0=[1.0, 1.0]).params.items()})
"""
    ),
    md(
        """
The backend is pluggable: `project_spectra(..., backend="auto")` uses a GPU
(`cupy` / `torch`) when one is available, else NumPy. `PartitionedBatchLSI`
combines the multi-channel batching with the one-pass reduce.
"""
    ),
]


# --------------------------------------------------------------------------- #
# 07 - Diagnostics, visualization & serialization
# --------------------------------------------------------------------------- #
NB_07 = [
    md(
        """
# Diagnostics, visualization & serialization

Evaluate a fitted dtfit model: information criteria, residual-structure tests,
ready-made plots, opt-in logging, and round-trip serialization. (For plain
scalar metrics on arrays, use `sklearn.metrics` / `scipy.stats` directly.)
"""
    ),
    PREAMBLE,
    code(
        """
from dtfit import fit_lsi

x = np.linspace(0, 4, 250)
y = 0.5 + 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.2, x.size)
res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x")
"""
    ),
    md("## `fit_report` - sample/param counts, RSS, RMSE, r^2, AIC/BIC, Durbin-Watson"),
    code(
        """
from dtfit.diagnostics import fit_report

for k, v in fit_report(res, x, y).items():
    print(f"{k:14s}: {v}")
"""
    ),
    md("## `residual_diagnostics` - residual structure (autocorrelation, normality)"),
    code(
        """
from dtfit.diagnostics import residual_diagnostics

rd = residual_diagnostics(res, x, y)
print("Durbin-Watson :", round(rd["durbin_watson"], 3))
print("lag-1 autocorr:", round(rd["lag1_autocorr"], 3))
print("normality p   :", round(rd["normality_p"], 3))
"""
    ),
    md(
        """
## Ready-made plots - `FitDisplay`, `ResidualsDisplay`

scikit-learn-style display objects with `from_predictions` / `from_estimator`
constructors (require matplotlib).
"""
    ),
    code(
        """
from dtfit.diagnostics import FitDisplay, ResidualsDisplay

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
FitDisplay.from_predictions(x, y, res.predict(x), ax=axes[0], estimator_name="LSI")
ResidualsDisplay.from_predictions(y, res.predict(x), ax=axes[1], estimator_name="LSI")
plt.tight_layout(); plt.show()
"""
    ),
    code(
        """
from dtfit import NonlineRegressor

reg = NonlineRegressor("a0 + a1*exp(a2*x)", "x").fit(x.reshape(-1, 1), y)
FitDisplay.from_estimator(reg, x.reshape(-1, 1), y)
plt.title("FitDisplay.from_estimator"); plt.show()
"""
    ),
    md(
        """
## Opt-in logging

dtfit logs through the standard `logging` module under the `"dtfit"` logger and
attaches only a `NullHandler` by default. `enable_logging(DEBUG)` surfaces the
fitting internals.
"""
    ),
    code(
        """
import logging
from dtfit import enable_logging

enable_logging(logging.DEBUG)
_ = fit_lsi(x, y, "a*exp(b*x)", "x")     # emits fitting detail
logging.getLogger("dtfit").handlers = [logging.NullHandler()]   # quiet again
"""
    ),
    md(
        """
## Serialize a fit - `to_dict` / `from_dict`

Everything needed to rebuild the model (expression, names, coefficients,
covariance) round-trips through a JSON-friendly dict.
"""
    ),
    code(
        """
from dtfit import FittingResult

blob = res.to_dict()
print(blob)
restored = FittingResult.from_dict(blob)
print("round-trip params:", {k: round(v, 3) for k, v in restored.params.items()})
"""
    ),
]


NOTEBOOKS = {
    "01_quickstart": NB_01,
    "02_fitting_methods": NB_02,
    "03_models_and_auto": NB_03,
    "04_sklearn_estimator": NB_04,
    "05_streaming": NB_05,
    "06_scaling": NB_06,
    "07_diagnostics": NB_07,
}


def make_nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(body) if kind == "md" else new_code_cell(body)
        for kind, body in cells
    ]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3",
                       "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


def main(argv: list[str]) -> int:
    selectors = [a for a in argv if not a.startswith("-")]
    names = [n for n in NOTEBOOKS
             if not selectors or any(n.startswith(s) for s in selectors)]
    failures: list[str] = []
    for name in names:
        nb = make_nb(NOTEBOOKS[name])
        print(f"executing {name} ({len(nb.cells)} cells) ...", flush=True)
        client = NotebookClient(nb, timeout=600, kernel_name=KERNEL,
                                allow_errors=True)
        client.execute()
        for i, cell in enumerate(nb.cells):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    failures.append(f"{name} cell {i}: "
                                    f"{out['ename']}: {out['evalue']}")
        nbf.write(nb, HERE / f"{name}.ipynb")
        print(f"  wrote {name}.ipynb")
    print("\n=== SUMMARY ===")
    if failures:
        print(f"{len(failures)} cell error(s):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"all {len(names)} notebook(s) executed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
