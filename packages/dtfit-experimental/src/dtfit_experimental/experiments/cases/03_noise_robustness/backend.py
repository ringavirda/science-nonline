"""Backend infrastructure for the noise & robustness sweep experiment.

This module is the **single source of truth for the simulation and estimation
code** behind ``03_noise_robustness.ipynb``; the notebook imports it and does all
the presentation (tables, figures, narrative). Keeping the infra here means the
data generator, sweeps and baselines are defined once and the notebook stays a
thin, rerunnable layer over them.

The experiment maps fitting accuracy across noise level, outlier fraction and
sample size for four model families, against the established methods a curve-
fitter reaches for. It provides:

* the **model families** -- :data:`FAMILIES` (expr / true params / clean signal /
  ``curve_fit`` func / ``x``-grid) and the noisy-data generator :func:`_noisy`;
* the **scorer** -- :func:`r2_clean` (R-squared against the *clean* signal);
* the **sweeps** -- :func:`noise_sweep` (R2 vs Gaussian noise per family),
  :func:`outlier_sweep` (stock vs the robust overlapping-window ensemble #3 and
  the soft-L1 loss under gross outliers) and :func:`param_grid_parallel` (median
  EAC parameter-recovery error over a noise x size grid, fanned via
  :func:`dtfit.fit_many`).

The dtfit integral fits (:func:`dtfit.fit_eac`, :func:`dtfit.fit_lsi`) are scored
against SciPy ``curve_fit``, ``numpy.polyfit`` and a scikit-learn MLP. SciPy and
scikit-learn are optional: the relevant baseline yields ``nan`` (and the caller
skips it) when the dependency is missing.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit import FittingProblem, fit_many, ensemble_fit  # ensemble_fit promoted

from dtfit_experimental.experiments.common import metrics
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "FAMILIES",
    "r2_clean",
    "noise_sweep",
    "outlier_sweep",
    "param_grid_parallel",
]

# family: expr, var, true params (name->val), clean signal fn, curve_fit func, p0
FAMILIES = {
    "exponential": dict(
        expr="a*exp(b*x)", var="x", true={"a": 1.0, "b": 1.2},
        clean=lambda x: 1.0 * np.exp(1.2 * x),
        f=lambda x, a, b: a * np.exp(b * x), p0=[1.0, 1.0],
        x=lambda n: np.linspace(0, 1.5, n)),
    "transcendental": dict(
        expr="a*atan(w*x)", var="x", true={"a": 2.0, "w": 3.0},
        clean=lambda x: 2.0 * np.arctan(3.0 * x),
        f=lambda x, a, w: a * np.arctan(w * x), p0=[1.0, 1.0],
        x=lambda n: np.linspace(0, 1.5, n)),
    "sine": dict(
        expr="A*sin(w*x)", var="x", true={"A": 2.0, "w": 1.5},
        clean=lambda x: 2.0 * np.sin(1.5 * x),
        f=lambda x, A, w: A * np.sin(w * x), p0=[1.5, 1.3],
        x=lambda n: np.linspace(0, 4 * np.pi, n)),
    "mixed": dict(
        expr="a0 + a1*x + a2*exp(a3*x)", var="x",
        true={"a0": 0.5, "a1": 0.2, "a2": 0.3, "a3": 0.4},
        clean=lambda x: 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x),
        f=lambda x, a0, a1, a2, a3: a0 + a1 * x + a2 * np.exp(a3 * x),
        p0=[1.0, 1.0, 1.0, 1.0], x=lambda n: np.linspace(0, 3, n)),
}


def _noisy(fam, n, noise, seed, outlier_frac=0.0):
    rng = np.random.default_rng(seed)
    x = fam["x"](n)
    clean = fam["clean"](x)
    y = clean + rng.normal(0, noise * clean.std(), n)
    if outlier_frac > 0:
        k = max(1, int(outlier_frac * n))
        idx = rng.choice(n, k, replace=False)
        y[idx] += rng.choice([-1, 1], k) * 6 * clean.std()
    return x, y, clean


def r2_clean(clean, pred):
    return metrics(clean, pred)["R2"]


def noise_sweep(fam, noises, n=120, seeds=4):
    """R2-vs-clean for each method across noise levels (averaged over seeds)."""
    out = {m: [] for m in ["EAC", "LSI", "curve_fit", "polyfit", "MLP"]}
    for noise in noises:
        acc = {m: [] for m in out}
        for s in range(seeds):
            x, y, clean = _noisy(fam, n, noise, s)
            try:
                acc["EAC"].append(r2_clean(clean, np.asarray(
                    dt.fit_eac(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["EAC"].append(np.nan)
            try:
                acc["LSI"].append(r2_clean(clean, np.asarray(
                    dt.fit_lsi(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["LSI"].append(np.nan)
            try:
                p = bl.scipy_curve_fit(x, y, fam["f"], fam["p0"])
                acc["curve_fit"].append(r2_clean(clean, fam["f"](x, *p)))
            except Exception:
                acc["curve_fit"].append(np.nan)
            try:
                acc["polyfit"].append(r2_clean(clean, bl.polyfit_predict(x, y, x, deg=5)))
            except Exception:
                acc["polyfit"].append(np.nan)
            try:
                acc["MLP"].append(r2_clean(clean, bl.mlp_curve(x, y, x, max_iter=800)))
            except Exception:
                acc["MLP"].append(np.nan)
        for m in out:
            out[m].append(np.nanmean(acc[m]) if any(np.isfinite(acc[m])) else np.nan)
    return out


def outlier_sweep(fam, fracs, n=120, seeds=5):
    """R2-vs-clean under outliers: stock EAC vs the robust ensemble (#3)."""
    methods = ["EAC", "LSI", "curve_fit", "EAC-ensemble", "EAC-softl1"]
    out = {m: [] for m in methods}
    for fr in fracs:
        acc = {m: [] for m in methods}
        for s in range(seeds):
            x, y, clean = _noisy(fam, n, 0.05, s, outlier_frac=fr)
            try:
                acc["EAC"].append(r2_clean(clean, np.asarray(
                    dt.fit_eac(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["EAC"].append(np.nan)
            try:
                acc["LSI"].append(r2_clean(clean, np.asarray(
                    dt.fit_lsi(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["LSI"].append(np.nan)
            try:
                p = bl.scipy_curve_fit(x, y, fam["f"], fam["p0"])
                acc["curve_fit"].append(r2_clean(clean, fam["f"](x, *p)))
            except Exception:
                acc["curve_fit"].append(np.nan)
            try:
                e = ensemble_fit(x, y, fam["expr"], fam["var"], method="eac",
                                 n_windows=10, overlap=0.5, p0=fam["p0"])
                acc["EAC-ensemble"].append(r2_clean(clean, e.predict(x)))
            except Exception:
                acc["EAC-ensemble"].append(np.nan)
            try:
                r = dt.fit_eac(x, y, fam["expr"], fam["var"], p0=fam["p0"],
                               loss="soft_l1",
                               bounds=[(-10, 10)] * len(fam["p0"]))
                acc["EAC-softl1"].append(r2_clean(clean, np.asarray(r.model(x))))
            except Exception:
                acc["EAC-softl1"].append(np.nan)
        for m in methods:
            out[m].append(np.nanmean(acc[m]) if any(np.isfinite(acc[m])) else np.nan)
    return out


def param_grid_parallel(fam, noises, sizes, seeds=3):
    """EAC param-recovery error over a (noise x size) grid, fanned via fit_many."""
    names = list(fam["true"])
    tv = np.array([fam["true"][k] for k in names])
    grid = np.full((len(noises), len(sizes)), np.nan)
    probs, coord = [], []
    for i, noise in enumerate(noises):
        for j, n in enumerate(sizes):
            for s in range(seeds):
                x, y, _ = _noisy(fam, n, noise, s)
                probs.append(FittingProblem(x=x, y=y, expr=fam["expr"],
                                        var=fam["var"], method="eac",
                                        kwargs={"p0": fam["p0"]}))
                coord.append((i, j))
    results = fit_many(probs, n_jobs=-1, backend="loky")
    bucket: dict[tuple, list] = {}
    for (i, j), r in zip(coord, results):
        if r.error is None and r.coeffs.size == len(names):
            err = float(np.mean(np.abs((r.coeffs - tv) / tv)) * 100)
            bucket.setdefault((i, j), []).append(err)
    for (i, j), errs in bucket.items():
        grid[i, j] = float(np.median(errs))
    return grid
