"""Shared scoring + fitting helpers for the accuracy corpus.

One implementation of "fit this scenario, score the recovery / curve quality,
and what would the gold-standard NLLS baseline do" -- used by the Phase-1 gate,
the Phase-3 seed-robustness tests, the Phase-4 golden snapshot and the
``dtfit-experimental`` exploration harness, so every consumer judges a fit the
same way.
"""

from __future__ import annotations

import warnings

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from .scenarios import Scenario


def ordered_params(scn: Scenario) -> list[str]:
    """Model parameter names in the sorted order the fitters lay coeffs out in."""
    m = scn.model()
    t = sp.Symbol(m.var)
    f = sp.sympify(m.expr)
    return [str(s) for s in sorted((s for s in f.free_symbols if s != t), key=str)]


def r2(clean: np.ndarray, pred: np.ndarray) -> float:
    """R^2 of ``pred`` against the *clean* signal (true-recovery curve quality)."""
    pred = np.asarray(pred, float)
    if pred.ndim == 0:
        pred = np.full_like(clean, float(pred))
    if not np.all(np.isfinite(pred)):
        return -np.inf
    ss_res = float(np.sum((pred - clean) ** 2))
    ss_tot = float(np.sum((clean - clean.mean()) ** 2)) or 1e-30
    return 1.0 - ss_res / ss_tot


def param_err(scn: Scenario, names: list[str], est: np.ndarray) -> float:
    """Max over parameters of the relative recovery error |est - true|/|true|."""
    errs = []
    for nm, v in zip(names, np.asarray(est, float)):
        tv = scn.true[nm]
        errs.append(abs(float(v) - tv) / (abs(tv) + 1e-9))
    return float(max(errs)) if errs else float("nan")


def predict(res, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(res.model(x), float)
    return np.full_like(x, float(pred)) if pred.ndim == 0 else pred


def metrics_for(scn: Scenario, noise: float, seed: int = 0) -> dict:
    """Recovery metrics for the realistic self-seeded ``Model.fit`` path -- the
    single measurement the golden baseline snapshots and the regression guard
    re-checks. Returns ``{"perr", "r2", "metric"}`` (NaN/inf folded to a large
    finite sentinel so the JSON round-trips)."""
    import warnings

    names = ordered_params(scn)
    x, y, clean = scn.make(noise, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scn.model().fit(x, y)
    perr = param_err(scn, names, np.asarray(res.coeffs, float))
    got_r2 = r2(clean, predict(res, x))
    return {
        "metric": scn.metric,
        "perr": float(perr) if np.isfinite(perr) else 1e9,
        "r2": float(got_r2) if np.isfinite(got_r2) else -1e9,
    }


def curve_fit_baseline(scn: Scenario, x, y, names):
    """``(popt, pred)`` from ``scipy.curve_fit`` with the model's data-driven
    seed -- the Levenberg-Marquardt gold standard the methods are measured
    against. Returns ``(None, reason)`` if it fails."""
    m = scn.model()
    t = sp.Symbol(m.var)
    f = sp.sympify(m.expr)
    syms = [sp.Symbol(n) for n in names]
    fn = sp.lambdify((t, *syms), f, "numpy")
    p0, bounds = m._seed_arrays(x, y)
    if p0 is None:
        p0 = [1.0] * len(names)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bounds is not None:
                lo = [b[0] for b in bounds]
                hi = [b[1] for b in bounds]
                popt, _ = curve_fit(fn, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
            else:
                popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
        pred = np.asarray(fn(x, *popt), dtype=float)
        return np.asarray(popt, float), pred
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
