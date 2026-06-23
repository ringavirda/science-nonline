"""Backend infrastructure for the control-systems system-identification experiment.

This module is the **single source of truth for the simulation and estimation
code** behind ``01_control_systems.ipynb``; the notebook imports it and does all
the presentation (tables, figures, narrative). Keeping the infra here means the
scenarios/estimators/baselines are defined once and the notebook stays a thin,
rerunnable layer over them.

It provides, for the control-engineering task of recovering the physical
parameters of a dynamic system from its noisy response:

* the **plant scenarios** -- :func:`scenario_damped` (underdamped second-order
  free response) and :func:`scenario_first_order` (first-order step), each
  returning ``(t, y, clean, true-params)``;
* the **identification benches** -- :func:`damped_table` and
  :func:`first_order_table` that run dtfit's EAC/LSI against the NLLS gold
  standard (SciPy ``curve_fit``) and a black-box neural net (sklearn MLP),
  returning recovered params / metrics / timings as plain dicts;
* the **online / coupled adaptations** -- :func:`regime_change` (streaming
  :class:`EACFilter` tracking a mid-run damping jump with drift detection) and
  :func:`mimo_joint` (a shared-frequency MIMO plant identified with
  ``fit_joint``);
* the **scoring helper** :func:`param_err` (mean relative parameter-recovery
  error) and the model exprs :data:`DAMP_EXPR` / :data:`FO_EXPR`.

The optional baselines (SciPy ``curve_fit``, sklearn ``MLPRegressor``) are
imported lazily inside :mod:`...common.baselines`; the notebook guards their use
so a missing dependency only skips that row.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit_experimental import fit_joint
from dtfit.streaming import EACFilter

from dtfit_experimental.experiments.common import metrics, timed
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "DAMP_EXPR", "FO_EXPR",
    "scenario_damped", "scenario_first_order", "param_err",
    "damped_table", "first_order_table", "regime_change", "mimo_joint",
]

# underdamped free response: y = A e^{-zwt} sin(w sqrt(1-z^2) t); params A,w,z
DAMP_EXPR = "A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)"
# first-order step: y = K(1 - e^{-t/tau}); params K, tau
FO_EXPR = "K*(1-exp(-t/tau))"


def _damped(t, A, w, z):
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t)


def scenario_damped(rng, n=240, noise=0.05):
    A, z, w = 2.0, 0.15, 3.0
    t = np.linspace(0, 6, n)
    clean = _damped(t, A, w, z)
    y = clean + rng.normal(0, noise * clean.std(), n)
    return t, y, clean, {"A": A, "w": w, "z": z}


def scenario_first_order(rng, n=200, noise=0.03):
    K, tau = 3.0, 1.2
    t = np.linspace(0, 6, n)
    clean = K * (1 - np.exp(-t / tau))
    y = clean + rng.normal(0, noise * clean.std(), n)
    return t, y, clean, {"K": K, "tau": tau}


def param_err(est: dict, true: dict) -> float:
    """Mean relative parameter-recovery error (%)."""
    return float(np.mean([abs(est[k] - true[k]) / abs(true[k]) for k in true]) * 100)


def damped_table(t, y, clean, true, *, with_scipy=True, with_mlp=True):
    """Identify the damped second-order response with EAC, LSI, and (optionally)
    the SciPy ``curve_fit`` NLLS gold standard and an sklearn MLP black-box.

    Returns ``(rows, preds)`` where ``rows`` is a list of dicts
    (method / param-err-% / R2 / RMSE / fit-ms; param-err is ``None`` for the MLP
    which recovers no physical parameters) and ``preds`` maps method -> fitted
    curve. Set ``with_scipy`` / ``with_mlp`` to ``False`` to skip a baseline whose
    optional dependency is unavailable."""
    names = ["A", "w", "z"]
    p0, lo, hi = [1.0, 2.0, 0.1], [0.1, 1.0, 0.01], [5, 6, 0.9]
    rows, preds = [], {}

    def add(label, coeffs, pred, ms):
        est = dict(zip(names, coeffs)) if coeffs is not None else None
        m = metrics(clean, pred)
        rows.append({"method": label,
                     "param err %": param_err(est, true) if est else None,
                     "R2": m["R2"], "RMSE": m["RMSE"], "fit (ms)": ms})
        preds[label] = pred

    r, ms = timed(lambda: dt.fit_eac(t, y, DAMP_EXPR, "t", p0=p0, bounds=(lo, hi)))
    add("EAC", r.coeffs, np.asarray(r.model(t)), ms)
    r, ms = timed(lambda: dt.fit_lsi(t, y, DAMP_EXPR, "t", p0=p0,
                                     bounds=list(zip(lo, hi))))
    add("LSI", r.coeffs, np.asarray(r.model(t)), ms)
    if with_scipy:
        p, ms = timed(lambda: bl.scipy_curve_fit(t, y, _damped, p0, bounds=(lo, hi)))
        add("SciPy curve_fit", p, _damped(t, *p), ms)
    if with_mlp:
        yhat, ms = timed(lambda: bl.mlp_curve(t, y, t, hidden=(64, 64)))
        add("sklearn MLP", None, yhat, ms)
    return rows, preds


def first_order_table(t, y, clean, true, *, with_scipy=True, with_mlp=True):
    """Identify the first-order step response (DC gain ``K``, time constant
    ``tau``). Same return shape and skip flags as :func:`damped_table`."""
    names = ["K", "tau"]
    rows, preds = [], {}

    def add(label, coeffs, pred, ms):
        est = dict(zip(names, coeffs)) if coeffs is not None else None
        m = metrics(clean, pred)
        rows.append({"method": label,
                     "param err %": param_err(est, true) if est else None,
                     "R2": m["R2"], "RMSE": m["RMSE"], "fit (ms)": ms})
        preds[label] = pred

    r, ms = timed(lambda: dt.fit_eac(t, y, FO_EXPR, "t", p0=[1.0, 1.0]))
    add("EAC", r.coeffs, np.asarray(r.model(t)), ms)
    r, ms = timed(lambda: dt.fit_lsi(t, y, FO_EXPR, "t", p0=[1.0, 1.0]))
    add("LSI", r.coeffs, np.asarray(r.model(t)), ms)

    def f(tt, K, tau):
        return K * (1 - np.exp(-tt / tau))

    if with_scipy:
        p, ms = timed(lambda: bl.scipy_curve_fit(t, y, f, [1.0, 1.0]))
        add("SciPy curve_fit", p, f(t, *p), ms)
    if with_mlp:
        yhat, ms = timed(lambda: bl.mlp_curve(t, y, t))
        add("sklearn MLP", None, yhat, ms)
    return rows, preds


def regime_change(rng, n=900):
    """Damping z jumps mid-run; the online filter should re-adapt + flag it.

    Returns ``(t, y, clean, track, z_hist, drift_idx, half)`` -- the time grid,
    noisy response, clean signal, the filter's online track, its tracked damping
    estimate, the sample indices it flagged as structural breaks, and the true
    change index."""
    t = np.linspace(0, 18, n)
    half = n // 2
    z1, z2, A, w = 0.08, 0.30, 2.0, 2.5
    z_arr = np.where(np.arange(n) < half, z1, z2)
    wd = w * np.sqrt(1 - z_arr ** 2)
    # phase-continuous across the change: integrate the (piecewise) frequency
    dtt = np.diff(t, prepend=t[0])
    phase = np.cumsum(wd * dtt)
    clean = A * np.exp(-z_arr * w * t) * np.sin(phase)
    y = clean + rng.normal(0, 0.05, n)
    flt = EACFilter(DAMP_EXPR, "t", p0=[2.0, 2.5, 0.1],
                    window_size=60, q_diag=[1e-3, 1e-3, 1e-3], r=0.5,
                    n_sub=2, adapt_r=True)
    track, z_hist, drift_idx = [], [], []
    for i in range(n):
        flt.partial_fit(t[i], y[i])
        if flt.drift_flag_:
            drift_idx.append(i)
        track.append(float(flt.predict(np.array([t[i]]))[0]) if len(flt._t) else np.nan)
        z_hist.append(flt.params_["z"])
    return t, y, clean, np.array(track), np.array(z_hist), drift_idx, half


def mimo_joint(rng, n=200):
    """3-output plant sharing a natural frequency w; identify jointly.

    Returns ``(w_true, amps, j, indep_w, chans, t)`` -- the true shared frequency,
    the per-channel amplitudes, the :func:`fit_joint` result ``j``, the per-channel
    independent-EAC frequency estimates, the channels, and the time grid."""
    t = np.linspace(0, 6, n)
    w_true, z_true = 3.0, 0.12
    amps = [1.0, 2.0, 3.0]
    chans = [(t, _damped(t, A, w_true, z_true) + rng.normal(0, 0.04, n))
             for A in amps]
    j = fit_joint(chans, DAMP_EXPR, "t", shared=["w"], n_windows=6,
                  p0_shared=[2.5], p0_private=[1.0, 0.1])
    # independent per-channel EAC for contrast
    indep_w = []
    for (tx, yx) in chans:
        r = dt.fit_eac(tx, yx, DAMP_EXPR, "t", p0=[1.0, 2.5, 0.1],
                       bounds=([0.1, 1, 0.01], [5, 6, 0.9]))
        indep_w.append(dict(zip(["A", "w", "z"], r.coeffs))["w"])
    return w_true, amps, j, indep_w, chans, t
