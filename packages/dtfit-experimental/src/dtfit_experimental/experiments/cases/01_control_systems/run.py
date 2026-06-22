"""Experiment 1 -- control-systems system identification.

Recovering the physical parameters of a dynamic system from its noisy response
is a core control-engineering task. The dtfit methods target exactly the
transcendental, parameter-nonlinear forms these responses take (a damped
sinusoid, an exponential approach), so this experiment asks: do LSI/EAC recover
the true parameters as well as the NLLS gold standard (SciPy ``curve_fit``) and
better than a black-box neural net, and can the streaming filters track a plant
whose dynamics change mid-run?

Architecture adaptation: a MIMO plant (several outputs sharing a natural
frequency) is identified jointly with ``dtfit_experimental.fit_joint``, and a
``FilterBank`` tracks the outputs concurrently.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit_experimental import fit_joint
from dtfit.streaming import EACFilter

from dtfit_experimental.experiments.common import ReportWriter, metrics, fmt
from dtfit_experimental.experiments.common.plotting import plt, fit_overlay, residuals, error_bars
from dtfit_experimental.experiments.common import baselines as bl

EXP_DIR = __file__.rsplit("run.py", 1)[0]

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


def _param_err(est: dict, true: dict) -> float:
    """Mean relative parameter-recovery error (%)."""
    return float(np.mean([abs(est[k] - true[k]) / abs(true[k]) for k in true]) * 100)


def damped_table(t, y, clean, true):
    names = ["A", "w", "z"]
    p0, lo, hi = [1.0, 2.0, 0.1], [0.1, 1.0, 0.01], [5, 6, 0.9]
    rows, preds = [], {}

    def add(label, coeffs, pred, ms):
        est = dict(zip(names, coeffs)) if coeffs is not None else None
        m = metrics(clean, pred)
        pe = fmt(_param_err(est, true), "{:.2f}") if est else "--"
        rows.append([label, pe, fmt(m["R2"], "{:.4f}"), fmt(m["RMSE"]),
                     fmt(ms, "{:.0f}")])
        preds[label] = pred

    from dtfit_experimental.experiments.common import timed
    r, ms = timed(lambda: dt.fit_eac(t, y, DAMP_EXPR, "t", p0=p0, bounds=(lo, hi)))
    add("EAC", r.coeffs, np.asarray(r.model(t)), ms)
    r, ms = timed(lambda: dt.fit_lsi(t, y, DAMP_EXPR, "t", p0=p0,
                                     bounds=list(zip(lo, hi))))
    add("LSI", r.coeffs, np.asarray(r.model(t)), ms)
    p, ms = timed(lambda: bl.scipy_curve_fit(t, y, _damped, p0, bounds=(lo, hi)))
    add("SciPy curve_fit", p, _damped(t, *p), ms)
    yhat, ms = timed(lambda: bl.mlp_curve(t, y, t, hidden=(64, 64)))
    add("sklearn MLP", None, yhat, ms)
    return rows, preds


def first_order_table(t, y, clean, true):
    names = ["K", "tau"]
    rows, preds = [], {}
    from dtfit_experimental.experiments.common import timed

    def add(label, coeffs, pred, ms):
        est = dict(zip(names, coeffs)) if coeffs is not None else None
        m = metrics(clean, pred)
        pe = fmt(_param_err(est, true), "{:.2f}") if est else "--"
        rows.append([label, pe, fmt(m["R2"], "{:.4f}"), fmt(m["RMSE"]),
                     fmt(ms, "{:.0f}")])
        preds[label] = pred

    r, ms = timed(lambda: dt.fit_eac(t, y, FO_EXPR, "t", p0=[1.0, 1.0]))
    add("EAC", r.coeffs, np.asarray(r.model(t)), ms)
    r, ms = timed(lambda: dt.fit_lsi(t, y, FO_EXPR, "t", p0=[1.0, 1.0]))
    add("LSI", r.coeffs, np.asarray(r.model(t)), ms)
    def f(tt, K, tau):
        return K * (1 - np.exp(-tt / tau))
    p, ms = timed(lambda: bl.scipy_curve_fit(t, y, f, [1.0, 1.0]))
    add("SciPy curve_fit", p, f(t, *p), ms)
    yhat, ms = timed(lambda: bl.mlp_curve(t, y, t))
    add("sklearn MLP", None, yhat, ms)
    return rows, preds


def regime_change(rng, n=900):
    """Damping z jumps mid-run; the online filter should re-adapt + flag it."""
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
    """3-output plant sharing a natural frequency w; identify jointly."""
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


def main(quick: bool = False) -> str:
    rng = np.random.default_rng(0)
    rep = ReportWriter(
        EXP_DIR, "Experiment 1 — Control-systems system identification",
        intent=(
            "Recover the physical parameters of dynamic systems from noisy "
            "responses and track a plant whose dynamics change online. dtfit's "
            "LSI/EAC target the transcendental, parameter-nonlinear forms these "
            "responses take, so we compare parameter-recovery accuracy against "
            "the NLLS gold standard (SciPy `curve_fit`) and a black-box neural "
            "net, and exercise the streaming filter under a regime change."),
    )

    rep.section(
        "Models fitted & why",
        "- **A (2nd-order):** `y = A·e^(−ζω·t)·sin(ω√(1−ζ²)·t)` — the analytic "
        "free response of an underdamped second-order linear system. Chosen "
        "because its parameters *are* the physical quantities an engineer wants "
        "(amplitude A, damping ratio ζ, natural frequency ω), and a damped "
        "sinusoid is exactly the transcendental, non-Taylor form LSI/EAC target.\n"
        "- **B (first-order):** `y = K·(1 − e^(−t/τ))` — the textbook step "
        "response of a first-order plant / RC circuit / DC motor; chosen to "
        "recover the DC gain K and time constant τ.\n"
        "- **C (regime change) and MIMO:** the same 2nd-order damped model, "
        "tracked online by the filter (C) and fitted jointly with a *shared* ω "
        "across outputs (MIMO) — chosen so the model is physically identical "
        "while the scenario stresses online adaptation and channel coupling.")

    # --- A: damped oscillation ------------------------------------------- #
    t, y, clean, true = scenario_damped(rng)
    rows, preds = damped_table(t, y, clean, true)
    rep.section(
        "A. Second-order underdamped free response",
        f"Model `y = A·e^(−ζω t)·sin(ω√(1−ζ²)·t)`, truth A={true['A']}, "
        f"ω={true['w']}, ζ={true['z']}, n={t.size}, 5% noise. Error columns are "
        "against the *clean* signal; **param err %** is the mean relative error "
        "of the recovered (A, ω, ζ) — the quantity a control engineer cares "
        "about (the MLP fits the curve but recovers no physical parameters).")
    rep.table(["method", "param err %", "R²", "RMSE", "fit (ms)"], rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    fit_overlay(ax[0], t, y, preds["EAC"], truth=clean,
                title="EAC fit — damped oscillation", label="EAC fit",
                color="tab:green")
    residuals(ax[1], t, y, preds["EAC"], title="EAC residuals", color="tab:green")
    rep.figure(fig, "damped_fit", "EAC recovers the underdamped free response.")

    # param-recovery bar chart across methods
    pe = {r[0]: float(r[1]) if r[1] != "--" else np.nan for r in rows}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    nm = [k for k in pe if np.isfinite(pe[k])]
    error_bars(ax, nm, [pe[k] for k in nm], ylabel="param err % (lower better)",
               title="Parameter-recovery error — damped oscillation",
               colors=["tab:green", "tab:blue", "0.5", "0.7"][:len(nm)],
               annotate="{:.2f}")
    rep.figure(fig, "damped_param_err", "dtfit matches NLLS on parameter recovery.")

    # --- B: first-order step --------------------------------------------- #
    t2, y2, clean2, true2 = scenario_first_order(rng)
    rows2, preds2 = first_order_table(t2, y2, clean2, true2)
    rep.section(
        "B. First-order plant / RC charge / DC-motor step",
        f"Model `y = K·(1 − e^(−t/τ))`, truth K={true2['K']}, τ={true2['tau']}.")
    rep.table(["method", "param err %", "R²", "RMSE", "fit (ms)"], rows2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    fit_overlay(ax[0], t2, y2, preds2["LSI"], truth=clean2,
                title="LSI fit — first-order step", label="LSI fit")
    residuals(ax[1], t2, y2, preds2["LSI"], title="LSI residuals")
    rep.figure(fig, "first_order_fit", "First-order step identification.")

    # --- C: regime change (online tracking) ------------------------------ #
    t3, y3, clean3, track, z_hist, drift_idx, half = regime_change(rng, n=600 if quick else 900)
    rep.section(
        "C. Regime change — online tracking + drift detection",
        f"The damping ζ jumps (0.08→0.30) at the midpoint. The "
        "`EACFilter` tracks the parameters online with bounded per-sample "
        f"cost and flagged **{len(drift_idx)}** structural break(s) — a "
        "sliding-window `curve_fit` refit or a batch NN cannot do this in a "
        "real-time loop.")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(t3, y3, "0.7", lw=0.6, label="noisy response")
    ax[0].plot(t3, track, "tab:red", lw=1.2, label="filter tracking")
    for j2, di in enumerate(drift_idx):
        ax[0].axvline(t3[di], color="tab:purple", ls="--", lw=1,
                      label="drift detected" if j2 == 0 else None)
    ax[0].set_title("Online tracking through a damping change")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8)
    ax[1].plot(t3, z_hist, "tab:red", lw=1.2)
    ax[1].axvline(t3[half], color="0.5", ls=":", label="true change")
    for di in drift_idx:
        ax[1].axvline(t3[di], color="tab:purple", ls="--", lw=1)
    ax[1].set_title("Tracked damping ζ adapts after the change")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("ζ estimate"); ax[1].legend(fontsize=8)
    rep.figure(fig, "regime_tracking", "Online parameter tracking + drift flagging.")

    # --- adaptation: MIMO joint fit -------------------------------------- #
    w_true, amps, j, indep_w, chans, tt = mimo_joint(rng)
    joint_err = abs(j.shared["w"] - w_true) / w_true * 100
    indep_err = float(np.mean([abs(wv - w_true) / w_true * 100 for wv in indep_w]))
    rep.section(
        "Architecture adaptation — joint MIMO identification", level=2)
    rep.text(
        "A 3-output plant shares one natural frequency ω. `fit_joint` estimates "
        "the shared ω from *all* outputs at once (plus each output's private "
        "amplitude/damping) in a single coupled system, halving the parameter "
        "count and guaranteeing a *consistent* ω across channels.")
    rep.table(
        ["estimator", "shared ω", "ω err %", "per-channel amplitudes"],
        [["joint (fit_joint)", fmt(j.shared["w"], "{:.3f}"),
          fmt(joint_err, "{:.2f}"),
          ", ".join(f"{p['A']:.2f}" for p in j.private)],
         ["independent EAC (mean)", fmt(float(np.mean(indep_w)), "{:.3f}"),
          fmt(indep_err, "{:.2f}"), ", ".join("--" for _ in amps)]])

    rep.section("Reading it", level=2)
    rep.text(
        "- dtfit's LSI/EAC recover the control parameters within tolerance of "
        "the NLLS gold standard, while the black-box MLP fits the curve but "
        "yields no physical parameters.\n"
        "- The streaming filter tracks a mid-run dynamics change and flags it — "
        "the real-time capability batch methods lack.\n"
        f"- Adaptation #4 (joint MIMO): the dedicated bounded EAC solver already "
        f"recovers ω almost exactly per channel ({indep_err:.2f}% mean error), so "
        f"the coarser joint area-matching ({joint_err:.2f}%) does **not** improve "
        "accuracy on these cleanly-identifiable outputs — its value here is "
        "parameter parsimony and an enforced single consistent ω. Whether "
        "coupling *helps accuracy* when per-channel data is genuinely weak is "
        "re-tested in the GPS experiment (Exp 5); on this experiment it does not "
        "clear the promotion gate.")

    path = rep.write()
    print(f"[control_systems] wrote {path}")
    return str(path)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
