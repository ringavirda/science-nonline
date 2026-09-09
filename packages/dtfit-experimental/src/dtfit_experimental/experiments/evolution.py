"""The evolution matrix: every stage of the method on one problem set.

Rows are the stages the estimator grew through, from the source-form
criteria in ``dtfit_legacy.book`` to the image in ``dtfit.fit``; columns are
model families crossed with noise, grid shape, outlier fraction and sample
size. Every stage starts from the same perturbed truth, so the matrix
measures the estimators, not the seeding. A second, smaller matrix compares
the weak form of ODE identification (no ODE solve, no start) with nonlinear
least squares on the integrated law.

Results go to CSV as they are produced, so a partial run is still usable::

    python -m dtfit_experimental.experiments.evolution --out DIR [--seeds N]
    python -m dtfit_experimental.experiments.evolution --out DIR --full
    python -m dtfit_experimental.experiments.evolution --out DIR --report

``--smoke`` runs one seed of two families to check the wiring; ``--full``
is the spec's problem set (eight families, four outlier fractions, three
sample sizes, thirty seeds); ``--report`` reads the CSVs and writes
``summary.csv``, ``summary.md`` and the figure.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time
import warnings

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from dtfit import Original, fit
from dtfit_legacy import (
    dsb_balance, eac_areas, find_degree, fit_dsb, lsi_integral_monomial,
)
from dtfit_legacy.integral import fit_eac, fit_lsi

FAMILIES = {
    "exp_decay": ("a*exp(b*t)", {"a": 3.0, "b": -1.2}, 2.0),
    "logistic": ("L/(1 + exp(-k*(t - c)))", {"L": 5.0, "c": 3.0, "k": 1.4},
                 8.0),
    "gauss_peak": ("a*exp(-(t - m)**2/(2*s**2))",
                   {"a": 2.0, "m": 2.5, "s": 0.6}, 5.0),
    "damped_sine": ("a*exp(-d*t)*sin(w*t)", {"a": 2.0, "d": 0.3, "w": 3.0},
                    6.0),
    "arctan": ("a*atan(w*t)", {"a": 1.0, "w": 2.0}, 3.0),
    "michaelis_menten": ("Vm*t/(Km + t)", {"Km": 0.8, "Vm": 1.2}, 8.0),
    "power_law": ("a*t**b", {"a": 2.0, "b": 0.7}, 5.0),
    "two_exp": ("a*exp(b*t) + c*exp(d*t)",
                {"a": 2.0, "b": -0.5, "c": 1.0, "d": -3.0}, 6.0),
}
FIRST_CUT = ("exp_decay", "logistic", "gauss_peak", "damped_sine", "arctan")
NOISES = (0.01, 0.05, 0.10, 0.30)
GRIDS = ("uniform", "clustered", "random")
OUTLIERS = (0.0, 0.10)
SIZES = (200,)
P0_FACTOR = 1.25


def _truth_vector(expr, truth):
    t = sp.Symbol("t")
    names = [str(s) for s in sorted(sp.sympify(expr).free_symbols - {t},
                                    key=str)]
    return names, np.array([truth[n] for n in names])


def _model(expr, names):
    t = sp.Symbol("t")
    f = sp.lambdify([t, *[sp.Symbol(n) for n in names]], sp.sympify(expr),
                    "numpy")
    return lambda x, *p: np.asarray(f(x, *p), dtype=float) + 0.0 * x


def _grid(kind, h, rng, n):
    if kind == "uniform":
        return np.linspace(0.0, h, n)
    if kind == "clustered":
        a = rng.uniform(0.0, 0.2 * h, n // 2)
        b = rng.uniform(0.2 * h, h, n - n // 2)
        x = np.sort(np.concatenate([a, b]))
    else:
        x = np.sort(rng.uniform(0.0, h, n))
    x[0], x[-1] = 0.0, h
    return x


def _data(model, truth, h, noise, grid, outliers, n, seed):
    rng = np.random.default_rng(seed)
    x = _grid(grid, h, rng, n)
    clean = model(x, *truth)
    amp = float(np.ptp(clean)) or 1.0
    sigma = noise * amp
    y = clean + sigma * rng.standard_normal(n)
    if outliers > 0:
        k = max(1, int(round(outliers * n)))
        idx = rng.choice(n, k, replace=False)
        y[idx] = clean[idx] + 10.0 * sigma * rng.standard_normal(k)
    return x, y


def _stages(model):
    """``(name, needs_p0, run)`` with ``run(x, y, expr, p0) -> (coeffs, cov)``;
    ``cov`` is ``None`` where the stage has none."""
    def s0(x, y, expr, p0):
        c, cov = curve_fit(model, x, y, p0=p0, maxfev=4000)
        return c, cov

    def s2(x, y, expr, p0):
        deg = find_degree(x, y)
        coeffs = np.polynomial.polynomial.polyfit(x, y, deg)
        r = fit_dsb(coeffs, expr, "t", p0=p0)
        return r.coeffs, r.cov

    def res(r):
        return r.coeffs, r.cov

    return [
        ("S0_nlls", True, s0),
        ("S1_balance_book", False,
         lambda x, y, e, p0: res(dsb_balance(x, y, e, "t", p0=p0))),
        ("S2_balance_adequate", False, s2),
        ("S3a_lsi_monomial", True,
         lambda x, y, e, p0: res(lsi_integral_monomial(x, y, e, "t", p0=p0))),
        ("S3b_lsi_legendre", True,
         lambda x, y, e, p0: res(fit_lsi(x, y, e, "t", p0=p0))),
        ("S3c_eac_book", True,
         lambda x, y, e, p0: res(eac_areas(x, y, e, "t", p0=p0))),
        ("S3d_eac_direct", True,
         lambda x, y, e, p0: res(fit_eac(x, y, e, "t", p0=p0))),
        ("S4a_image_legendre", True,
         lambda x, y, e, p0: res(fit(e, Original(x, y), "t",
                                     basis="legendre", p0=p0))),
        ("S4b_image_block", True,
         lambda x, y, e, p0: res(fit(e, Original(x, y), "t", basis="block",
                                     p0=p0))),
        ("S4r_image_robust", True,
         lambda x, y, e, p0: res(fit(e, Original(x, y), "t",
                                     basis="legendre", robust=True, p0=p0))),
    ]


def _cov95(c, cov, tv):
    if cov is None:
        return ""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (tv.size, tv.size) or not np.all(np.isfinite(cov)):
        return ""
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return bool(np.all(np.abs(c - tv) <= 1.96 * se))


def batch_matrix(out: pathlib.Path, seeds: int, families) -> None:
    path = out / "batch.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["stage", "needs_p0", "family", "noise", "grid",
                        "outliers", "n", "seed", "err", "signed", "cov95",
                        "ok", "time", "error"])
        for fam in families:
            expr, truth, h = FAMILIES[fam]
            names, tv = _truth_vector(expr, truth)
            model = _model(expr, names)
            p0 = tv * P0_FACTOR
            stages = _stages(model)
            t0 = time.time()
            for n in SIZES:
                for noise in NOISES:
                    for grid in GRIDS:
                        for outl in OUTLIERS:
                            for seed in range(seeds):
                                x, y = _data(model, tv, h, noise, grid, outl,
                                             n, seed)
                                for name, needs, run in stages:
                                    tic = time.perf_counter()
                                    try:
                                        c, cov = run(x, y, expr, p0)
                                        c = np.asarray(c, dtype=float)
                                        rel = c / tv - 1.0
                                        err = float(np.median(np.abs(rel)))
                                        signed = float(np.mean(rel))
                                        ok = bool(np.isfinite(err))
                                        c95 = _cov95(c, cov, tv)
                                        msg = ""
                                    except Exception as exc:  # noqa: BLE001
                                        err = signed = float("nan")
                                        ok, c95, msg = False, "", \
                                            type(exc).__name__
                                    w.writerow([name, needs, fam, noise,
                                                grid, outl, n, seed, err,
                                                signed, c95, ok,
                                                time.perf_counter() - tic,
                                                msg])
                                fh.flush()
            print(f"{fam}: done in {time.time() - t0:.0f} s", flush=True)


def ode_matrix(out: pathlib.Path, seeds: int) -> None:
    from scipy.integrate import solve_ivp

    from dtfit_experimental.weak_ode import (
        fit_damped_oscillator, fit_logistic, fit_lotka_volterra_prey,
        fit_michaelis_menten,
    )

    def integ(rhs, y0, t, p):
        return solve_ivp(lambda tt, s: rhs(tt, s, *p), (t[0], t[-1]), y0,
                         t_eval=t, rtol=1e-8, atol=1e-10).y[0]

    laws = {
        "logistic": dict(
            rhs=lambda t, y, r, k: r * y * (1 - y / k), y0=[0.4],
            t=np.linspace(0, 8, 500), truth={"r": 1.4, "K": 5.0},
            scale=0.4, weak=lambda t, y: fit_logistic(t, y)),
        "michaelis_menten": dict(
            rhs=lambda t, y, vm, km: -vm * y / (km + y), y0=[4.0],
            t=np.linspace(0, 8, 400), truth={"Vm": 1.2, "Km": 0.8},
            scale=4.0, weak=lambda t, y: fit_michaelis_menten(t, y)),
        "damped_osc": dict(
            rhs=lambda t, s, om, z: [s[1], -2 * z * om * s[1] - om * om * s[0]],
            y0=[1.0, 0.0], t=np.linspace(0, 8, 600),
            truth={"omega": 3.0, "zeta": 0.15}, scale=1.0,
            weak=lambda t, y: fit_damped_oscillator(t, y)),
    }
    path = out / "ode.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["law", "noise", "seed", "method", "err", "time"])
        for law, d in laws.items():
            names = list(d["truth"])
            tv = np.array([d["truth"][n] for n in names])
            clean = integ(d["rhs"], d["y0"], d["t"], tv)
            for noise in (0.02, 0.05, 0.10):
                for seed in range(seeds):
                    rng = np.random.default_rng(seed)
                    y = clean + noise * d["scale"] * rng.standard_normal(
                        d["t"].size)
                    tic = time.perf_counter()
                    p = d["weak"](d["t"], y)
                    err = max(abs(p[n] / d["truth"][n] - 1) for n in names)
                    w.writerow([law, noise, seed, "weak", err,
                                time.perf_counter() - tic])
                    tic = time.perf_counter()
                    try:
                        c = curve_fit(
                            lambda t, *pp: integ(d["rhs"], d["y0"], d["t"],
                                                 pp),
                            d["t"], y, p0=tv * P0_FACTOR, maxfev=400)[0]
                        err = float(np.max(np.abs(c / tv - 1)))
                    except Exception:  # noqa: BLE001
                        err = float("nan")
                    w.writerow([law, noise, seed, "nlls", err,
                                time.perf_counter() - tic])
                    fh.flush()
            print(f"ode {law}: done", flush=True)
        # the prey-only Lotka-Volterra law has no NLLS counterpart here
        al, be, de, ga = 1.1, 0.4, 0.1, 0.4
        t = np.linspace(0, 30, 1500)
        xc = solve_ivp(lambda _t, z: [al * z[0] - be * z[0] * z[1],
                                      de * z[0] * z[1] - ga * z[1]],
                       (0, 30), [10.0, 5.0], t_eval=t, rtol=1e-10,
                       atol=1e-12).y[0]
        for noise in (0.005, 0.01, 0.02):
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                x = np.clip(xc * (1 + noise * rng.standard_normal(t.size)),
                            1e-3, None)
                tic = time.perf_counter()
                p = fit_lotka_volterra_prey(t, x)
                err = float(np.median([abs(p["alpha"] / al - 1),
                                       abs(p["gamma"] / ga - 1),
                                       abs(p["delta"] / de - 1)]))
                w.writerow(["lotka_volterra_prey", noise, seed, "weak", err,
                            time.perf_counter() - tic])
        fh.flush()


def report(out: pathlib.Path, fig_path: pathlib.Path | None) -> None:
    import pandas as pd

    b = pd.read_csv(out / "batch.csv")
    if "n" not in b.columns:
        b["n"] = 200
    b["cov95"] = pd.to_numeric(b.get("cov95"), errors="coerce")
    g = b.groupby(["stage", "needs_p0", "family", "noise", "grid",
                   "outliers", "n"])
    s = g.agg(err=("err", "median"), bias=("signed", "median"),
              ok=("ok", "mean"), cov95=("cov95", "mean"),
              time=("time", "median")).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    n_ref = 200 if 200 in set(s.n) else int(s.n.iloc[0])
    lines = ["# Evolution matrix", "",
             f"seeds per cell: {b['seed'].nunique()}, sample sizes "
             f"{sorted(set(b.n))}, p0 = truth x {P0_FACTOR} for every stage "
             "that takes one", ""]

    def table(sub, title, value="err", scale=100.0, fmt="{:.2f}"):
        if sub.empty:
            return
        piv = sub.pivot_table(index="stage", columns="family", values=value,
                              aggfunc="median")
        okp = sub.pivot_table(index="stage", columns="family", values="ok",
                              aggfunc="mean")
        cols = list(piv.columns)
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| stage | needs p0 | " + " | ".join(cols) + " |")
        lines.append("|---|---|" + "---|" * len(cols))
        for st in piv.index:
            needs = "yes" if b.loc[b.stage == st, "needs_p0"].iloc[0] else "no"
            cells = []
            for c in cols:
                e, o = piv.loc[st, c], okp.loc[st, c]
                cells.append("n/a" if not np.isfinite(e) else
                             fmt.format(scale * e) + ("" if o > 0.99 else
                                                      f" ({100 * o:.0f}% ok)"))
            lines.append(f"| {st} | {needs} | " + " | ".join(cells) + " |")
        lines.append("")

    base = s[(s.n == n_ref)]
    for noise in NOISES:
        table(base[(base.noise == noise) & (base.grid == "uniform")
                   & (base.outliers == 0.0)],
              f"median relative parameter error, %, noise {100 * noise:.0f}%, "
              f"uniform grid, no outliers, n = {n_ref}")
    for outl in sorted(set(s.outliers)):
        if outl > 0:
            table(base[(base.noise == 0.05) & (base.grid == "uniform")
                       & (base.outliers == outl)],
                  f"noise 5%, uniform grid, {100 * outl:.0f}% outliers")
    table(base[(base.noise == 0.05) & (base.grid == "random")
               & (base.outliers == 0.0)],
          "noise 5%, random grid, no outliers")
    table(base[(base.noise == 0.05) & (base.grid == "clustered")
               & (base.outliers == 0.0)],
          "noise 5%, clustered grid, no outliers")
    for n in sorted(set(s.n)):
        if n != n_ref:
            table(s[(s.n == n) & (s.noise == 0.05) & (s.grid == "uniform")
                    & (s.outliers == 0.0)],
                  f"noise 5%, uniform grid, no outliers, n = {n}")
    table(base[(base.noise == 0.05) & (base.grid == "uniform")
               & (base.outliers == 0.0)],
          "signed relative bias, %, noise 5%, uniform grid", value="bias")
    table(base[(base.noise == 0.05) & (base.grid == "uniform")
               & (base.outliers == 0.0)],
          "95% interval coverage (stages with a covariance), noise 5%, "
          "uniform grid", value="cov95", scale=1.0, fmt="{:.2f}")
    op = out / "ode.csv"
    if op.exists():
        o = pd.read_csv(op)
        piv = o.groupby(["law", "noise", "method"])["err"].median().unstack()
        tm = o.groupby(["law", "method"])["time"].median().unstack()
        lines += ["## weak form against NLLS on the integrated law, median "
                  "max relative error, %", "",
                  "| law | noise | weak | nlls | speed-up |",
                  "|---|---|---|---|---|"]
        for (law, noise), row in piv.iterrows():
            wk = row.get("weak", np.nan)
            nl = row.get("nlls", np.nan)
            sp_ = (tm.loc[law, "nlls"] / tm.loc[law, "weak"]
                   if "nlls" in tm.columns and np.isfinite(tm.loc[law].get(
                       "nlls", np.nan)) else np.nan)
            lines.append(f"| {law} | {100 * noise:g}% | {100 * wk:.2f} | "
                         + ("n/a" if not np.isfinite(nl) else f"{100 * nl:.2f}")
                         + " | " + ("n/a" if not np.isfinite(sp_)
                                    else f"{sp_:.0f}x") + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if fig_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fams = [f for f in FAMILIES if f in set(base.family)]
        fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.4),
                                 sharey=True)
        sub = base[(base.grid == "uniform") & (base.outliers == 0.0)]
        for ax, fam in zip(np.atleast_1d(axes), fams):
            for st in sorted(sub.stage.unique()):
                d = sub[(sub.family == fam) & (sub.stage == st)].sort_values(
                    "noise")
                ax.plot(100 * d.noise, 100 * d.err, marker="o", ms=3,
                        label=st)
            ax.set_yscale("log")
            ax.set_title(fam, fontsize=9)
            ax.set_xlabel("noise, %")
            ax.grid(True, alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("median relative error, %")
        np.atleast_1d(axes)[-1].legend(fontsize=5, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
    print("\n".join(lines[:40]))


def main() -> None:
    global NOISES, GRIDS, OUTLIERS, SIZES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--ode", action="store_true", help="the ODE matrix only")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    if a.report:
        report(out, pathlib.Path(a.fig) if a.fig else None)
        return
    if a.smoke:
        NOISES, GRIDS, OUTLIERS, SIZES = (0.05,), ("uniform",), (0.0,), (200,)
        batch_matrix(out, 1, ["exp_decay", "power_law"])
        ode_matrix(out, 1)
        return
    if a.ode:
        ode_matrix(out, a.seeds or 30)
        return
    if a.full:
        OUTLIERS, SIZES = (0.0, 0.05, 0.10, 0.20), (50, 200, 1000)
        seeds = a.seeds or 30
        batch_matrix(out, seeds, list(FAMILIES))
        ode_matrix(out, min(seeds * 2, 60))
        return
    seeds = a.seeds or 10
    batch_matrix(out, seeds, list(FIRST_CUT))
    ode_matrix(out, min(seeds * 3, 30))


if __name__ == "__main__":
    main()
