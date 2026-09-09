"""The evolution matrix: every stage of the method on one problem set.

Rows are the stages the estimator grew through, from the source-form
criteria in ``dtfit_legacy.book`` to the image in ``dtfit.fit``; columns are
model families crossed with noise, grid shape and outlier fraction. Every
stage starts from the same perturbed truth, so the matrix measures the
estimators, not the seeding. A second, smaller matrix compares the weak
form of ODE identification (no ODE solve, no start) with nonlinear least
squares on the integrated law.

Results go to CSV as they are produced, so a partial run is still usable::

    python -m dtfit_experimental.experiments.evolution --out DIR [--seeds N]
    python -m dtfit_experimental.experiments.evolution --out DIR --report

``--smoke`` runs one seed of one family to check the wiring. ``--report``
reads the CSVs and writes ``summary.csv``, ``summary.md`` and the figure.
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
}
NOISES = (0.01, 0.05, 0.10, 0.30)
GRIDS = ("uniform", "clustered", "random")
OUTLIERS = (0.0, 0.10)
N = 200
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


def _grid(kind, h, rng):
    if kind == "uniform":
        return np.linspace(0.0, h, N)
    if kind == "clustered":
        a = rng.uniform(0.0, 0.2 * h, N // 2)
        b = rng.uniform(0.2 * h, h, N - N // 2)
        x = np.sort(np.concatenate([a, b]))
    else:
        x = np.sort(rng.uniform(0.0, h, N))
    x[0], x[-1] = 0.0, h
    return x


def _data(model, truth, h, noise, grid, outliers, seed):
    rng = np.random.default_rng(seed)
    x = _grid(grid, h, rng)
    clean = model(x, *truth)
    amp = float(np.ptp(clean)) or 1.0
    sigma = noise * amp
    y = clean + sigma * rng.standard_normal(N)
    if outliers > 0:
        k = max(1, int(round(outliers * N)))
        idx = rng.choice(N, k, replace=False)
        y[idx] = clean[idx] + 10.0 * sigma * rng.standard_normal(k)
    return x, y


def _stages(model):
    def s0(x, y, expr, p0):
        return curve_fit(model, x, y, p0=p0, maxfev=4000)[0]

    def s2(x, y, expr, p0):
        deg = find_degree(x, y)
        coeffs = np.polynomial.polynomial.polyfit(x, y, deg)
        return fit_dsb(coeffs, expr, "t", p0=p0).coeffs

    return [
        ("S0_nlls", True, s0),
        ("S1_balance_book", False,
         lambda x, y, e, p0: dsb_balance(x, y, e, "t", p0=p0).coeffs),
        ("S2_balance_adequate", False, s2),
        ("S3a_lsi_monomial", True,
         lambda x, y, e, p0: lsi_integral_monomial(x, y, e, "t", p0=p0).coeffs),
        ("S3b_lsi_legendre", True,
         lambda x, y, e, p0: fit_lsi(x, y, e, "t", p0=p0).coeffs),
        ("S3c_eac_book", True,
         lambda x, y, e, p0: eac_areas(x, y, e, "t", p0=p0).coeffs),
        ("S3d_eac_direct", True,
         lambda x, y, e, p0: fit_eac(x, y, e, "t", p0=p0).coeffs),
        ("S4a_image_legendre", True,
         lambda x, y, e, p0: fit(e, Original(x, y), "t", basis="legendre",
                                 p0=p0).coeffs),
        ("S4b_image_block", True,
         lambda x, y, e, p0: fit(e, Original(x, y), "t", basis="block",
                                 p0=p0).coeffs),
        ("S4r_image_robust", True,
         lambda x, y, e, p0: fit(e, Original(x, y), "t", basis="legendre",
                                 robust=True, p0=p0).coeffs),
    ]


def batch_matrix(out: pathlib.Path, seeds: int, families) -> None:
    path = out / "batch.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["stage", "needs_p0", "family", "noise", "grid",
                        "outliers", "seed", "err", "ok", "time", "error"])
        for fam in families:
            expr, truth, h = FAMILIES[fam]
            names, tv = _truth_vector(expr, truth)
            model = _model(expr, names)
            p0 = tv * P0_FACTOR
            stages = _stages(model)
            t0 = time.time()
            for noise in NOISES:
                for grid in GRIDS:
                    for outl in OUTLIERS:
                        for seed in range(seeds):
                            x, y = _data(model, tv, h, noise, grid, outl,
                                         seed)
                            for name, needs, run in stages:
                                tic = time.perf_counter()
                                try:
                                    c = np.asarray(run(x, y, expr, p0),
                                                   dtype=float)
                                    err = float(np.median(
                                        np.abs(c / tv - 1.0)))
                                    ok = bool(np.isfinite(err))
                                    msg = ""
                                except Exception as exc:  # noqa: BLE001
                                    err, ok = float("nan"), False
                                    msg = type(exc).__name__
                                w.writerow([name, needs, fam, noise, grid,
                                            outl, seed, err, ok,
                                            time.perf_counter() - tic, msg])
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
    g = b.groupby(["stage", "needs_p0", "family", "noise", "grid",
                   "outliers"])
    s = g.agg(err=("err", "median"), ok=("ok", "mean"),
              time=("time", "median")).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    lines = ["# Evolution matrix, first cut", "",
             f"seeds per cell: {b['seed'].nunique()}, n = {N}, "
             f"p0 = truth x {P0_FACTOR} for every stage that takes one", ""]

    def table(sub, title):
        if sub.empty:
            return
        piv = sub.pivot_table(index="stage", columns="family", values="err",
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
                             f"{100 * e:.2f}" + ("" if o > 0.99 else
                                                  f" ({100 * o:.0f}% ok)"))
            lines.append(f"| {st} | {needs} | " + " | ".join(cells) + " |")
        lines.append("")

    for noise in NOISES:
        table(s[(s.noise == noise) & (s.grid == "uniform")
                & (s.outliers == 0.0)],
              f"median relative parameter error, %, noise {100 * noise:.0f}%, "
              "uniform grid, no outliers")
    table(s[(s.noise == 0.05) & (s.grid == "uniform") & (s.outliers == 0.10)],
          "noise 5%, uniform grid, 10% outliers")
    table(s[(s.noise == 0.05) & (s.grid == "random") & (s.outliers == 0.0)],
          "noise 5%, random grid, no outliers")
    op = out / "ode.csv"
    if op.exists():
        o = pd.read_csv(op)
        piv = o.groupby(["law", "noise", "method"])["err"].median().unstack()
        lines += ["## weak form against NLLS on the integrated law, median "
                  "max relative error, %", "",
                  "| law | noise | weak | nlls |", "|---|---|---|---|"]
        for (law, noise), row in piv.iterrows():
            wk = row.get("weak", np.nan)
            nl = row.get("nlls", np.nan)
            lines.append(f"| {law} | {100 * noise:g}% | {100 * wk:.2f} | "
                         + ("n/a" if not np.isfinite(nl) else f"{100 * nl:.2f}")
                         + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if fig_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fams = list(FAMILIES)
        fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4),
                                 sharey=True)
        sub = s[(s.grid == "uniform") & (s.outliers == 0.0)]
        for ax, fam in zip(axes, fams):
            for st in sorted(sub.stage.unique()):
                d = sub[(sub.family == fam) & (sub.stage == st)].sort_values(
                    "noise")
                ax.plot(100 * d.noise, 100 * d.err, marker="o", ms=3,
                        label=st)
            ax.set_yscale("log")
            ax.set_title(fam)
            ax.set_xlabel("noise, %")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("median relative error, %")
        axes[-1].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
    print("\n".join(lines[:40]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
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
        global NOISES, GRIDS, OUTLIERS
        NOISES, GRIDS, OUTLIERS = (0.05,), ("uniform",), (0.0,)
        batch_matrix(out, 1, ["exp_decay", "arctan"])
        ode_matrix(out, 1)
        return
    if a.ode:
        ode_matrix(out, a.seeds)
        return
    batch_matrix(out, a.seeds, list(FAMILIES))
    ode_matrix(out, min(a.seeds * 3, 30))


if __name__ == "__main__":
    main()
