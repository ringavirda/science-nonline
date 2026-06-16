"""Benchmark + figure generation for the dtfit method docs.

Produces, for the per-method documentation under ``docs/methods/``:

  * figures (PNG) into ``docs/methods/figures/`` -- one usage plot per method,
    plus a cross-method error bar chart, in the style of the paper figures;
  * comparison tables (printed as GitHub-flavoured markdown to stdout) of the
    dtfit methods against established baselines -- SciPy ``curve_fit``
    (Levenberg-Marquardt NLLS, the NLS gold standard), ``numpy.polyfit``
    (linear-in-parameters surrogate) and, for streaming, the naive random
    walk -- on both *model* (synthetic, known ground truth) and *real*
    (COVID-19, USD/UAH) data.

Run (after ``python experiments/download_data.py``):

    python experiments/benchmark.py

Everything real is downloaded data; the synthetic signals are clearly labelled
as model data with a known ground truth and are used only where an exact
reference is needed to quote recovery error against the truth.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import dtfit as dt
from dtfit.streaming import EqualAreasFilter

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "methods" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True})


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_csv(name: str) -> tuple[list[str], np.ndarray]:
    rows = list(csv.reader((DATA_DIR / name).open()))[1:]
    dates = [r[0] for r in rows]
    values = np.array([float(r[1]) for r in rows])
    return dates, values


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err**2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"R2": r2, "RMSE": rmse, "MAPE": mape}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def timed(fn) -> tuple[object, float]:
    t0 = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - t0) * 1e3  # ms


# --------------------------------------------------------------------------- #
# model (synthetic) data with a known ground truth
# --------------------------------------------------------------------------- #
def synthetic_exponential(n: int = 80, noise: float = 0.05):
    """y = a*exp(b*x), a=1.0, b=1.2 on a modest range, with Gaussian noise.

    Reseeded per call so the table and the figure use identical samples.
    """
    rng = np.random.default_rng(0)
    a_true, b_true = 1.0, 1.2
    x = np.linspace(0.0, 1.5, n)
    clean = a_true * np.exp(b_true * x)
    y = clean + rng.normal(0.0, noise * clean.std(), n)
    return x, y, clean, (a_true, b_true)


def synthetic_atan(n: int = 120, noise: float = 0.08):
    """y = a*atan(w*x), a transcendental (non-Taylor) saturation curve."""
    rng = np.random.default_rng(1)
    a_true, w_true = 2.0, 3.0
    x = np.linspace(0.0, 1.5, n)
    clean = a_true * np.arctan(w_true * x)
    y = clean + rng.normal(0.0, noise * clean.std(), n)
    return x, y, clean, (a_true, w_true)


# --------------------------------------------------------------------------- #
# table 1: model-data exponential recovery (methods vs baselines)
# --------------------------------------------------------------------------- #
def table_model_exponential() -> str:
    x, y, clean, (a_t, b_t) = synthetic_exponential()
    expr, var = "a*exp(b*x)", "x"
    bounds = [(0.2, 5.0), (0.5, 4.0)]
    rows: list[list[str]] = []

    def add(name, coeffs, pred, dt_ms):
        m = metrics(clean, pred)
        ab = f"a={coeffs[0]:.3f}, b={coeffs[1]:.3f}" if coeffs is not None else "--"
        rows.append([name, ab, f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                     f"{m['MAPE']:.2f}", f"{dt_ms:.1f}"])

    res, ms = timed(lambda: dt.fit_lsi(x, y, expr, var, bounds=bounds))
    add("LSI", res.coeffs, np.asarray(res.model(x)), ms)

    res, ms = timed(lambda: dt.fit_eda(x, y, expr, var, p0=[1.0, 1.0]))
    add("EDA", res.coeffs, np.asarray(res.model(x)), ms)

    def sci():
        p, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), x, y,
                         p0=[1.0, 1.0], maxfev=10000)
        return p
    p, ms = timed(sci)
    add("SciPy curve_fit", p, p[0] * np.exp(p[1] * x), ms)

    def poly():
        return np.polyfit(x, y, 5)
    c, ms = timed(poly)
    add("numpy.polyfit (deg 5)", None, np.polyval(c, x), ms)

    headers = ["method", "recovered params", "R²", "RMSE", "MAPE %", "fit (ms)"]
    title = (f"**Model data** — `y = a·exp(b·x)`, ground truth a={a_t}, "
             f"b={b_t}, 5 % noise, n={x.size}. Error is against the *clean* "
             f"signal (true recovery). DSB is excluded from this head-to-head: "
             f"its symbolic reflection mishandles a bare leading coefficient "
             f"(see `dsb.md`); on its additive form it is a curve fit, not a "
             f"point estimator.")
    return title + "\n\n" + md_table(headers, rows)


# --------------------------------------------------------------------------- #
# table 1b: DSB on its intended additive form (curve-fit quality)
# --------------------------------------------------------------------------- #
def table_dsb_additive() -> str:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 150)
    clean = 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x)
    y = clean + rng.normal(0, 0.05, x.size)
    expr, var = "a0 + a1*x + a2*exp(a3*x)", "x"

    rows = []
    reg, ms = timed(lambda: dt.NonlineRegressor(expr, var, method="dsb").fit(x, y))
    m = metrics(clean, reg.predict(x))
    rows.append(["DSB (symbolic ref.)", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                 f"{m['MAPE']:.2f}", f"{ms:.1f}"])
    res, ms = timed(lambda: dt.fit_lsi(x, y, expr, var))
    m = metrics(clean, np.asarray(res.model(x)))
    rows.append(["LSI (same model)", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                 f"{m['MAPE']:.2f}", f"{ms:.1f}"])
    p, ms = timed(lambda: curve_fit(
        lambda xx, a0, a1, a2, a3: a0 + a1 * xx + a2 * np.exp(a3 * xx),
        x, y, p0=[1, 1, 1, 1], maxfev=20000)[0])
    m = metrics(clean, p[0] + p[1] * x + p[2] * np.exp(p[3] * x))
    rows.append(["SciPy curve_fit", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                 f"{m['MAPE']:.2f}", f"{ms:.1f}"])

    title = ("**Model data** — `y = a0 + a1·x + a2·exp(a3·x)` (DSB's intended "
             "additive form), x∈[0,3], 5 % noise. DSB is evaluated as a *curve "
             "fit* (R²); under noise it does not uniquely recover the four "
             "parameters, which is why the numeric LSI/EDA successors exist.")
    return title + "\n\n" + md_table(
        ["method", "R²", "RMSE", "MAPE %", "fit (ms)"], rows)


# --------------------------------------------------------------------------- #
# table 2: real-data summary (COVID + USD/UAH), reusing the validation setup
# --------------------------------------------------------------------------- #
def table_real_data() -> str:
    # --- COVID exponential growth window ---
    _, cum = load_csv("covid_ukraine_confirmed.csv")
    length, start = 28, next(i for i, v in enumerate(cum) if v >= 500)
    y = cum[start:start + length]
    t = np.linspace(0, 1.5, y.size)
    ys = y / y[0]
    bounds = [(0.2, 5.0), (0.5, 4.0)]

    cov_rows = []
    for name, coeffs in [
        ("LSI", dt.fit_lsi(t, ys, "a*exp(b*x)", "x", bounds=bounds).coeffs),
        ("EDA", dt.fit_eda(t, ys, "a*exp(b*x)", "x", p0=[ys[0], 1.0]).coeffs),
    ]:
        a, b = coeffs
        m = metrics(y, a * np.exp(b * t) * y[0])
        cov_rows.append([name, f"{m['R2']:.4f}", f"{m['RMSE']:.4g}", f"{m['MAPE']:.2f}"])
    p, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), t, ys,
                     p0=[ys[0], 1.0], maxfev=10000)
    m = metrics(y, p[0] * np.exp(p[1] * t) * y[0])
    cov_rows.append(["SciPy curve_fit", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}", f"{m['MAPE']:.2f}"])

    cov_tbl = md_table(["method", "R²", "RMSE", "MAPE %"], cov_rows)

    # --- USD/UAH streaming one-step-ahead vs random walk ---
    dates, rate = load_csv("usd_uah_2014_2015.csv")
    n = rate.size
    h = 1.5 / n
    tt = np.arange(n) * h
    r0 = rate[0]
    rs = rate / r0
    flt = EqualAreasFilter("a*exp(b*x)", "x", p0=[1.0, 0.0],
                           window_size=30, q_diag=[5e-3, 5e-3], r=0.5)
    track, truth, drifts = [], [], 0
    for i in range(n):
        flt.partial_fit(tt[i], rs[i])
        drifts += int(flt.drift_flag_)
        if len(flt._t) > 0:
            track.append(float(flt.predict(np.array([tt[i]]))[0]) * r0)
            truth.append(rate[i])
    track, truth = np.array(track), np.array(truth)
    m_track = metrics(truth, track)
    m_step = metrics(truth[1:], track[:-1])
    m_rw = metrics(truth[1:], truth[:-1])
    fx_rows = [
        ["EqualAreasFilter (lag-1 tracking)", f"{m_track['R2']:.4f}",
         f"{m_track['RMSE']:.4g}", f"{m_track['MAPE']:.2f}"],
        ["EqualAreasFilter (1-step-ahead)", f"{m_step['R2']:.4f}",
         f"{m_step['RMSE']:.4g}", f"{m_step['MAPE']:.2f}"],
        ["naive random walk (1-step)", f"{m_rw['R2']:.4f}",
         f"{m_rw['RMSE']:.4g}", f"{m_rw['MAPE']:.2f}"],
    ]
    fx_tbl = md_table(["method", "R²", "RMSE", "MAPE %"], fx_rows)

    return (
        "**Real data — COVID-19 Ukraine** (cumulative confirmed, 28-day "
        f"take-off, {y[0]:.0f}→{y[-1]:.0f} cases), exponential `y = a·exp(b·t)`:\n\n"
        + cov_tbl
        + "\n\n**Real data — USD/UAH 2014-2015** (NBU daily, "
        f"{rate[0]:.2f}→{rate[-1]:.2f}), streaming one-step-ahead vs the "
        f"random-walk benchmark; the filter flagged **{drifts}** structural "
        "break(s):\n\n" + fx_tbl
    )


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_lsi() -> None:
    x, y, clean, _ = synthetic_exponential()
    n_tr = int(x.size * 0.7)
    res = dt.fit_lsi(x[:n_tr], y[:n_tr], "a*exp(b*x)", "x", bounds=[(0.2, 5), (0.5, 4)])
    yhat = np.asarray(res.model(x))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x[:n_tr], y[:n_tr], s=14, c="0.5", label="train samples")
    ax[0].scatter(x[n_tr:], y[n_tr:], s=14, c="tab:orange", label="held-out")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    ax[0].plot(x, yhat, "tab:blue", lw=2, label="LSI fit")
    ax[0].axvline(x[n_tr], color="0.7", ls=":")
    ax[0].set_title("LSI fit — y = a·exp(b·x)")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    # spectrum view: empirical vs model Maclaurin discretes
    k = np.arange(6)
    z_emp = np.polyfit(x[:n_tr], y[:n_tr], 5)[::-1]
    a, b = res.coeffs
    z_mod = np.array([a * b**i / math.factorial(i) for i in k])
    width = 0.38
    ax[1].bar(k - width / 2, z_emp, width, label="empirical Z(k)", color="0.6")
    ax[1].bar(k + width / 2, z_mod, width, label="model F(k,c)", color="tab:blue")
    ax[1].set_title("Differential spectra matched by LSI")
    ax[1].set_xlabel("discrete order k")
    ax[1].set_ylabel("coefficient")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lsi_fit.png")
    plt.close(fig)


def fig_eda() -> None:
    x, y, clean, _ = synthetic_atan()
    res = dt.fit_eda(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    yhat = np.asarray(res.model(x))
    a, w = res.coeffs

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x, y, s=12, c="0.6", label="noisy samples (8 %)")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    ax[0].plot(x, yhat, "tab:green", lw=2, label="EDA fit")
    # shade the per-parameter integration windows
    n = 2
    idx_max = max(int(x.size * 0.8), n + 1)
    win = max(idx_max // n, 2)
    for i in range(n):
        s = i * win
        e = (i + 1) * win if i < n - 1 else idx_max
        ax[0].axvspan(x[s], x[min(e, x.size) - 1], alpha=0.07, color="tab:green")
    ax[0].set_title(f"EDA fit — y = a·atan(w·x)  (a={a:.2f}, w={w:.2f})")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    # cumulative area: the quantity EDA actually matches
    from scipy.integrate import cumulative_trapezoid
    area_d = cumulative_trapezoid(y, x, initial=0)
    area_m = cumulative_trapezoid(yhat, x, initial=0)
    ax[1].plot(x, area_d, "0.5", lw=2, label="∫ data")
    ax[1].plot(x, area_m, "tab:green", lw=1.5, ls="--", label="∫ model")
    ax[1].set_title("Equal-areas criterion (integral match)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("cumulative area")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_fit.png")
    plt.close(fig)


def fig_filter() -> None:
    dates, rate = load_csv("usd_uah_2014_2015.csv")
    n = rate.size
    h = 1.5 / n
    t = np.arange(n) * h
    r0 = rate[0]
    rs = rate / r0
    flt = EqualAreasFilter("a*exp(b*x)", "x", p0=[1.0, 0.0],
                           window_size=30, q_diag=[5e-3, 5e-3], r=0.5)
    track, drift_idx = [], []
    b_hist = []
    for i in range(n):
        flt.partial_fit(t[i], rs[i])
        if flt.drift_flag_:
            drift_idx.append(i)
        track.append(float(flt.predict(np.array([t[i]]))[0]) * r0 if len(flt._t) else np.nan)
        b_hist.append(flt.p[1])
    track = np.array(track)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(np.arange(n), rate, "0.5", lw=1, label="USD/UAH (real)")
    ax[0].plot(np.arange(n), track, "tab:red", lw=1.5, label="filter tracking")
    for j, di in enumerate(drift_idx):
        ax[0].axvline(di, color="tab:purple", ls="--", lw=1.2,
                      label="drift detected" if j == 0 else None)
    ax[0].set_title("EqualAreasFilter — online tracking + drift detection")
    ax[0].set_xlabel("sample (trading day)")
    ax[0].set_ylabel("UAH per USD")
    ax[0].legend(fontsize=8)

    ax[1].plot(np.arange(n), b_hist, "tab:red", lw=1.5)
    for di in drift_idx:
        ax[1].axvline(di, color="tab:purple", ls="--", lw=1.2)
    ax[1].set_title("Tracked growth parameter b (adapts after each reset)")
    ax[1].set_xlabel("sample")
    ax[1].set_ylabel("b estimate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "filter_tracking.png")
    plt.close(fig)


def fig_dsb() -> None:
    # DSB on its intended additive form (a leading multiplicative coefficient on
    # the whole expression is mishandled by the symbolic reflection). It is a
    # curve fit here, not an exact point estimator under noise.
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 150)
    clean = 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x)
    y = clean + rng.normal(0, 0.05, x.size)
    reg = dt.NonlineRegressor("a0 + a1*x + a2*exp(a3*x)", "x", method="dsb").fit(x, y)
    yhat = reg.predict(x)
    r2 = reg.score(x, y)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x, y, s=10, c="0.6", label="noisy samples")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    ax[0].plot(x, yhat, "tab:purple", lw=2, label="DSB fit")
    ax[0].set_title(f"DSB symbolic fit — a0+a1·x+a2·exp(a3·x)  (R²={r2:.3f})")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    ax[1].plot(x, yhat - y, "tab:purple", lw=1)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_title("Residuals (fit − data)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dsb_fit.png")
    plt.close(fig)


def fig_comparison() -> None:
    # MAPE on the synthetic exponential for each method (true-recovery error).
    x, y, clean, _ = synthetic_exponential()
    expr, var = "a*exp(b*x)", "x"
    preds = {}
    preds["LSI"] = np.asarray(dt.fit_lsi(x, y, expr, var, bounds=[(0.2, 5), (0.5, 4)]).model(x))
    preds["EDA"] = np.asarray(dt.fit_eda(x, y, expr, var, p0=[1.0, 1.0]).model(x))
    p, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), x, y, p0=[1, 1], maxfev=10000)
    preds["SciPy\ncurve_fit"] = p[0] * np.exp(p[1] * x)
    c = np.polyfit(x, y, 5)
    preds["numpy\npolyfit"] = np.polyval(c, x)

    names = list(preds)
    mapes = [metrics(clean, preds[k])["MAPE"] for k in names]
    colors = ["tab:blue", "tab:green", "0.5", "0.7"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(names, mapes, color=colors)
    ax.set_ylabel("MAPE % vs ground truth")
    ax.set_title("Parameter-recovery error on model exponential (5 % noise, lower is better)")
    for bar, mp in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2, mp, f"{mp:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_mape.png")
    plt.close(fig)


def main() -> None:
    if not (DATA_DIR / "covid_ukraine_confirmed.csv").exists():
        raise SystemExit("Datasets missing -- run: python experiments/download_data.py")

    print("Generating figures into", FIG_DIR)
    fig_lsi()
    fig_eda()
    fig_filter()
    fig_dsb()
    fig_comparison()
    print("  wrote:", ", ".join(sorted(p.name for p in FIG_DIR.glob("*.png"))))

    print("\n\n===== TABLE: model-data exponential recovery =====\n")
    print(table_model_exponential())
    print("\n\n===== TABLE: DSB additive-form curve fit =====\n")
    print(table_dsb_additive())
    print("\n\n===== TABLE: real-data summary =====\n")
    print(table_real_data())


if __name__ == "__main__":
    main()
