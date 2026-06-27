"""Backend for the **stochastic-series** domain experiment.

Single source of truth for the question: *can dtfit -- a deterministic curve
fitter -- be put to work on genuinely random series (economic / financial
data)?* dtfit cannot fit a martingale path; but it can fit the **deterministic
functionals** of a stochastic process (its autocovariance, spectrum,
aggregated-variance and trend/cycle), and from those recover the process's
parameters. The estimators live in :mod:`dtfit.stochastic`; this
module supplies the **ground-truth data generators** and the **evaluation
harness** that measures, for each possibility, whether the dtfit route actually
recovers the known parameter (and how it compares to the standard estimator).

Each experiment simulates a process with a *known* parameter, runs the dtfit
estimator (and a plain baseline) over several seeds, and reports the mean
recovery error plus a categorical verdict (VIABLE / MARGINAL / NOT VIABLE):

* **E1 long memory (aggregated variance)** -- ARFIMA(0,d,0); recover the Hurst
  exponent ``H = d + 1/2`` from the power-law aggregated-variance curve.
* **E2 long memory (spectrum)** -- same series; recover ``H`` from the
  low-frequency log-periodogram slope (the GPH route) via LSI.
* **E3 mean reversion** -- AR(1)/OU; recover the AR(1) coefficient from an
  exponential fit to the ACF.
* **E4 volatility persistence** -- GARCH(1,1); recover ``alpha + beta`` from an
  exponential fit to the ACF of squared returns.
* **E5 stochastic cycle** -- AR(2) with complex roots; recover the cycle period
  from a damped-cosine fit to the ACF.
* **E6 trend + cycle decomposition** -- structural series + noise; recover the
  trend slope and cycle period, leaving a stochastic residual.

``run()`` returns the rows; ``summary()`` renders the markdown verdict table.
"""

from __future__ import annotations

import numpy as np

from dtfit.stochastic import (
    hurst_aggvar,
    hurst_spectral,
    ar1_reversion,
    garch_persistence,
    cycle_period,
    decompose_trend_cycle,
    fit_stochastic,
    StochasticFilter,
)
from dtfit_experimental.experiments.common import EXPERIMENTS_DIR, metrics
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "gen_arfima", "gen_ar1", "gen_garch", "gen_ar2_cycle", "gen_trend_cycle",
    "exp_hurst_aggvar", "exp_hurst_spectral", "exp_ar1", "exp_garch",
    "exp_cycle", "exp_decompose",
    "exp_merged_router", "forecast_skill", "exp_forecast_skill",
    "load_series", "exp_real_data",
    "REAL_DATASETS", "hurst_comparison", "exp_real_suite", "panel_forecasts",
    "exp_online_filter", "filter_trace",
    "exp_simulate_roundtrip", "simulate_example",
    "filter_break_demo", "filter_characteristics",
    "run", "summary",
]


# --------------------------------------------------------------------------- #
# ground-truth data generators
# --------------------------------------------------------------------------- #
def gen_arfima(n: int, d: float, rng: np.random.Generator,
               *, ntrunc: int = 1200) -> np.ndarray:
    """ARFIMA(0, d, 0): white noise fractionally integrated to order ``d``.

    Built from the truncated MA(inf) expansion of ``(1 - B)^(-d)``
    (``psi_0 = 1``, ``psi_j = psi_{j-1} (j-1+d)/j``). Exhibits long memory with
    Hurst exponent ``H = d + 1/2``.
    """
    psi = np.empty(ntrunc)
    psi[0] = 1.0
    for j in range(1, ntrunc):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    e = rng.standard_normal(n + ntrunc)
    x = np.convolve(e, psi)[ntrunc: ntrunc + n]
    return np.asarray(x, dtype=float)


def gen_ar1(n: int, phi: float, rng: np.random.Generator,
            *, sigma: float = 1.0, burn: int = 200) -> np.ndarray:
    """AR(1) / discretely-sampled Ornstein-Uhlenbeck: ``x_t = phi x_{t-1} + e``."""
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def gen_garch(n: int, omega: float, alpha: float, beta: float,
              rng: np.random.Generator, *, burn: int = 500) -> np.ndarray:
    """GARCH(1,1) returns; volatility persistence is ``alpha + beta``."""
    N = n + burn
    z = rng.standard_normal(N)
    s2 = np.empty(N)
    r = np.empty(N)
    s2[0] = omega / max(1e-9, 1.0 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, N):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:]


def gen_ar2_cycle(n: int, period: float, damping: float,
                  rng: np.random.Generator, *, burn: int = 300) -> np.ndarray:
    """AR(2) with complex roots: a stochastic pseudo-cycle of the given period
    and damping (root modulus). ``phi1 = 2 r cos(2pi/period)``, ``phi2 = -r^2``."""
    phi1 = 2.0 * damping * np.cos(2.0 * np.pi / period)
    phi2 = -(damping ** 2)
    N = n + burn
    e = rng.standard_normal(N)
    x = np.zeros(N)
    for t in range(2, N):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x[burn:]


def gen_trend_cycle(n: int, slope: float, period: float, amp: float,
                    noise_sd: float, rng: np.random.Generator,
                    *, intercept: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic trend + cycle plus i.i.d. noise -> ``(t, y)``."""
    t = np.arange(n, dtype=float)
    y = (intercept + slope * t + amp * np.sin(2.0 * np.pi * t / period)
         + rng.normal(0.0, noise_sd, n))
    return t, y


# --------------------------------------------------------------------------- #
# verdict helper
# --------------------------------------------------------------------------- #
def _verdict(err: float, good: float, ok: float) -> str:
    if not np.isfinite(err):
        return "NOT VIABLE"
    if err <= good:
        return "VIABLE"
    if err <= ok:
        return "MARGINAL"
    return "NOT VIABLE"


def _rel(est: float, truth: float) -> float:
    return abs(est - truth) / abs(truth) * 100.0 if truth != 0 else float("nan")


def _mean(vals: list[float]) -> float:
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)) if a.size and np.any(np.isfinite(a)) else float("nan")


# --------------------------------------------------------------------------- #
# E1 -- long memory via aggregated variance
# --------------------------------------------------------------------------- #
def exp_hurst_aggvar(seeds: int = 8, *, n: int = 4096,
                     ds: tuple[float, ...] = (0.2, 0.3, 0.4)) -> dict:
    dt_e, base_e, rs_e, dfa_e = [], [], [], []
    for d in ds:
        H = d + 0.5
        for s in range(seeds):
            x = gen_arfima(n, d, np.random.default_rng(1000 + s))
            try:
                dt_e.append(abs(hurst_aggvar(x, method="lsi")["H"] - H))
            except Exception:
                dt_e.append(np.nan)
            try:
                base_e.append(abs(hurst_aggvar(x, method="ols")["H"] - H))
            except Exception:
                base_e.append(np.nan)
            try:
                rs_e.append(abs(bl.hurst_rs(x) - H))
            except Exception:
                rs_e.append(np.nan)
            try:
                dfa_e.append(abs(bl.hurst_dfa(x) - H))
            except Exception:
                dfa_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E1", "name": "long memory (aggregated variance)",
        "param": "Hurst H", "metric": "MAE(H)",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 0.07, 0.15),
        "method": "LSI power-law fit to Var(block mean) vs scale",
        "extra": {"OLS log-log": _mean(base_e), "R/S": _mean(rs_e),
                  "DFA": _mean(dfa_e)},
    }


# --------------------------------------------------------------------------- #
# E2 -- long memory via the low-frequency spectrum (GPH)
# --------------------------------------------------------------------------- #
def exp_hurst_spectral(seeds: int = 8, *, n: int = 4096,
                       ds: tuple[float, ...] = (0.2, 0.3, 0.4)) -> dict:
    dt_e, base_e, rs_e, dfa_e = [], [], [], []
    for d in ds:
        H = d + 0.5
        for s in range(seeds):
            x = gen_arfima(n, d, np.random.default_rng(2000 + s))
            try:
                dt_e.append(abs(hurst_spectral(x, method="lsi")["H"] - H))
            except Exception:
                dt_e.append(np.nan)
            try:
                base_e.append(abs(hurst_spectral(x, method="ols")["H"] - H))
            except Exception:
                base_e.append(np.nan)
            try:
                rs_e.append(abs(bl.hurst_rs(x) - H))
            except Exception:
                rs_e.append(np.nan)
            try:
                dfa_e.append(abs(bl.hurst_dfa(x) - H))
            except Exception:
                dfa_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E2", "name": "long memory (low-freq spectrum / GPH)",
        "param": "Hurst H", "metric": "MAE(H)",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 0.10, 0.20),
        "method": "LSI slope of log-periodogram (smoothed)",
        "extra": {"GPH/OLS": _mean(base_e), "R/S": _mean(rs_e),
                  "DFA": _mean(dfa_e)},
    }


# --------------------------------------------------------------------------- #
# E3 -- mean reversion (AR(1) / OU)
# --------------------------------------------------------------------------- #
def exp_ar1(seeds: int = 8, *, n: int = 1500,
            phis: tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
    dt_e, base_e = [], []
    for phi in phis:
        for s in range(seeds):
            x = gen_ar1(n, phi, np.random.default_rng(3000 + s))
            try:
                dt_e.append(_rel(ar1_reversion(x, method="lsi")["phi"], phi))
            except Exception:
                dt_e.append(np.nan)
            try:
                base_e.append(_rel(ar1_reversion(x, method="acf1")["phi"], phi))
            except Exception:
                base_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E3", "name": "mean reversion (AR(1) / OU)",
        "param": "AR(1) phi", "metric": "rel.err %",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 8.0, 20.0),
        "method": "LSI exponential fit to the ACF",
    }


# --------------------------------------------------------------------------- #
# E4 -- volatility persistence (GARCH(1,1))
# --------------------------------------------------------------------------- #
def exp_garch(seeds: int = 8, *, n: int = 4000,
              params: tuple[tuple[float, float, float], ...] = (
                  (0.05, 0.08, 0.90), (0.05, 0.10, 0.85), (0.02, 0.05, 0.93))
              ) -> dict:
    dt_e = []
    for (omega, alpha, beta) in params:
        persist = alpha + beta
        for s in range(seeds):
            r = gen_garch(n, omega, alpha, beta, np.random.default_rng(4000 + s))
            try:
                est = garch_persistence(r, method="lsi", use="abs")["persistence"]
                dt_e.append(_rel(est, persist))
            except Exception:
                dt_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E4", "name": "volatility persistence (GARCH(1,1))",
        "param": "alpha+beta", "metric": "rel.err %",
        "dt_err": err, "base_err": float("nan"),
        "verdict": _verdict(err, 12.0, 30.0),
        "method": "LSI exponential fit to ACF of |returns|",
    }


# --------------------------------------------------------------------------- #
# E5 -- stochastic cycle (AR(2), complex roots)
# --------------------------------------------------------------------------- #
def exp_cycle(seeds: int = 8, *, n: int = 1200,
              periods: tuple[float, ...] = (10.0, 16.0, 25.0),
              damping: float = 0.97) -> dict:
    dt_e, base_e = [], []
    for P in periods:
        for s in range(seeds):
            x = gen_ar2_cycle(n, P, damping, np.random.default_rng(5000 + s))
            try:
                dt_e.append(_rel(cycle_period(x)["period"], P))
            except Exception:
                dt_e.append(np.nan)
            # baseline: FFT periodogram peak period
            try:
                xx = x - x.mean()
                spec = np.abs(np.fft.rfft(xx)) ** 2
                freqs = np.fft.rfftfreq(xx.size, d=1.0)
                spec[0] = 0.0
                pk = freqs[int(np.argmax(spec))]
                base_e.append(_rel(1.0 / pk if pk > 0 else np.inf, P))
            except Exception:
                base_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E5", "name": "stochastic cycle (AR(2) complex roots)",
        "param": "cycle period", "metric": "rel.err %",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 12.0, 30.0),
        "method": "LSI damped-cosine fit to the ACF",
    }


# --------------------------------------------------------------------------- #
# E6 -- trend + cycle decomposition
# --------------------------------------------------------------------------- #
def exp_decompose(seeds: int = 8, *, n: int = 600, slope: float = 0.02,
                  period: float = 50.0, amp: float = 3.0,
                  noise_sd: float = 1.0) -> dict:
    slope_e, period_e = [], []
    for s in range(seeds):
        t, y = gen_trend_cycle(n, slope, period, amp, noise_sd,
                               np.random.default_rng(6000 + s))
        try:
            dec = decompose_trend_cycle(t, y, trend_deg=1)
            slope_e.append(_rel(dec["slope"], slope))
            period_e.append(_rel(dec["period"], period))
        except Exception:
            slope_e.append(np.nan)
            period_e.append(np.nan)
    err = _mean([_mean(slope_e), _mean(period_e)])
    return {
        "id": "E6", "name": "trend + cycle decomposition",
        "param": "slope & period", "metric": "rel.err %",
        "dt_err": err, "base_err": float("nan"),
        "verdict": _verdict(err, 12.0, 30.0),
        "method": "LSI trend + oscillatory-recipe cycle",
        "extra": {"slope_err": _mean(slope_e), "period_err": _mean(period_e)},
    }


# --------------------------------------------------------------------------- #
# E7 -- the merged solution: does the single fit_stochastic router identify the
# right regime for each process? (the analogue of the other domains' "merged"
# pipeline + applicability map)
# --------------------------------------------------------------------------- #
def _regime_match(model, expect: str) -> bool:
    """Does the model's detected regime / components match the expected label?"""
    reg = model.regime.lower()
    if expect == "white noise":
        return reg.startswith("white noise")
    if expect == "random walk":
        return reg.startswith("random walk")
    if expect == "mean-reverting":
        return "mean-revert" in reg
    if expect == "long-memory":
        return "long-memory" in reg
    if expect == "vol-clustering":
        return bool(model.has_vol_clustering)
    if expect == "cyclical":
        return any(w in reg for w in ("cyclical", "cycle", "seasonal"))
    if expect == "trend+cycle":
        return "trend+cycle" in reg or "trend+seasonal" in reg
    return False


def _router_cases() -> list[tuple[str, str, object]]:
    """``(process name, expected regime, generator(seed)->series)``."""
    return [
        ("white noise", "white noise",
         lambda s: np.random.default_rng(s).standard_normal(1500)),
        ("random walk", "random walk",
         lambda s: np.cumsum(np.random.default_rng(s).standard_normal(1500))),
        ("AR(1)/OU", "mean-reverting",
         lambda s: gen_ar1(1500, 0.7, np.random.default_rng(s))),
        ("ARFIMA (long memory)", "long-memory",
         lambda s: gen_arfima(4096, 0.3, np.random.default_rng(s))),
        ("GARCH(1,1)", "vol-clustering",
         lambda s: gen_garch(4000, 0.05, 0.08, 0.90, np.random.default_rng(s))),
        ("AR(2) cycle", "cyclical",
         lambda s: gen_ar2_cycle(1500, 16.0, 0.97, np.random.default_rng(s))),
        ("trend + cycle", "trend+cycle",
         lambda s: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0,
                                   np.random.default_rng(s))[1]),
    ]


def exp_merged_router(seeds: int = 8) -> dict:
    """Run ``fit_stochastic`` over every process and score regime-ID accuracy."""
    rows = []
    correct = total = 0
    for name, expect, gen in _router_cases():
        hits = 0
        for s in range(seeds):
            try:
                if _regime_match(fit_stochastic(gen(700 + s)), expect):
                    hits += 1
            except Exception:
                pass
        rows.append({"process": name, "expected": expect,
                     "detected %": 100.0 * hits / seeds})
        correct += hits
        total += seeds
    acc = 100.0 * correct / total if total else float("nan")
    return {
        "id": "E7", "name": "merged router (regime identification)",
        "param": "regime", "metric": "accuracy %",
        "dt_err": 100.0 - acc, "base_err": float("nan"),
        "verdict": _verdict(100.0 - acc, 10.0, 25.0),
        "method": "fit_stochastic gated composition", "accuracy": acc,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# forecast skill: the merged solution vs the established forecasters
# --------------------------------------------------------------------------- #
def forecast_skill(y: np.ndarray, h: int) -> tuple[dict, str]:
    """Hold out the last ``h`` points; return ``({method: RMSE}, regime)`` for the
    merged solution and the established baselines (random walk, drift, AR(1),
    ARIMA, ETS, Theta -- the statsmodels ones skipped if unavailable)."""
    y = np.asarray(y, dtype=float)
    train, test = y[:-h], y[-h:]
    out: dict[str, float] = {}
    model = fit_stochastic(train)
    out["dtfit merged"] = metrics(test, model.forecast(h))["RMSE"]
    out["random walk"] = metrics(test, bl.random_walk_forecast(train, h))["RMSE"]
    out["drift"] = metrics(test, bl.drift_forecast(train, h))["RMSE"]
    for label, fn in [
        ("AR(1)", lambda: bl.arima_forecast(train, h, order=(1, 0, 0))),
        ("ARIMA(2,1,2)", lambda: bl.arima_forecast(train, h, order=(2, 1, 2))),
        ("ETS", lambda: bl.ets_forecast(train, h)),
        ("Theta", lambda: bl.theta_forecast(train, h)),
    ]:
        try:
            out[label] = metrics(test, fn())["RMSE"]
        except Exception:
            out[label] = float("nan")
    return out, model.regime


def exp_forecast_skill(seeds: int = 5) -> dict:
    """Forecast-skill of the merged solution vs baselines on the structured
    processes where extrapolation is meaningful (trend+cycle, mean-reverting).
    Reported as RMSE ratio to the random-walk benchmark (``<1`` = beats RW)."""
    cases = [
        ("trend + cycle", lambda s: gen_trend_cycle(
            400, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(s))[1], 40),
        ("AR(1) mean-revert", lambda s: gen_ar1(
            800, 0.6, np.random.default_rng(s)), 30),
    ]
    rows = []
    for name, gen, h in cases:
        ratios = {}
        for s in range(seeds):
            sk, _ = forecast_skill(gen(800 + s), h)
            rw = sk.get("random walk", np.nan)
            for k, v in sk.items():
                ratios.setdefault(k, []).append(v / rw if rw else np.nan)
        rows.append({"process": name,
                     **{k: _mean(v) for k, v in ratios.items()}})
    return {"id": "E8", "name": "forecast skill (RMSE ratio to random walk)",
            "rows": rows}


# --------------------------------------------------------------------------- #
# E9 -- real economic data (USD/UAH 2014-15 daily rate)
# --------------------------------------------------------------------------- #
def load_series(name: str, col: int = 1) -> np.ndarray:
    """Load one column from a bundled real-data CSV (``experiments/data``)."""
    import csv

    rows = list(csv.reader((EXPERIMENTS_DIR / "data" / name).open()))[1:]
    return np.array([float(r[col]) for r in rows if r and r[col] not in ("", "NA")])


def exp_real_data() -> dict:
    """Characterize a real economic series with the merged solution and check the
    forecast against the established baselines.

    USD/UAH 2014-15 daily exchange rate: the **level** is a near-random-walk
    (the classic finding that nothing beats persistence on an FX level); its
    **log-returns** are the stationary object that carries volatility clustering.
    The merged router should call the level a random walk (and tie RW on
    forecast), and flag clustering in the returns.
    """
    rate = load_series("usd_uah_2014_2015.csv")
    logret = np.diff(np.log(rate))

    m_level = fit_stochastic(rate)
    m_ret = fit_stochastic(logret)

    h = 20
    skill, regime = forecast_skill(rate, h)

    # Hurst of |log-returns| (volatility long memory) -- dtfit vs the classics
    absr = np.abs(logret - logret.mean())
    hurst_abs = {
        "dtfit spectral": _safe(lambda: hurst_spectral(absr)["H"]),
        "dtfit aggvar": _safe(lambda: hurst_aggvar(absr)["H"]),
        "R/S": _safe(lambda: bl.hurst_rs(absr)),
        "DFA": _safe(lambda: bl.hurst_dfa(absr)),
    }
    return {
        "id": "E9", "name": "real data (USD/UAH 2014-15)",
        "level_regime": m_level.regime,
        "level_components": ", ".join(m_level.components),
        "returns_vol_clustering": bool(m_ret.has_vol_clustering),
        "returns_vol_persistence": m_ret.vol_persistence,
        "forecast_regime": regime,
        "forecast_rmse": skill,
        "abs_returns_hurst": hurst_abs,
    }


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# E10 -- a gallery of real datasets, one per regime, vs known literature
# --------------------------------------------------------------------------- #
def _sm_series(key: str) -> np.ndarray:
    """Load a canonical real series from ``statsmodels.datasets`` (no download)."""
    import statsmodels.api as sm

    if key == "nile":
        return sm.datasets.nile.load_pandas().data["volume"].to_numpy(float)
    if key == "sunspots":
        return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)
    if key == "co2":
        s = sm.datasets.co2.load_pandas().data["co2"]
        return s.interpolate().to_numpy(float)
    if key == "gdp":
        return sm.datasets.macrodata.load_pandas().data["realgdp"].to_numpy(float)
    if key == "tbill":
        return sm.datasets.macrodata.load_pandas().data["tbilrate"].to_numpy(float)
    raise KeyError(key)


# Each entry: a real series, the regime domain knowledge / the literature predict,
# and the established result it should reproduce. ``source`` is a thunk so a
# missing statsmodels (or data file) only fails that one row.
REAL_DATASETS = [
    dict(key="nile", title="Nile annual volume (1871-1970)", domain="hydrology",
         expect="long memory / trend", source=lambda: _sm_series("nile"),
         lit="Hurst's canonical long-memory series (H ~ 0.9)"),
    dict(key="sunspots", title="Sunspot number (1700-2008)", domain="solar physics",
         expect="cyclical", source=lambda: _sm_series("sunspots"),
         lit="the ~11-year solar cycle"),
    dict(key="co2", title="Mauna Loa CO2 (weekly)", domain="climate",
         expect="trend+cycle", source=lambda: _sm_series("co2"),
         lit="rising trend + annual cycle"),
    dict(key="gdp", title="US real GDP (quarterly)", domain="macroeconomics",
         expect="random walk + drift", source=lambda: _sm_series("gdp"),
         lit="Nelson-Plosser: GDP is a random walk with drift"),
    dict(key="tbill", title="US 3-month T-bill rate (quarterly)",
         domain="macroeconomics", expect="random walk / mean-revert",
         source=lambda: _sm_series("tbill"),
         lit="short rates are near-unit-root (highly persistent)"),
    dict(key="usd_uah", title="USD/UAH (2014-15, daily)", domain="FX",
         expect="random walk", source=lambda: load_series("usd_uah_2014_2015.csv"),
         lit="an FX level is a near-random walk; clustering in the returns"),
    dict(key="fx_ltsf", title="LTSF exchange rate (daily)", domain="FX",
         expect="random walk", source=lambda: load_series("ltsf/exchange_rate.csv", -1),
         lit="FX level near-random walk; long memory in volatility"),
]


def hurst_comparison(y: np.ndarray) -> dict[str, float]:
    """Hurst of ``y`` by every estimator (dtfit spectral / aggvar, R/S, DFA)."""
    return {
        "dtfit spectral": _safe(lambda: hurst_spectral(y)["H"]),
        "dtfit aggvar": _safe(lambda: hurst_aggvar(y)["H"]),
        "R/S": _safe(lambda: bl.hurst_rs(y)),
        "DFA": _safe(lambda: bl.hurst_dfa(y)),
    }


def _suite_horizon(n: int) -> int:
    """Held-out forecast horizon scaled to the series length."""
    return int(min(max(n // 8, 12), 40))


def exp_real_suite() -> dict:
    """Run the merged solution on every real dataset; report the detected regime
    and the held-out forecast skill of the merged solution **and every baseline**
    (RMSE ratio to the random walk; ``< 1`` beats RW)."""
    rows = []
    for ds in REAL_DATASETS:
        try:
            y = ds["source"]()
        except Exception as exc:
            rows.append({"dataset": ds["title"], "domain": ds["domain"], "n": 0,
                         "expected": ds["expect"],
                         "detected regime": f"(unavailable: {type(exc).__name__})",
                         "components": "", "h": 0, "merged/RW": float("nan"),
                         "skill": {}, "literature": ds["lit"]})
            continue
        m = fit_stochastic(y)
        h = _suite_horizon(y.size)
        ratio = float("nan")
        skill: dict[str, float] = {}
        if y.size > h + 60:
            try:
                sk, _ = forecast_skill(y, h)
                rw = sk.get("random walk", np.nan)
                skill = {k: (v / rw if rw else np.nan) for k, v in sk.items()}
                ratio = skill.get("dtfit merged", np.nan)
            except Exception:
                pass
        rows.append({
            "dataset": ds["title"], "domain": ds["domain"], "n": int(y.size),
            "expected": ds["expect"], "detected regime": m.regime,
            "components": ", ".join(m.components), "h": h, "merged/RW": ratio,
            "skill": skill, "literature": ds["lit"],
        })
    return {"id": "E10", "name": "real-data gallery (regime + forecast skill)",
            "rows": rows}


def panel_forecasts(y: np.ndarray, h: int) -> dict:
    """Held-out forecast of every method for one series, as arrays for plotting.

    Returns ``{"train", "test", "preds": {method: array}}`` -- the merged
    solution plus the established baselines over the last ``h`` points."""
    y = np.asarray(y, dtype=float)
    train, test = y[:-h], y[-h:]
    preds = {
        "dtfit merged": fit_stochastic(train).forecast(h),
        "random walk": bl.random_walk_forecast(train, h),
        "drift": bl.drift_forecast(train, h),
    }
    for label, fn in [
        ("ARIMA(2,1,2)", lambda: bl.arima_forecast(train, h, order=(2, 1, 2))),
        ("Theta", lambda: bl.theta_forecast(train, h)),
    ]:
        try:
            preds[label] = np.asarray(fn(), dtype=float)
        except Exception:
            preds[label] = np.full(h, np.nan)
    return {"train": train, "test": test, "preds": preds}


# --------------------------------------------------------------------------- #
# E11 -- streaming characterization + online regime-change detection
# (the online twin of fit_stochastic's second-order stage; the stochastic
# counterpart of the embedded domain's streaming filter + fault detector)
# --------------------------------------------------------------------------- #
def filter_trace(y: np.ndarray, *, nlags: int = 24, halflife: float = 150.0,
                 warmup: int = 80, settle: int = 500,
                 z_thresh: float = 4.0) -> dict:
    """Run a :class:`StochasticFilter` over ``y`` and return the online phi /
    volatility traces and the detected change-point times (arrays for plotting)."""
    f = StochasticFilter(nlags=nlags, halflife=halflife, warmup=warmup,
                         settle=settle, z_thresh=z_thresh)
    phi = np.empty(y.size)
    vol = np.empty(y.size)
    for i, x in enumerate(np.asarray(y, dtype=float)):
        f.update(x)
        p = f.params_
        phi[i] = p["ar1_phi"]
        vol[i] = p["sigma"]
    return {"phi": phi, "sigma": vol, "flags": list(f.flag_times_),
            "snapshot": f.snapshot()}


def exp_online_filter(seeds: int = 8) -> dict:
    """Score the streaming filter on three axes: (1) does the online AR(1) phi
    converge to the truth, (2) does it flag a structural break (persistence jump,
    volatility switch) promptly, (3) how often does it false-alarm on a
    stationary stream. The streaming analogue of E7's batch router."""
    # (1) online phi tracking error vs ground truth
    phi_err = []
    for phi in (0.3, 0.6, 0.85):
        for s in range(seeds):
            y = gen_ar1(3000, phi, np.random.default_rng(700 + s))
            f = StochasticFilter(halflife=300, warmup=100)
            f.partial_fit(y)
            phi_err.append(abs(f.params_["ar1_phi"] - phi))
    track_err = _mean(phi_err)

    # (2) regime-break detection: a persistence jump and a volatility switch at
    # the midpoint; report hit-rate and median detection latency.
    def _break_case(kind, s):
        r = np.random.default_rng(800 + s)
        if kind == "persistence":
            a = gen_ar1(1500, 0.2, r)
            b = gen_ar1(1500, 0.9, r)
            b = b - b.mean() + a[-1]
            return np.concatenate([a, b])
        a = r.normal(0.0, 1.0, 1500)
        b = r.normal(0.0, 2.0, 1500)
        return np.concatenate([a, b])

    hits, lats = 0, []
    total = 0
    for kind in ("persistence", "volatility"):
        for s in range(seeds):
            total += 1
            f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
            f.partial_fit(_break_case(kind, s))
            near = [t for t in f.flag_times_ if 1500 <= t <= 1900]
            if near:
                hits += 1
                lats.append(near[0] - 1500)
    hit_rate = 100.0 * hits / total if total else float("nan")
    latency = float(np.median(lats)) if lats else float("nan")

    # (3) false-alarm rate on a stationary stream
    fa = []
    for s in range(seeds):
        f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
        f.partial_fit(gen_ar1(3000, 0.6, np.random.default_rng(900 + s)))
        fa.append(f.n_flags_)
    false_alarms = _mean(fa)

    return {
        "id": "E11", "name": "streaming filter (online tracking + break detection)",
        "param": "phi track / break hit-rate", "metric": "MAE / %",
        "dt_err": track_err, "base_err": float("nan"),
        "verdict": _verdict(track_err, 0.05, 0.12),
        "method": "EWMA-autocovariance StochasticFilter + fused change detector",
        "track_mae": track_err, "break_hit_rate": hit_rate,
        "median_latency": latency, "false_alarms_per_3000": false_alarms,
    }


# --------------------------------------------------------------------------- #
# E12 -- the generative model: fit -> simulate -> refit round-trip
# (StochasticModel.simulate draws a fresh realization from the detected
# components; re-fitting it must recover the same regime -- the honest proof that
# the batch model is a faithful *generator*, not only a summary)
# --------------------------------------------------------------------------- #
def exp_simulate_roundtrip(seeds: int = 5) -> dict:
    """Per-regime round-trip recovery rate of ``StochasticModel.simulate``."""
    cases = [
        ("trend + cycle",
         lambda r: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, r)[1], "has_cycle"),
        ("AR(1) mean-reverting", lambda r: gen_ar1(1500, 0.7, r),
         "has_mean_reversion"),
        ("ARFIMA long-memory", lambda r: gen_arfima(4096, 0.3, r),
         "has_long_memory"),
        ("GARCH vol-clustering",
         lambda r: gen_garch(4000, 0.05, 0.08, 0.90, r), "has_vol_clustering"),
        ("random walk", lambda r: np.cumsum(r.standard_normal(1500)), None),
        ("white noise", lambda r: r.standard_normal(1500), None),
    ]
    rows = []
    for name, gen, attr in cases:
        hits = 0
        for s in range(seeds):
            m = fit_stochastic(gen(np.random.default_rng(10 + s)))
            m2 = fit_stochastic(m.simulate(seed=s))
            if attr is None:
                hits += int(m.regime.split()[0] == m2.regime.split()[0])
            else:
                hits += int(bool(getattr(m2, attr)))
        rows.append({"process": name, "round-trip recovery %": 100.0 * hits / seeds})
    return {"id": "E12", "name": "generative round-trip (fit->simulate->refit)",
            "rows": rows}


def simulate_example(seed: int = 0) -> dict:
    """One original vs simulated realization per regime, for the figure."""
    specs = [
        ("trend + cycle",
         gen_trend_cycle(500, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(1))[1]),
        ("AR(1) mean-reverting", gen_ar1(800, 0.75, np.random.default_rng(2))),
        ("GARCH vol-clustering",
         gen_garch(2000, 0.05, 0.08, 0.90, np.random.default_rng(3))),
    ]
    out = []
    for name, y in specs:
        m = fit_stochastic(y)
        out.append({"name": name, "regime": m.regime,
                    "original": y, "simulated": m.simulate(seed=seed)})
    return {"examples": out}


# --------------------------------------------------------------------------- #
# streaming-filter demonstrations (E11 companion): a change-point trace and the
# flat-memory / bounded-cost profile that matches dtfit's own streaming filters
# --------------------------------------------------------------------------- #
def filter_break_demo(*, n_seg: int = 1500, seed: int = 0) -> dict:
    """A stream with a KNOWN structural break at the midpoint (an AR(1)
    persistence jump 0.2 -> 0.9) run through :class:`StochasticFilter`; returns
    the series, the true break, the online phi trace and the detected flags."""
    r = np.random.default_rng(seed)
    a = gen_ar1(n_seg, 0.2, r)
    b = gen_ar1(n_seg, 0.9, r)
    b = b - b.mean() + a[-1]
    y = np.concatenate([a, b])
    tr = filter_trace(y, halflife=220.0, warmup=80, settle=500, z_thresh=4.0)
    return {"y": y, "true_break": n_seg, "phi": tr["phi"], "sigma": tr["sigma"],
            "flags": tr["flags"], "true_phi": (0.2, 0.9)}


def filter_characteristics(lengths: tuple[int, ...] = (1000, 10000, 100000)) -> dict:
    """Memory + per-sample-cost profile of :class:`StochasticFilter` vs stream
    length, plus a reference dtfit ``LSIFilter`` per-sample cost -- evidence the
    filter has the FLAT-memory / BOUNDED-fast characteristics of dtfit's own
    streaming filters (state is independent of stream length; cost per sample does
    not grow)."""
    import sys
    import time

    rng = np.random.default_rng(0)
    rows = []
    for n in lengths:
        f = StochasticFilter()
        y = gen_ar1(n, 0.6, rng)
        t0 = time.perf_counter()
        f.partial_fit(y)
        us = (time.perf_counter() - t0) / n * 1e6
        state = (sys.getsizeof(f._buf) + f._acov.nbytes + f._vacov.nbytes
                 + sys.getsizeof(f.flag_times_))
        rows.append({"stream length N": int(n), "filter state (bytes)": int(state),
                     "us / sample": round(us, 1)})
    ref = float("nan")
    try:
        from dtfit import LSIFilter
        f2 = LSIFilter("a0 + a1*t", "t", window_size=60)
        y = np.cumsum(rng.standard_normal(5000))
        t0 = time.perf_counter()
        for i, v in enumerate(y):
            f2.partial_fit(float(i), float(v))
        ref = round((time.perf_counter() - t0) / y.size * 1e6, 1)
    except Exception:
        pass
    return {"rows": rows, "lsifilter_us_per_sample": ref}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(seeds: int = 8, *, quick: bool = False) -> list[dict]:
    """Run the six recovery experiments + the merged router and return the rows."""
    if quick:
        seeds = max(2, seeds // 2)
    return [
        exp_hurst_aggvar(seeds, n=2048 if quick else 4096),
        exp_hurst_spectral(seeds, n=2048 if quick else 4096),
        exp_ar1(seeds, n=1200 if quick else 1500),
        exp_garch(seeds, n=2500 if quick else 4000),
        exp_cycle(seeds, n=1000 if quick else 1200),
        exp_decompose(seeds, n=500 if quick else 600),
        exp_merged_router(seeds),
        exp_online_filter(max(3, seeds // 2) if quick else seeds),
    ]


def summary(rows: list[dict]) -> str:
    """Render the verdict table as GitHub-flavoured markdown."""
    head = ("| ID | possibility | parameter | dtfit err | baseline err | "
            "verdict |")
    sep = "|----|-------------|-----------|-----------|--------------|---------|"
    lines = [head, sep]
    for r in rows:
        de = "--" if not np.isfinite(r["dt_err"]) else f"{r['dt_err']:.3g}"
        be = "--" if not np.isfinite(r["base_err"]) else f"{r['base_err']:.3g}"
        lines.append(
            f"| {r['id']} | {r['name']} | {r['param']} ({r['metric']}) | "
            f"{de} | {be} | {r['verdict']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import sys

    quick = "--quick" in sys.argv
    rows = run(seeds=6, quick=quick)
    print(summary(rows))
    for r in rows:
        if "extra" in r:
            print(f"  {r['id']} detail: {r['extra']}")
        if r["id"] == "E7":
            print(f"  E7 regime-ID accuracy: {r['accuracy']:.0f}%")
            for rr in r["rows"]:
                print(f"     {rr['process']:<22} {rr['expected']:<14} "
                      f"{rr['detected %']:.0f}%")
        if r["id"] == "E11":
            print(f"  E11 streaming: phi track MAE {r['track_mae']:.3f}, "
                  f"break hit-rate {r['break_hit_rate']:.0f}%, "
                  f"median latency {r['median_latency']:.0f} steps, "
                  f"false alarms {r['false_alarms_per_3000']:.2f}/3000")

    print("\n---- E8 forecast skill (RMSE ratio to random walk; <1 beats RW) ----")
    fs = exp_forecast_skill(seeds=4)
    for rr in fs["rows"]:
        print(" ", rr["process"], {k: round(v, 2) for k, v in rr.items()
                                    if k != "process"})

    print("\n---- E9 real data: USD/UAH 2014-15 ----")
    rd = exp_real_data()
    print(f"  level regime    : {rd['level_regime']}  [{rd['level_components']}]")
    print(f"  returns vol-clust: {rd['returns_vol_clustering']} "
          f"(persistence {rd['returns_vol_persistence']:.3f})")
    print("  |returns| Hurst  : "
          + ", ".join(f"{k}={v:.3f}" for k, v in rd["abs_returns_hurst"].items()))
    print(f"  forecast ({rd['forecast_regime']}) RMSE: "
          + ", ".join(f"{k}={v:.4g}" for k, v in rd["forecast_rmse"].items()))

    print("\n---- E10 real-data gallery (one per regime) ----")
    suite = exp_real_suite()
    for rr in suite["rows"]:
        ratio = rr["merged/RW"]
        ratio_s = "--" if not np.isfinite(ratio) else f"{ratio:.2f}xRW"
        print(f"  {rr['dataset']:<34} n={rr['n']:<5} -> {rr['detected regime']:<22} "
              f"[{rr['components']}]  fc {ratio_s}")
        print(f"      expect {rr['expected']:<22} | {rr['literature']}")
