"""Ground-truth process generators and the real-series catalogue the
stochastic tier's notebooks draw on.

Each generator produces a series with a known parameter, for measuring
whether ``dtfit.stochastic``'s estimators recover it. :data:`ROUTER_CASES`
and :func:`regime_matches` are the router's own applicability map, one
process class per regime; :data:`REAL_SERIES` is the gallery of real
records the notebooks characterize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from dtfit.stochastic import fit_stochastic, hurst_aggvar, hurst_spectral

from . import baselines as bl
from .metrics import metrics
from .paths import data_dir

__all__ = [
    "gen_arfima", "gen_ar1", "gen_garch", "gen_ar2_cycle", "gen_trend_cycle",
    "gen_arp", "ROUTER_CASES", "regime_matches", "RealSeries", "REAL_SERIES",
    "load_series", "suite_horizon", "forecast_skill", "hurst_comparison",
]


# the ground-truth data generators
def gen_arfima(n: int, d: float, rng: np.random.Generator,
               *, ntrunc: int = 1200) -> np.ndarray:
    """ARFIMA(0, d, 0): white noise fractionally integrated to order ``d``.

    Built from the truncated MA(inf) expansion of ``(1 - B)^(-d)``
    (``psi_0 = 1``, ``psi_j = psi_{j-1} (j-1+d)/j``). Exhibits long memory
    with Hurst exponent ``H = d + 1/2``.

    Args:
        n: Length of the returned series, in samples.
        d: Fractional differencing order, ``d`` in ``(-0.5, 0.5)`` for a
            stationary, invertible process; long memory for ``d > 0``.
        rng: Source of the driving white noise.
        ntrunc: Truncation length of the MA(inf) expansion, in samples.

    Returns:
        A 1-D ``float`` array of length ``n``.
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
    """AR(1) / discretely-sampled Ornstein-Uhlenbeck: ``x_t = phi x_{t-1} +
    e_t``.

    Args:
        n: Length of the returned series, in samples.
        phi: AR(1) coefficient (mean-reversion strength); ``|phi| < 1`` for
            a stationary process.
        rng: Source of the driving noise.
        sigma: Innovation standard deviation.
        burn: Burn-in length discarded before the returned samples, in
            samples.

    Returns:
        A 1-D ``float`` array of length ``n``.
    """
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def gen_garch(n: int, omega: float, alpha: float, beta: float,
              rng: np.random.Generator, *, burn: int = 500) -> np.ndarray:
    """GARCH(1,1) returns; volatility persistence is ``alpha + beta``.

    Args:
        n: Length of the returned series, in samples.
        omega: Variance-equation intercept, positive.
        alpha: ARCH coefficient (reaction to the last squared return).
        beta: GARCH coefficient (persistence of past variance);
            ``alpha + beta < 1`` for a stationary variance process.
        rng: Source of the driving innovations.
        burn: Burn-in length discarded before the returned samples, in
            samples.

    Returns:
        A 1-D ``float`` array of length ``n``.
    """
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
    """AR(2) with complex roots: a stochastic pseudo-cycle of the given
    period and damping (root modulus). ``phi1 = 2 r cos(2pi/period)``,
    ``phi2 = -r^2``.

    Args:
        n: Length of the returned series, in samples.
        period: Cycle period, in samples.
        damping: Root modulus ``r`` in ``(0, 1)``; closer to 1 gives a
            slower-decaying, more clearly periodic series.
        rng: Source of the driving noise.
        burn: Burn-in length discarded before the returned samples, in
            samples.

    Returns:
        A 1-D ``float`` array of length ``n``.
    """
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
    """Deterministic trend + cycle plus i.i.d. noise.

    Args:
        n: Number of samples.
        slope: Trend slope, in units of ``y`` per sample.
        period: Cycle period, in samples.
        amp: Cycle amplitude, in units of ``y``.
        noise_sd: Standard deviation of the added i.i.d. Gaussian noise.
        rng: Source of the noise.
        intercept: Trend intercept at ``t = 0``, in units of ``y``.

    Returns:
        ``(t, y)``: the sample times ``0..n-1`` and the series, both 1-D
        arrays of length ``n``.
    """
    t = np.arange(n, dtype=float)
    y = (intercept + slope * t + amp * np.sin(2.0 * np.pi * t / period)
         + rng.normal(0.0, noise_sd, n))
    return t, y


def gen_arp(n: int, phi: tuple[float, ...], rng: np.random.Generator,
            *, sigma: float = 1.0, burn: int = 300) -> np.ndarray:
    """General AR(p): ``x_t = sum_j phi_j x_{t-j} + e_t`` (lag ``1..p``).

    Args:
        n: Length of the returned series, in samples.
        phi: AR coefficients ``(phi_1, ..., phi_p)``, lag 1 first; the
            process is stationary only if every root of the characteristic
            polynomial lies outside the unit circle.
        rng: Source of the driving noise.
        sigma: Innovation standard deviation.
        burn: Burn-in length discarded before the returned samples, in
            samples.

    Returns:
        A 1-D ``float`` array of length ``n``.
    """
    p = len(phi)
    N = n + burn
    e = rng.normal(0.0, sigma, N)
    x = np.zeros(N)
    phi_a = np.asarray(phi, dtype=float)
    for t in range(p, N):
        x[t] = float(phi_a @ x[t - p:t][::-1]) + e[t]
    return x[burn:]


# the router's applicability map: one process class per regime it detects
ROUTER_CASES: list[tuple[str, str, Callable[[int], np.ndarray]]] = [
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


class _RegimeModel(Protocol):
    regime: str
    has_vol_clustering: bool


def regime_matches(model: _RegimeModel, expected: str) -> bool:
    """Does a fitted model's detected regime match an expected label?

    Args:
        model: A fitted ``StochasticModel`` or ``ClassicalStochasticModel``,
            read through ``.regime`` and ``.has_vol_clustering``.
        expected: One of ``"white noise"``, ``"random walk"``,
            ``"mean-reverting"``, ``"long-memory"``, ``"vol-clustering"``,
            ``"cyclical"`` or ``"trend+cycle"``, the router's own
            vocabulary, matching :data:`ROUTER_CASES`'s expected column.

    Returns:
        ``True`` if ``model``'s regime (or, for ``"vol-clustering"``, its
        flag) matches ``expected``; ``False`` for an unrecognized
        ``expected`` too.
    """
    reg = model.regime.lower()
    if expected == "white noise":
        return reg.startswith("white noise")
    if expected == "random walk":
        return reg.startswith("random walk")
    if expected == "mean-reverting":
        return "mean-revert" in reg
    if expected == "long-memory":
        return "long-memory" in reg
    if expected == "vol-clustering":
        return bool(model.has_vol_clustering)
    if expected == "cyclical":
        return any(w in reg for w in ("cyclical", "cycle", "seasonal"))
    if expected == "trend+cycle":
        return "trend+cycle" in reg or "trend+seasonal" in reg
    return False


def load_series(name: str, col: int = 1) -> np.ndarray:
    """Load one column of a bundled real-data CSV.

    Args:
        name: Path relative to :func:`study.paths.data_dir`, e.g.
            ``"usd_uah_2014_2015.csv"`` or ``"ltsf/exchange_rate.csv"``.
        col: Zero-based column index to read.

    Returns:
        A 1-D ``float`` array of the column's values, blank and ``"NA"``
        cells dropped.

    Raises:
        FileNotFoundError: No such file under ``data_dir()``.
    """
    import csv

    rows = list(csv.reader((data_dir() / name).open()))[1:]
    return np.array([float(r[col]) for r in rows if r and r[col] not in ("", "NA")])


def _guarded(
    label: str, fn: Callable[[], np.ndarray],
) -> Callable[[], np.ndarray | None]:
    """Wrap a real-series loader so a missing dependency or data file fails
    only that one catalogue entry: print one line naming the failure and
    return ``None`` instead of letting the exception propagate."""
    def loader() -> np.ndarray | None:
        try:
            return np.asarray(fn(), dtype=float)
        except Exception as exc:
            print(f"{label}: unavailable ({type(exc).__name__}: {exc})")
            return None
    return loader


def _load_nile() -> np.ndarray:
    import statsmodels.api as sm

    return sm.datasets.nile.load_pandas().data["volume"].to_numpy(float)


def _load_sunspots() -> np.ndarray:
    import statsmodels.api as sm

    return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)


def _load_co2() -> np.ndarray:
    import statsmodels.api as sm

    s = sm.datasets.co2.load_pandas().data["co2"]
    return s.interpolate().to_numpy(float)


def _load_gdp() -> np.ndarray:
    import statsmodels.api as sm

    return sm.datasets.macrodata.load_pandas().data["realgdp"].to_numpy(float)


def _load_tbill() -> np.ndarray:
    import statsmodels.api as sm

    return sm.datasets.macrodata.load_pandas().data["tbilrate"].to_numpy(float)


@dataclass(frozen=True)
class RealSeries:
    """One entry of the real-data catalogue :data:`REAL_SERIES`.

    Attributes:
        key: Short identifier, unique within the catalogue.
        title: Human-readable series name for a table or figure caption.
        domain: The field the series comes from, for grouping in a gallery.
        expect: The regime the literature attributes to this series, in the
            router's own vocabulary (fed to :func:`regime_matches`).
        load: Loader returning a finite 1-D ``float`` array, or ``None``
            with one printed line when its source (a missing
            ``statsmodels`` install, or a gitignored CSV under
            ``$DTFIT_DATA``) is unavailable. Never raises.
        lit: One-line citation of the literature result the entry checks
            against.
    """

    key: str
    title: str
    domain: str
    expect: str
    load: Callable[[], np.ndarray | None]
    lit: str


REAL_SERIES: list[RealSeries] = [
    RealSeries("nile", "Nile annual volume (1871-1970)", "hydrology",
               "long memory / trend", _guarded("nile", _load_nile),
               "Hurst's canonical long-memory series (H ~ 0.9)"),
    RealSeries("sunspots", "Sunspot number (1700-2008)", "solar physics",
               "cyclical", _guarded("sunspots", _load_sunspots),
               "the ~11-year solar cycle"),
    RealSeries("co2", "Mauna Loa CO2 (weekly)", "climate", "trend+cycle",
               _guarded("co2", _load_co2), "rising trend + annual cycle"),
    RealSeries("gdp", "US real GDP (quarterly)", "macroeconomics",
               "random walk + drift", _guarded("gdp", _load_gdp),
               "Nelson-Plosser: GDP is a random walk with drift"),
    RealSeries("tbill", "US 3-month T-bill rate (quarterly)",
               "macroeconomics", "random walk / mean-revert",
               _guarded("tbill", _load_tbill),
               "short rates are near-unit-root (highly persistent)"),
    RealSeries("usd_uah", "USD/UAH (2014-15, daily)", "FX", "random walk",
               _guarded("usd_uah",
                        lambda: load_series("usd_uah_2014_2015.csv")),
               "an FX level is a near-random walk; clustering in the "
               "returns"),
    RealSeries("fx_ltsf", "LTSF exchange rate (daily)", "FX", "random walk",
               _guarded("fx_ltsf",
                        lambda: load_series("ltsf/exchange_rate.csv", -1)),
               "FX level near-random walk; long memory in volatility"),
]


def suite_horizon(n: int) -> int:
    """Held-out forecast horizon scaled to a series length.

    Args:
        n: Series length, in samples.

    Returns:
        ``n // 8`` clamped to ``[12, 40]`` samples.
    """
    return int(min(max(n // 8, 12), 40))


def forecast_skill(y: np.ndarray, h: int) -> tuple[dict[str, float], str]:
    """Held-out forecast skill of the merged solution against established
    forecasters.

    Args:
        y: The series, at least ``h + 1`` samples.
        h: Forecast horizon, in samples, held out from the tail of ``y``.

    Returns:
        ``(rmse, regime)``: ``rmse`` maps each method's name (``"dtfit
        merged"``, ``"random walk"``, ``"drift"``, ``"AR(1)"``,
        ``"ARIMA(2,1,2)"``, ``"ETS"``, ``"Theta"``) to its RMSE on the
        held-out tail, the statsmodels-backed methods reading ``nan`` when
        unavailable; ``regime`` is the merged model's detected regime.
    """
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


def _safe(fn: Callable[[], float]) -> float:
    try:
        return float(fn())
    except Exception:
        return float("nan")


def hurst_comparison(y: np.ndarray) -> dict[str, float]:
    """Hurst exponent of ``y`` by every available estimator.

    Args:
        y: The series.

    Returns:
        A dict with keys ``"dtfit spectral"``, ``"dtfit aggvar"``, ``"R/S"``
        and ``"DFA"``, each ``nan`` where that estimator raises.
    """
    return {
        "dtfit spectral": _safe(lambda: hurst_spectral(y)["H"]),
        "dtfit aggvar": _safe(lambda: hurst_aggvar(y)["H"]),
        "R/S": _safe(lambda: bl.hurst_rs(y)),
        "DFA": _safe(lambda: bl.hurst_dfa(y)),
    }
