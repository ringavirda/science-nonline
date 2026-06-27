"""Stochastic-series adaptations -- using dtfit's deterministic fitters on the
*deterministic functionals* of a random process.

dtfit fits a deterministic ``y = f(t; theta)``; a genuinely random series
(economic / financial data, near a martingale) has no such ``f`` -- fitting a
curve to the path is meaningless (the stable forecaster already guards against
it by falling back to persistence). The way to *use* dtfit on such data is to
point it not at the random **path** but at a deterministic **summary** of the
process, whose functional form is known and happens to be exactly the shapes
dtfit excels at:

* the **autocovariance function** of an ARMA / OU process is a sum of damped
  exponentials / damped cosines  ->  exponential-decay / damped-cosine fit;
* the **spectral density** is rational in ``e^{i w}`` and, for a long-memory
  process, a **power law** ``S(f) ~ c f^{-2d}`` near zero  ->  power-law fit;
* the **aggregated-variance** curve of a self-similar process is a power law
  ``Var(block mean at scale m) ~ c m^{2H-2}``  ->  power-law fit;
* the **conditional mean** (trend + cycle) is a structural curve  ->  the
  stable LSI trend + oscillatory-recipe cycle, leaving the stochastic residual.

So each estimator below recovers a *parameter of a stochastic model* (Hurst /
fractional-integration order, mean-reversion speed, volatility persistence,
cycle period, trend) by feeding a deterministic functional of the data to
:func:`dtfit.fit_lsi` / :func:`dtfit.fit_eac`.

The merged entry point :func:`fit_stochastic` composes these routes behind
significance gates into a single :class:`StochasticModel` that characterizes,
forecasts, and *generates* an arbitrary series; the streaming counterpart is
:class:`~dtfit.stochastic.StochasticFilter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, cast

import numpy as np

from dtfit.methods import fit_lsi, fit_eac


__all__ = [
    "sample_acf",
    "hurst_aggvar",
    "hurst_spectral",
    "ar1_reversion",
    "garch_persistence",
    "cycle_period",
    "decompose_trend_cycle",
    "StochasticModel",
    "fit_stochastic",
]


# --------------------------------------------------------------------------- #
# shared: sample autocorrelation (the deterministic functional most of these
# estimators feed to dtfit).
# --------------------------------------------------------------------------- #
def sample_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Biased sample autocorrelation ``rho[0..nlags]`` (``rho[0] == 1``).

    The biased (divide-by-``n``) estimator is used deliberately: it is the
    positive-definite, lower-variance choice that the moment estimators below
    want, and it tapers the noisy long-lag tail toward zero rather than letting
    it explode.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    denom = float(x @ x)
    if denom == 0.0:
        out = np.zeros(nlags + 1)
        out[0] = 1.0
        return out
    # FFT autocovariance (Wiener-Khinchin): O(n log n) vs the O(nlags*n) loop,
    # numerically identical to the biased estimator (the 1/n cancels in the ratio).
    m = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(x, m)
    acov = np.fft.irfft(f * np.conj(f), m)[: nlags + 1]
    return acov / acov[0]


def _loglog_slope(lx: np.ndarray, ly: np.ndarray, *, method: str) -> float:
    """Slope ``b`` of ``ly = a + b*lx``.

    ``method="lsi"`` fits the line with dtfit's reconditioned Legendre LSI
    (orthogonal-basis least squares); ``method="ols"`` is the plain
    ``numpy.polyfit`` baseline the domain experiment compares against.
    """
    if method == "ols":
        return float(np.polyfit(lx, ly, 1)[0])
    r = fit_lsi(lx, ly, "a + b*m", "m", k_star=1, filter_data=False)
    # sympy name-sorted order -> [a, b]
    return float(r.coeffs[1])


# --------------------------------------------------------------------------- #
# long memory / self-similarity: Hurst exponent H (fractional-integration
# order d = H - 1/2).
# --------------------------------------------------------------------------- #
def hurst_aggvar(
    x: np.ndarray,
    *,
    n_scales: int = 14,
    min_block: int = 2,
    method: str = "lsi",
) -> dict[str, float]:
    """Hurst exponent via the **aggregated-variance** power law.

    Block-average the series at geometrically spaced scales ``m``; for a
    self-similar process the block-mean variance scales as
    ``Var(m) ~ c * m^(2H - 2)``. The exponent is read from a power-law fit of
    ``Var`` against ``m`` -- ``method="lsi"`` / ``"ols"`` fit it in log-log
    space (slope ``= 2H - 2``); ``method="eac"`` fits the power law
    ``c*m**b`` directly in linear space with the equal-areas criterion.

    Returns ``{"H", "slope", "d"}`` (``d = H - 1/2``).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    max_block = max(min_block * 2, n // 4)
    blocks = np.unique(
        np.round(np.geomspace(min_block, max_block, n_scales)).astype(int)
    )
    blocks = blocks[blocks >= 2]

    ms, vs = [], []
    for m in blocks:
        nb = n // m
        if nb < 8:  # need enough blocks for a stable per-scale variance
            continue
        bmean = x[: nb * m].reshape(nb, m).mean(axis=1)
        v = float(bmean.var())
        if v > 0:
            ms.append(float(m))
            vs.append(v)
    ms_a = np.asarray(ms)
    vs_a = np.asarray(vs)
    if ms_a.size < 3:
        raise RuntimeError("too few usable scales for aggregated-variance Hurst")

    if method == "eac":
        # nonlinear power law in linear space (no log transform)
        b0 = float(np.polyfit(np.log(ms_a), np.log(vs_a), 1)[0])
        r = fit_eac(ms_a, vs_a, "c*m**b", "m",
                    p0=[float(vs_a[0]), b0],
                    bounds=([1e-12, -2.0], [1e6, 0.0]))
        slope = float(r.coeffs[0])  # sorted names [b, c] -> b first
    else:
        slope = _loglog_slope(np.log(ms_a), np.log(vs_a), method=method)

    H = (slope + 2.0) / 2.0
    return {"H": float(np.clip(H, 0.0, 1.0)), "slope": float(slope),
            "d": float(np.clip(H, 0.0, 1.0) - 0.5)}


def hurst_spectral(
    x: np.ndarray,
    *,
    n_freq: int | None = None,
    method: str = "lsi",
) -> dict[str, float]:
    """Hurst / fractional-integration order via the **low-frequency spectrum**.

    The log-periodogram of a long-memory process is ``log S(f) = const - 2d
    log f + noise`` for small ``f`` (the GPH regression). The slope of the
    low-frequency log-log periodogram gives ``d = H - 1/2``. dtfit's role
    (``method="lsi"``) is the orthogonal-basis line fit, with its Savitzky-Golay
    pre-filter taming the periodogram's chi-squared scatter; ``method="ols"`` is
    the plain GPH baseline.

    Returns ``{"H", "d", "slope"}`` (``slope = -2d``).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    per = (np.abs(np.fft.rfft(x)) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0)
    if n_freq is None:
        n_freq = max(8, int(n ** 0.6))
    hi = min(1 + n_freq, freqs.size)
    f = freqs[1:hi]
    p = per[1:hi]
    keep = p > 0
    f, p = f[keep], p[keep]
    if f.size < 3:
        raise RuntimeError("too few usable frequencies for spectral Hurst")

    lf, lp = np.log(f), np.log(p)
    if method == "ols":
        slope = float(np.polyfit(lf, lp, 1)[0])
    else:
        # smoothing on: denoise the log-periodogram before the slope read-out
        r = fit_lsi(lf, lp, "a + b*m", "m", k_star=1, filter_data=True)
        slope = float(r.coeffs[1])
    d = -slope / 2.0
    H = d + 0.5
    return {"H": float(H), "d": float(d), "slope": float(slope)}


# --------------------------------------------------------------------------- #
# mean reversion: OU / AR(1) -- ACF is a single exponential exp(-k/tau).
# --------------------------------------------------------------------------- #
def ar1_reversion(
    x: np.ndarray,
    *,
    nlags: int | None = None,
    method: str = "lsi",
) -> dict[str, float]:
    """Mean-reversion speed from an **exponential fit to the ACF**.

    An OU / AR(1) process has ``rho(k) = phi^k = exp(-k/tau)``. Fitting a
    decaying exponential to the sample ACF (``method="lsi"`` / ``"eac"``) reads
    off the AR(1) coefficient ``phi`` and the reversion time ``tau`` using many
    lags at once (the integral fitters average the per-lag ACF noise), where the
    plain ``method="acf1"`` baseline just takes the lag-1 autocorrelation.

    Returns ``{"phi", "tau", "halflife"}``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if nlags is None:
        nlags = int(np.clip(n // 4, 10, 60))
    acf = sample_acf(x, nlags)

    if method == "acf1":
        phi = float(np.clip(acf[1], 1e-6, 0.999999))
    else:
        phi = _phi_from_acf(acf, n, method=method)
    tau = -1.0 / np.log(phi) if 0.0 < phi < 1.0 else np.inf
    halflife = tau * np.log(2.0) if np.isfinite(tau) else np.inf
    return {"phi": phi, "tau": float(tau), "halflife": float(halflife)}


def _phi_from_acf(acf: np.ndarray, n: int, *, method: str = "lsi") -> float:
    """AR(1) ``phi`` from a **decaying-exponential dtfit fit to the ACF**.

    The shared core of the batch :func:`ar1_reversion` and the streaming
    :class:`~dtfit.stochastic.StochasticFilter`: both feed
    their (sample or EWMA) ACF here so the persistence is read off the same
    ``fit_lsi`` / ``fit_eac`` exponential fit, not a single-lag shortcut.

    Restrict the fit to lags where the ACF is still above the white-noise band
    (~``2/sqrt(n)``); beyond it the ACF is pure sampling noise and only drags the
    decay rate. The amplitude is anchored (``exp(-g*k)``, no free ``A``): an AR(1)
    ACF is exactly ``phi^k`` and passes through 1 at lag 0, so freeing ``A`` lets
    it absorb the lag-1 noise and badly biases the fast-decay case.
    """
    nlags = acf.size - 1
    band = max(0.05, 2.0 / np.sqrt(max(n, 1)))
    above = np.where(np.abs(acf[1:]) >= band)[0]
    keff = int(above[-1] + 1) if above.size else nlags
    keff = int(np.clip(keff, 5, nlags))
    k = np.arange(1, keff + 1, dtype=float)
    y = acf[1: keff + 1]
    fitter = fit_eac if method == "eac" else fit_lsi
    r = fitter(k, y, "exp(-g*k)", "k", p0=[0.1])
    return float(np.exp(-abs(float(r.coeffs[0]))))


# --------------------------------------------------------------------------- #
# volatility clustering: GARCH(1,1) persistence alpha+beta -- ACF of squared
# returns decays geometrically (alpha+beta)^k.
# --------------------------------------------------------------------------- #
def garch_persistence(
    returns: np.ndarray,
    *,
    nlags: int | None = None,
    method: str = "lsi",
    use: str = "square",
) -> dict[str, float]:
    """Volatility persistence from the **ACF of squared (or |.|) returns**.

    For a GARCH(1,1) the autocorrelation of the squared returns decays
    geometrically with ratio ``alpha + beta`` (the persistence). Fitting a
    decaying exponential to that ACF with dtfit recovers it without running a
    full GARCH likelihood. ``use="abs"`` fits the ACF of absolute returns (often
    cleaner empirically).

    Returns ``{"persistence", "tau"}``.
    """
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    z = np.abs(r) if use == "abs" else r ** 2
    n = z.size
    if nlags is None:
        nlags = int(np.clip(n // 8, 10, 50))
    acf = sample_acf(z, nlags)
    persistence = _persistence_from_acf(acf, method=method)
    tau = -1.0 / np.log(persistence) if 0.0 < persistence < 1.0 else np.inf
    return {"persistence": persistence, "tau": float(tau)}


def _persistence_from_acf(acf: np.ndarray, *, method: str = "lsi") -> float:
    """Volatility persistence from a **decaying-exponential dtfit fit to the ACF
    of |returns| / squared returns**. Shared by the batch :func:`garch_persistence`
    and the streaming filter (which feeds the EWMA ACF of its |residual|)."""
    nlags = acf.size - 1
    k = np.arange(1, nlags + 1, dtype=float)
    fitter = fit_eac if method == "eac" else fit_lsi
    rr = fitter(k, acf[1:], "A*exp(-g*k)", "k", p0=[float(acf[1]) or 0.2, 0.1])
    g = float(rr.coeffs[1])
    return float(np.clip(np.exp(-abs(g)), 0.0, 0.9999))


# --------------------------------------------------------------------------- #
# pseudo-cycle: AR(2) with complex roots -- ACF is a damped cosine.
# --------------------------------------------------------------------------- #
def cycle_period(
    x: np.ndarray,
    *,
    nlags: int | None = None,
) -> dict[str, float]:
    """Dominant cycle period from a **damped-cosine fit to the ACF**.

    An AR(2) with complex roots (a stochastic pseudo-cycle, e.g. a business
    cycle) has ``rho(k) = r^k cos(w k + phi)``. dtfit's oscillatory recipe
    (FFT-seeded angular frequency, no smoothing, raised spectral order) fits that
    damped cosine to the sample ACF and returns the cycle period ``2*pi/w`` and
    the damping ``r``.

    Returns ``{"period", "w", "damping"}``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if nlags is None:
        nlags = int(np.clip(n // 2, 20, 100))
    acf = sample_acf(x, nlags)
    return _cycle_from_acf(acf)


def _cycle_from_acf(acf: np.ndarray) -> dict[str, float]:
    """Dominant cycle from a **damped-cosine dtfit fit to the ACF**. Shared by the
    batch :func:`cycle_period` and the streaming filter; the angular frequency is
    FFT-seeded from the ACF's own spectral peak (``dtfit.fit_lsi`` oscillatory
    recipe), so only the damping and frequency are read back."""
    nlags = acf.size - 1
    k = np.arange(nlags + 1, dtype=float)
    w0 = _acf_peak_w(acf)
    r = fit_lsi(k, acf, "A*exp(-g*k)*cos(w*k + p)", "k",
                freq_param="w", p0=[1.0, 0.05, w0, 0.0])
    # sorted names [A, g, p, w] -- only the damping g and frequency w are needed
    g, w = float(r.coeffs[1]), abs(float(r.coeffs[3]))
    # A near-zero recovered frequency means "no cycle found" -- report it as such
    # rather than an enormous spurious period.
    period = 2.0 * np.pi / w if w > (2.0 * np.pi / (2.0 * nlags)) else float("inf")
    return {"period": float(period), "w": float(w),
            "damping": float(np.exp(-abs(g)))}


def _acf_peak_w(acf: np.ndarray) -> float:
    """Angular frequency of the dominant peak in the FFT of the ACF (the seed
    for the damped-cosine fit). Falls back to a quarter-Nyquist guess."""
    a = np.asarray(acf, dtype=float)
    a = a - a.mean()
    spec = np.abs(np.fft.rfft(a))
    if spec.size > 1:
        spec[0] = 0.0
    freqs = np.fft.rfftfreq(a.size, d=1.0)
    kpk = int(np.argmax(spec)) if spec.size else 0
    f = float(freqs[kpk]) if kpk > 0 else 0.0
    return 2.0 * np.pi * f if f > 0 else (np.pi / 2.0)


# --------------------------------------------------------------------------- #
# structural decomposition: trend + cycle + stochastic residual.
# --------------------------------------------------------------------------- #
def decompose_trend_cycle(
    t: np.ndarray,
    y: np.ndarray,
    *,
    trend_deg: int = 1,
    with_cycle: bool = True,
) -> dict[str, object]:
    """Split ``y`` into a deterministic **trend + cycle** (fit by dtfit) and a
    **stochastic residual** (left for a noise model).

    The trend is a low-order LSI polynomial; the cycle is the LSI oscillatory
    recipe fit to the de-trended residual. This is the honest way dtfit handles
    economic data: it claims only the structured part and hands the rest to
    persistence / a stochastic model.

    Returns a dict with the fitted ``trend``/``cycle``/``residual`` arrays, the
    recovered ``slope`` and ``period``/``amp``, and a ``forecast(h, dt)`` closure
    that extrapolates trend + cycle ``h`` steps ahead.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    terms = " + ".join(f"a{i}*x**{i}" for i in range(trend_deg + 1))
    rt = fit_lsi(t, y, terms, "x", k_star=max(2, trend_deg), filter_data=True)
    trend = np.asarray(rt.model(t), dtype=float)
    if np.ndim(trend) == 0:
        trend = np.full_like(t, float(trend))
    # ascending coeff order a0,a1,... after sympy name sort (a0 < a1 < ...)
    tcoeffs = [float(c) for c in rt.coeffs]
    slope = tcoeffs[1] if trend_deg >= 1 else 0.0

    resid = y - trend
    period = amp = w = phase = float("nan")
    a_sig = 0.0  # signed cycle amplitude used by the forecast closure
    cycle = np.zeros_like(t)
    if with_cycle and t.size >= 8:
        rc = fit_lsi(t, resid, "A*sin(w*x + p)", "x", freq_param="w",
                     p0=[float(np.std(resid)) + 1e-3, 0.5, 0.0])
        cycle = np.asarray(rc.model(t), dtype=float)
        if np.ndim(cycle) == 0:
            cycle = np.full_like(t, float(cycle))
        a_sig, phase, w = float(rc.coeffs[0]), float(rc.coeffs[1]), float(rc.coeffs[2])
        amp = abs(a_sig)
        period = 2.0 * np.pi / w if w > 1e-9 else float("nan")
    noise = resid - cycle

    def forecast(h: int, dt: float | None = None) -> np.ndarray:
        step = float(t[1] - t[0]) if (dt is None and t.size > 1) else (dt or 1.0)
        tf = t[-1] + step * np.arange(1, h + 1)
        tr = np.asarray(rt.model(tf), dtype=float)
        if np.ndim(tr) == 0:
            tr = np.full_like(tf, float(tr))
        cy = a_sig * np.sin(w * tf + phase) if np.isfinite(w) else np.zeros_like(tf)
        return tr + cy

    return {
        "trend": trend, "cycle": cycle, "residual": noise,
        "slope": float(slope), "period": float(period), "amp": float(amp),
        "noise_std": float(np.std(noise)), "forecast": forecast,
    }


# --------------------------------------------------------------------------- #
# THE MERGED SOLUTION -- one entry point that characterizes an arbitrary
# stochastic series across all the routes above and forecasts it.
# --------------------------------------------------------------------------- #
def _bind_forecaster(
    fn: Callable[..., Any], y: np.ndarray
) -> Callable[[int], np.ndarray]:
    """Bind a ``(train, h) -> array`` forecaster to the fitted series, yielding the
    ``h -> array`` closure stored on the model (a typed helper rather than a
    default-arg lambda, so the captured series is explicit)."""
    def _f(h: int) -> np.ndarray:
        return np.asarray(fn(y, h), dtype=float)
    return _f


def _ols_line(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS line ``y ~ a + b*t``; returns ``(slope, intercept, |t-stat of slope|)``."""
    n = t.size
    tm, ym = t.mean(), y.mean()
    st = t - tm
    denom = float(st @ st)
    if denom <= 0 or n < 3:
        return 0.0, float(ym), 0.0
    slope = float(st @ (y - ym)) / denom
    intercept = float(ym - slope * tm)
    resid = y - (intercept + slope * t)
    s2 = float(resid @ resid) / max(1, n - 2)
    se = np.sqrt(s2 / denom) if s2 > 0 else 0.0
    tstat = abs(slope) / se if se > 0 else (np.inf if slope != 0 else 0.0)
    return slope, intercept, float(tstat)


# --------------------------------------------------------------------------- #
# Vendored augmented Dickey-Fuller unit-root test -- NO statsmodels dependency.
#
# The unit-root gate is the load-bearing guard of the whole pipeline ("don't fit
# trend / cycle / long memory to a wandering random-walk level" -- the classic
# spurious regression). It was the one piece that needed statsmodels, which the
# stable ``dtfit`` wheel deliberately does not carry. This reproduces
# ``statsmodels.adfuller(y, regression="ct", autolag="AIC")`` to machine
# precision (cross-checked in the test suite) in pure numpy, so the gate keeps
# its quality with zero new dependency -- clearing the one hard blocker to ever
# promoting this module. The p-value is MacKinnon's (1994) approximate asymptotic
# surface for the constant+trend (``ct``) regression with a single I(1) series
# (N = 1); the constants below are his published response-surface coefficients.
# --------------------------------------------------------------------------- #
_ADF_TAU_MAX = 0.7         # tau above this -> p = 1 (clearly stationary side cap)
_ADF_TAU_MIN = -16.18      # tau below this -> p = 0
_ADF_TAU_STAR = -2.89      # split between the small-p and large-p polynomials
_ADF_SMALLP = (3.2512, 1.6047, 0.049588)               # Phi^-1(p) = sum c_i tau^i
_ADF_LARGEP = (2.5261, 0.61654, -0.37956, -0.060285)


def _adf_pvalue(tau: float) -> float:
    """MacKinnon (1994) approximate p-value for an ADF ``tau`` stat (ct, N=1)."""
    if tau > _ADF_TAU_MAX:
        return 1.0
    if tau < _ADF_TAU_MIN:
        return 0.0
    from scipy.stats import norm
    coef = _ADF_SMALLP if tau <= _ADF_TAU_STAR else _ADF_LARGEP
    return float(norm.cdf(np.polyval(coef[::-1], tau)))


def _adf_design(x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """ADF (ct) regression design at ``lag`` difference lags: the response is the
    difference ``dy_t`` and the columns are ``[y_{t-1}, dy_{t-1}, .., dy_{t-lag},
    1, t]`` (the level lag first, so its coefficient is the gamma being tested)."""
    dx = np.diff(x)
    nobs = dx.size - lag
    cols = [x[lag:lag + nobs]]                          # level lag y_{t-1}
    for j in range(1, lag + 1):
        cols.append(dx[lag - j:lag - j + nobs])         # dy_{t-j}
    cols.append(np.ones(nobs))                          # const
    cols.append(np.arange(1, nobs + 1, dtype=float))    # linear trend
    return dx[lag:], np.column_stack(cols)


def _adf_tau(x: np.ndarray, maxlag: int | None = None) -> float:
    """ADF ``tau`` statistic with AIC lag selection (ct regression).

    Matches statsmodels: AIC is compared on the fixed ``maxlag`` sample, then the
    chosen lag is refit on its own (larger) sample for the reported t-stat. The
    Schwert ``maxlag`` is CAPPED (``min(12*(n/100)^.25, 12, n//3)``) -- the cap
    keeps the unit-root verdict while cutting the lag search several-fold (e.g.
    the Nile stays trend-stationary / long-memory, not flipped to a random walk).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if maxlag is None:
        maxlag = int(min(12 * (n / 100.0) ** 0.25, 12, n // 3))
    maxlag = max(0, int(maxlag))
    # AIC selection on the FIXED maxlag sample
    y0, x0 = _adf_design(x, maxlag)
    nobs = y0.size
    trend2 = x0[:, -2:]
    best_ic, best_lag = np.inf, 0
    for lag in range(0, maxlag + 1):
        cols = np.column_stack([x0[:, :lag + 1], trend2])
        beta, *_ = np.linalg.lstsq(cols, y0, rcond=None)
        resid = y0 - cols @ beta
        ssr = float(resid @ resid)
        if ssr <= 0:
            ic = -np.inf
        else:
            llf = -nobs / 2.0 * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1.0)
            ic = -2.0 * llf + 2.0 * cols.shape[1]
        if ic < best_ic:
            best_ic, best_lag = ic, lag
    # final refit at the chosen lag, on its own sample; tau = t-stat on the level lag
    y1, x1 = _adf_design(x, best_lag)
    beta, *_ = np.linalg.lstsq(x1, y1, rcond=None)
    resid = y1 - x1 @ beta
    dof = y1.size - x1.shape[1]
    if dof <= 0:
        return 0.0
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(x1.T @ x1)
    se0 = np.sqrt(s2 * xtx_inv[0, 0])
    return float(beta[0] / se0) if se0 > 0 else 0.0


def _is_nonstationary(y: np.ndarray, *, alpha: float = 0.05) -> bool:
    """Unit-root gate: ``True`` when the series is I(1) (a random walk), so it must
    be *differenced* rather than have level structure fitted to it.

    Uses the vendored augmented Dickey-Fuller test with a constant+trend
    regression (so a genuine trend-stationary series is *not* mislabelled a unit
    root); falls back to a near-unit AR(1) coefficient on the level only if the
    regression is degenerate.
    """
    n = y.size
    if n < 12:
        return False
    try:
        return _adf_pvalue(_adf_tau(np.asarray(y, dtype=float))) > alpha
    except Exception:
        b = float(np.polyfit(y[:-1], y[1:], 1)[0])
        return b > 0.98


def _cycle_strength(y: np.ndarray, *, min_period: int = 4) -> tuple[float, float]:
    """Dominant interior spectral peak of a linearly-detrended series ->
    ``(period, strength)``; ``strength`` is the peak's share of detrended power."""
    n = y.size
    if n < 2 * min_period:
        return float("nan"), 0.0
    tt = np.arange(n)
    yc = y - np.polyval(np.polyfit(tt, y, 1), tt)
    spec = np.abs(np.fft.rfft(yc)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0
    valid = (freqs > 1.0 / n) & (freqs <= 1.0 / min_period)
    tot = float(spec[1:].sum())
    if not valid.any() or tot == 0:
        return float("nan"), 0.0
    k = int(np.argmax(np.where(valid, spec, 0.0)))
    strength = float(spec[k] / tot)
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else float("nan")
    return period, strength


def _seasonal_design(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Fourier design matrix: ``n_harmonics`` cos/sin pairs at the seasonal
    frequency and its harmonics (so non-sinusoidal seasonal shapes -- a sawtooth
    trend in CO2, the pulse of the sunspot cycle -- are representable)."""
    cols = []
    for j in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * j / period
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    return np.column_stack(cols)


def _fit_seasonal(t: np.ndarray, y: np.ndarray, period: float,
                  max_harmonics: int) -> tuple[int, np.ndarray]:
    """Fit a multi-harmonic Fourier seasonal model by **linear** least squares,
    picking the number of harmonics by BIC (fewer harmonics generalize better, so
    the forecast does not overfit the seasonal shape). Returns ``(K, coef)``."""
    n = y.size
    best = None
    for k in range(1, max_harmonics + 1):
        if 2 * k >= n:
            break
        x = _seasonal_design(t, period, k)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        rss = float(resid @ resid)
        bic = n * np.log(rss / n + 1e-12) + (2 * k) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, coef)
    assert best is not None
    return best[1], best[2]


@dataclass
class StochasticModel:
    """A unified second-order characterization of a stochastic series.

    Produced by :func:`fit_stochastic`. Records which structural components were
    *detected* (each behind a significance gate, so white noise yields none), the
    recovered parameter of each, the primary ``regime`` label for the level
    dynamics, and a :meth:`forecast` that composes the deterministic mean with the
    stochastic mean-reversion and emits regime-appropriate prediction intervals.
    """

    n: int
    level: float
    # deterministic mean (1st order)
    trend_slope: float
    has_trend: bool
    cycle_period: float
    cycle_amp: float
    has_cycle: bool
    n_harmonics: int
    seasonal: bool
    # stochastic structure (2nd order)
    hurst: float
    has_long_memory: bool
    ar1_phi: float
    has_mean_reversion: bool
    vol_persistence: float
    has_vol_clustering: bool
    sigma: float          # one-step innovation std (level)
    sigma_walk: float     # std of first differences (random-walk scale)
    components: tuple[str, ...]
    regime: str
    forecaster_name: str   # the candidate model chosen by backtest selection
    _forecaster: Callable[[int], np.ndarray] = field(
        repr=False, default=lambda h: np.zeros(h))
    # deterministic mean (trend + multi-harmonic seasonal) over a time axis --
    # captured so :meth:`simulate` can regenerate the structured part of the
    # series. The default (flat zero) is used by the unit-root branch, whose
    # mean is the integrated random walk, not a function of ``t``.
    _mean_fn: Callable[[np.ndarray], np.ndarray] = field(
        repr=False,
        default=lambda t: np.zeros_like(np.asarray(t, dtype=float)))

    def fingerprint(self) -> dict[str, object]:
        """The detected structure as a flat ``{name: value}`` dict (for tables)."""
        return {
            "regime": self.regime,
            "components": ", ".join(self.components),
            "trend slope": self.trend_slope if self.has_trend else float("nan"),
            "cycle period": self.cycle_period if self.has_cycle else float("nan"),
            "Hurst H": self.hurst if self.has_long_memory else float("nan"),
            "AR(1) phi": self.ar1_phi if self.has_mean_reversion else float("nan"),
            "vol persistence": self.vol_persistence if self.has_vol_clustering
            else float("nan"),
            "forecaster": self.forecaster_name,
        }

    def summary(self) -> str:
        lines = [f"StochasticModel  regime={self.regime!r}  n={self.n}",
                 f"  components: {', '.join(self.components)}"]
        if self.has_trend:
            lines.append(f"  trend       slope = {self.trend_slope:.4g}")
        if self.has_cycle:
            kind = "seasonal" if self.seasonal else "cycle"
            lines.append(f"  {kind:<11} period = {self.cycle_period:.4g} "
                         f"({self.n_harmonics} harmonic(s))")
        if self.has_long_memory:
            lines.append(f"  long memory H = {self.hurst:.3f} (d = {self.hurst - 0.5:.3f})")
        if self.has_mean_reversion:
            tau = -1.0 / np.log(self.ar1_phi) if 0 < self.ar1_phi < 1 else np.inf
            lines.append(f"  mean revert phi = {self.ar1_phi:.3f} (tau = {tau:.1f})")
        if self.has_vol_clustering:
            lines.append(f"  volatility  persistence = {self.vol_persistence:.3f}")
        lines.append(f"  innovation sigma = {self.sigma:.4g}")
        lines.append(f"  forecaster: {self.forecaster_name}")
        return "\n".join(lines)

    def forecast(self, h: int, *, return_conf_int: bool = False, alpha: float = 0.05):
        """Forecast ``h`` steps with the **backtest-selected** forecaster
        (:attr:`forecaster_name`). With ``return_conf_int`` also returns
        ``(lower, upper)`` bands whose growth matches that forecaster (bounded for
        mean reversion, ``h^(2H)`` for long memory, ``~sqrt(h)`` for a random
        walk / drift)."""
        steps = np.arange(1, h + 1, dtype=float)
        point = np.asarray(self._forecaster(h), dtype=float)
        if not return_conf_int:
            return point
        from scipy.stats import norm
        name = self.forecaster_name
        if name == "mean-reversion" and 0.0 < self.ar1_phi < 1.0:
            var = (self.sigma ** 2) * (1.0 - self.ar1_phi ** (2 * steps)) \
                / (1.0 - self.ar1_phi ** 2)
        elif name in ("random walk", "drift"):
            var = (self.sigma_walk ** 2) * steps
        elif self.has_long_memory:
            var = (self.sigma ** 2) * steps ** (2.0 * self.hurst)
        else:                                   # trend / seasonal
            var = (self.sigma ** 2) * steps
        z = float(norm.ppf(1.0 - alpha / 2.0))
        sd = np.sqrt(np.clip(var, 0.0, None))
        return point, point - z * sd, point + z * sd

    def simulate(self, n: int | None = None, *, seed: int | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a fresh **realization** of length ``n`` from the fitted model.

        This is what makes :class:`StochasticModel` a *generative* model -- the
        tunable simulator of "typical stochastic data" -- not merely a
        forecaster. It composes the detected deterministic mean (trend +
        multi-harmonic seasonal) with a stochastic residual drawn to match the
        detected **second-order regime**:

        * **unit-root** -> an integrated random walk (drift + innovations,
          optionally with GARCH volatility clustering on the increments);
        * **mean-reverting** -> a stationary AR(1) residual at the recovered phi;
        * **long-memory** -> fractional Gaussian noise at the recovered Hurst H
          (spectral synthesis, ``S(f) ~ |f|^{1-2H}``);
        * **vol-clustering** -> a GARCH(1,1) residual at the recovered
          persistence;
        * **white noise** -> i.i.d. Gaussian innovations.

        Re-fitting the simulated path recovers the same regime and parameters
        (the round-trip test), which is the honest validation that the model is a
        faithful generator of the process it claims to characterize.

        Args:
            n: Length of the realization (defaults to the fitted ``n``).
            seed / rng: Randomness control (``rng`` takes precedence).
        """
        n = int(self.n if n is None else n)
        if rng is None:
            rng = np.random.default_rng(seed)
        sig = (self.sigma if np.isfinite(self.sigma) and self.sigma > 0.0
               else (self.sigma_walk if np.isfinite(self.sigma_walk)
                     and self.sigma_walk > 0.0 else 1.0))
        # unit-root level: integrate drift + innovations (the mean is the walk).
        if "unit-root" in self.components:
            if self.has_vol_clustering and np.isfinite(self.vol_persistence):
                steps = _sim_garch(n, self.vol_persistence, sig, rng)
            else:
                steps = rng.standard_normal(n) * sig
            drift = self.trend_slope if np.isfinite(self.trend_slope) else 0.0
            return self.level + np.cumsum(drift + steps)
        # stationary / trend-stationary: deterministic mean + a regime residual.
        t = np.arange(n, dtype=float)
        mean = np.asarray(self._mean_fn(t), dtype=float)
        if mean.shape != t.shape:
            mean = np.full(n, float(self.level))
        # Residual regime in the SAME dominance order the regime label uses
        # (long memory subsumes a short-range AR(1), so it is drawn first -- a
        # long-memory series also trips the mean-reversion flag, but simulating it
        # as a bare AR(1) would erase the long memory and break the round-trip).
        if self.has_long_memory and np.isfinite(self.hurst):
            resid = _sim_long_memory(n, self.hurst, sig, rng)
        elif self.has_mean_reversion and 0.0 < self.ar1_phi < 1.0:
            resid = _sim_ar1(n, self.ar1_phi, sig, rng)
        elif self.has_vol_clustering and np.isfinite(self.vol_persistence):
            resid = _sim_garch(n, self.vol_persistence, sig, rng)
        else:
            resid = rng.standard_normal(n) * sig
        return mean + resid


def fit_stochastic(
    y: np.ndarray,
    t: np.ndarray | None = None,
    *,
    period: float | None = None,
    max_harmonics: int = 4,
    forecaster: object = "auto",
    trend_t: float = 3.0,
    cycle_strength: float = 0.08,
    min_cycles: float = 2.5,
    lm_hurst: float = 0.68,
    mr_phi: float = 0.15,
    vol_persist: float = 0.60,
) -> StochasticModel:
    """**The merged stochastic-series solution.** Characterize an arbitrary series
    across every route at once and return a single coherent model.

    The routes are not run blindly; they are composed in the order the second-order
    theory dictates, each behind a significance gate:

    1. **deterministic mean** -- an LSI trend (kept only if its slope is
       significant, ``|t| > trend_t``) and an LSI cycle (kept only if a genuine
       interior spectral peak carries ``> cycle_strength`` of the power and repeats
       ``>= min_cycles`` times);
    2. **whiten** -- fit an AR(1) to the residual; its innovations are tested for
    3. **long memory** -- the spectral Hurst of the *innovations* (so a
       near-unit-root AR(1), whose innovations are white, is *not* mislabelled as
       long memory; declared when ``H > lm_hurst``);
    4. **mean reversion** -- the AR(1) coefficient, kept when
       ``mr_phi < phi < 0.99`` and the lag-1 ACF is significant;
    5. **volatility clustering** -- the persistence of the ``|residual|`` ACF, kept
       when significant and ``> vol_persist``.

    A series with none of these gates open is reported as ``regime="white noise /
    random walk"`` with no components -- the honest "no structure" verdict.

    **Forecasting** is by backtest model selection: a regime-informed set of
    candidate forecasters (random walk, drift, mean reversion, local-slope trend,
    multi-harmonic seasonal continuation) is rolling-origin backtested and the
    **RMSE-optimal** one is chosen (defaulting to the random walk when nothing
    beats it). The choice is recorded in :attr:`StochasticModel.forecaster_name`.

    Args:
        y: The series. ``t`` defaults to ``0..n-1`` (uniform spacing).
        period: A seasonal period to use (else it is detected from the spectrum).
        max_harmonics: Cap on the Fourier harmonics of the seasonal component
            (the actual count is chosen by BIC).
        forecaster: How to forecast. ``"auto"`` (default) backtest-selects the
            RMSE-optimal candidate; a built-in name (one of :data:`FORECASTERS`)
            forces that model; a callable ``(train, h) -> array`` is used
            directly; a list of names / callables / ``(name, fn)`` pairs is a
            custom candidate set to backtest-select among.
        trend_t, cycle_strength, min_cycles, lm_hurst, mr_phi, vol_persist:
            Detection gates (see above).

    Returns:
        A :class:`StochasticModel` with the detected components, parameters and a
        backtest-selected :meth:`~StochasticModel.forecast`.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    t = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
    level = float(y.mean())
    sigma_walk = float(np.std(np.diff(y))) if n > 1 else 0.0
    band = 2.0 / np.sqrt(max(n, 1))

    # -- Stage 0: unit-root gate. An I(1) level (random walk) must be DIFFERENCED;
    # fitting a trend / cycle / long memory to its wandering level is the classic
    # spurious regression. Characterize the increments instead -- which is also
    # where a financial series carries its structure (returns = the increments).
    # EXEMPTION: a strongly cyclical series also has AR roots near the unit circle
    # -- but at the cycle frequency, not at f=0 -- so ADF mis-flags it; a genuine
    # interior spectral peak means the persistence is a *cycle*, not a random
    # walk, so keep it for the stationary branch (which detects the cycle).
    # The exemption is deliberately STRICTER than the general cycle gate: a true
    # cycle repeats many times (peak well away from f=0), whereas a random walk's
    # one-off low-frequency wander also shows a spectral peak. Require >= 5
    # repetitions and a clear peak so FX/RW levels are still differenced.
    if period is not None:
        pre_cyclical = True   # the caller declared a seasonal period -> keep it
    else:
        pre_period, pre_strength = _cycle_strength(y) if n >= 8 else (float("nan"), 0.0)
        pre_cyclical = (pre_strength > 0.12 and np.isfinite(pre_period)
                        and 4 <= pre_period <= n / 5.0)
    if _is_nonstationary(y) and not pre_cyclical:
        w = np.diff(y)
        drift = float(w.mean())
        drift_sig = abs(drift) > 2.0 * (np.std(w) / np.sqrt(max(w.size, 1)))
        vol = float("nan")
        has_vol = False
        try:
            aw = np.abs(w - w.mean())
            if w.size > 2 and float(sample_acf(aw, 1)[1]) > band:
                vol = float(garch_persistence(w, use="abs")["persistence"])
                has_vol = vol > vol_persist
        except Exception:
            pass
        comps = (["unit-root"] + (["drift"] if drift_sig else [])
                 + (["vol-clustering"] if has_vol else []))
        regime = "random walk + drift" if drift_sig else "random walk"
        # An I(1) level is forecast by persistence or drift -- NOT by reversion to
        # a sample mean, which is meaningless for a non-stationary level (it pulls
        # an interest rate back to an outdated long-run average -- a garbage
        # forecast). The caller can still force any model via ``forecaster=``.
        ur_per = float(period) if period is not None else _cycle_strength(w)[0]
        fname, ffn = _resolve_forecaster(
            forecaster,
            [("random walk", _fc_rw), ("drift", _fc_drift)],
            per=ur_per, max_harmonics=max_harmonics, y=y)
        return StochasticModel(
            n=n, level=level, trend_slope=drift if drift_sig else 0.0,
            has_trend=False,
            cycle_period=float("nan"), cycle_amp=float("nan"), has_cycle=False,
            n_harmonics=0, seasonal=False,
            hurst=float("nan"), has_long_memory=False,
            ar1_phi=float("nan"), has_mean_reversion=False,
            vol_persistence=vol, has_vol_clustering=has_vol,
            sigma=float(np.std(w)), sigma_walk=sigma_walk,
            components=tuple(comps), regime=regime, forecaster_name=fname,
            _forecaster=_bind_forecaster(ffn, y),
        )

    # -- Stage 1: deterministic mean (series is stationary or trend-stationary).
    slope, intercept, tstat = _ols_line(t, y)
    resid_lin = y - (intercept + slope * t)
    sst = float((y - y.mean()) @ (y - y.mean())) + 1e-12
    trend_r2 = 1.0 - float(resid_lin @ resid_lin) / sst
    # the trend must be significant AND explain real variance (a persistent but
    # stationary AR(1) can show a spuriously "significant" OLS slope that explains
    # almost no variance -- the r2 gate rejects it).
    has_trend = (tstat > trend_t) and (trend_r2 > 0.10)
    if has_trend:
        rt = fit_lsi(t, y, "a0 + a1*x", "x", k_star=1, filter_data=True)
        a0, a1 = float(rt.coeffs[0]), float(rt.coeffs[1])
        trend = a0 + a1 * t
        slope = a1
        mean_a0, mean_a1 = a0, a1
    else:
        trend = np.full(n, level)
        slope = 0.0
        mean_a0, mean_a1 = level, 0.0
    d1 = y - trend

    # 1b. seasonal / cyclical component -- a MULTI-HARMONIC Fourier model at the
    # detected (or caller-given) period, fit by fast linear least squares. The
    # harmonics capture non-sinusoidal seasonal shapes (the CO2 sawtooth, the
    # sunspot pulse) a single sinusoid cannot, which both reports the cycle right
    # and forecasts it far better; the period is FFT-detected (robust) or supplied.
    if period is not None:
        per, strength = float(period), 1.0
        has_cycle = 4 <= per <= n / 2.0
    else:
        per, strength = _cycle_strength(d1)
        has_cycle = (strength > cycle_strength and np.isfinite(per)
                     and 4 <= per <= n / min_cycles)
    cyc_amp = float("nan")
    n_harm = 0
    seas_r2 = 0.0
    if has_cycle:
        n_harm, s_coef = _fit_seasonal(t, d1, per, max_harmonics)
        x_seas = _seasonal_design(t, per, n_harm)
        cyc = x_seas @ s_coef
        cyc_amp = float(np.hypot(s_coef[0], s_coef[1]))   # fundamental amplitude
        seas_r2 = 1.0 - float(np.var(d1 - cyc)) / (float(np.var(d1)) + 1e-12)
    else:
        cyc = np.zeros(n)
        s_coef = None
    e = d1 - cyc  # stochastic residual
    # "seasonal" (a strong, clean repeating pattern) vs "cyclical" (a weaker,
    # stochastic cycle): the caller-given period, or a component that explains a
    # good share of the de-trended variance, is treated as seasonal.
    is_seasonal = has_cycle and (period is not None or seas_r2 > 0.30)

    # 2-4. whiten with AR(1), then test long memory on the innovations
    try:
        phi = float(ar1_reversion(e)["phi"])
    except Exception:
        phi = 0.0
    acf1 = float(sample_acf(e, 1)[1]) if n > 2 else 0.0
    has_mr = (mr_phi < phi < 0.99) and (abs(acf1) > band)
    innov = e[1:] - phi * e[:-1] if has_mr else (e - e.mean())
    sigma = float(np.std(innov)) if innov.size else float(np.std(e))

    hurst = float("nan")
    has_lm = False
    try:
        target = innov if innov.size > 128 else e
        hurst = float(hurst_spectral(target)["H"])
        has_lm = hurst > lm_hurst
    except Exception:
        pass

    # 5. volatility clustering on the WHITENED residual: a persistent AR(1) level
    # has trivially autocorrelated |values|, so testing the raw residual gives a
    # false positive; genuine ARCH-type clustering survives whitening.
    vol = float("nan")
    has_vol = False
    try:
        aw = np.abs(innov - innov.mean())
        if innov.size > 2 and float(sample_acf(aw, 1)[1]) > band:
            vol = float(garch_persistence(innov, use="abs")["persistence"])
            has_vol = vol > vol_persist
    except Exception:
        pass

    comps = []
    if has_trend:
        comps.append("trend")
    if has_cycle:
        comps.append("seasonal" if is_seasonal else "cycle")
    if has_lm:
        comps.append("long-memory")
    if has_mr:
        comps.append("mean-reversion")
    if has_vol:
        comps.append("vol-clustering")
    if has_trend and has_cycle:
        regime = "trend+seasonal" if is_seasonal else "trend+cycle"
    elif has_cycle:
        regime = "seasonal" if is_seasonal else "cyclical"
    elif has_trend:
        regime = "trend"
    elif has_lm:
        regime = "long-memory"
    elif has_mr:
        regime = "mean-reverting"
    elif has_vol:
        regime = "white-mean / vol-clustering"
    else:
        regime = "white noise / random walk"

    # Forecast by RMSE-optimal backtest selection over a REGIME-APPROPRIATE
    # candidate set (vs the random walk fallback). The set is scoped to the
    # detected structure so the chosen forecaster *reflects* it -- a cyclical
    # series picks its seasonal continuation vs RW (it is not handed a
    # mean-reverting level model that would forecast a flat line and erase the
    # cycle), a trending series picks trend/drift, a stationary mean-reverting one
    # picks mean reversion (toward its well-defined mean). The caller overrides via
    # ``forecaster=``.
    # When a deterministic structure (cycle/seasonal/trend) is detected, prefer to
    # forecast WITH it -- show the structure -- unless the backtest finds it
    # clearly worse than RW (a lenient margin, since a per-fold backtest under-
    # rates a phase-sensitive cyclical forecast that is actually RMSE-competitive
    # on the full record). Level regimes stay strict (RMSE-optimal, parsimony->RW).
    candidates: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]]
    sel_margin = 0.98
    if has_cycle:
        seasonal_name = "trend+seasonal" if has_trend else "seasonal"
        # two seasonal forecasters: the unbiased fitted extrapolation (wins on a
        # noisy seasonal series -- no last-value bias) and the anchored one (wins
        # on a clean trend+seasonal series like CO2). The backtest picks per series.
        candidates = [("random walk", _fc_rw),
                      (seasonal_name,
                       _make_seasonal_fc(per, max_harmonics, has_trend)),
                      (seasonal_name + " (anchored)",
                       _make_seasonal_fc_anchored(per, max_harmonics, has_trend))]
        sel_margin = 1.15
    elif has_trend:
        # Pure-trend regime keeps the conservative LINEAR trend candidate: a
        # backtest on a short record (the Nile, n=100) mis-selects a curved fit
        # that over-extrapolates a spurious level-shift "trend". The curvature-
        # aware dtfit extrapolation is used in the SEASONAL path instead (a
        # trend+seasonal series like CO2 has a genuine accelerating trend).
        candidates = [("random walk", _fc_rw), ("drift", _fc_drift),
                      ("trend", _fc_trend)]
        sel_margin = 1.15
    elif has_mr:
        candidates = [("random walk", _fc_rw), ("mean-reversion", _fc_meanrev)]
    else:                                   # long-memory / white noise
        candidates = [("random walk", _fc_rw)]
    fname, ffn = _resolve_forecaster(forecaster, candidates, per=per,
                                     max_harmonics=max_harmonics, y=y,
                                     margin=sel_margin)

    # Deterministic mean over an arbitrary time axis (trend + multi-harmonic
    # seasonal), captured for StochasticModel.simulate to regenerate the
    # structured part. Bind the fitted coefficients as defaults (no late binding).
    def _mean_fn(tt, _a0=mean_a0, _a1=mean_a1, _hc=bool(has_cycle),
                 _p=float(per) if has_cycle else float("nan"),
                 _nh=int(n_harm), _co=s_coef):
        tt = np.asarray(tt, dtype=float)
        m = _a0 + _a1 * tt
        if _hc and _co is not None:
            m = m + _seasonal_design(tt, _p, _nh) @ _co
        return m

    return StochasticModel(
        n=n, level=level, trend_slope=float(slope), has_trend=bool(has_trend),
        cycle_period=float(per) if has_cycle else float("nan"),
        cycle_amp=float(cyc_amp), has_cycle=bool(has_cycle),
        n_harmonics=int(n_harm), seasonal=bool(is_seasonal),
        hurst=hurst, has_long_memory=bool(has_lm),
        ar1_phi=float(phi), has_mean_reversion=bool(has_mr),
        vol_persistence=vol, has_vol_clustering=bool(has_vol),
        sigma=float(sigma), sigma_walk=float(sigma_walk),
        components=tuple(comps) if comps else ("none",), regime=regime,
        forecaster_name=fname, _forecaster=_bind_forecaster(ffn, y),
        _mean_fn=_mean_fn,
    )


# --------------------------------------------------------------------------- #
# Candidate forecasters + backtest model selection. Rather than a binary "trust
# the structural model or fall back to RW" guard, the merged solution backtests a
# small SET of forecasters and picks the one that actually forecasts best on the
# series -- so it beats the random walk wherever some model genuinely can (drift
# for GDP, mean reversion for rates, a seasonal continuation for a cyclical
# series) and ties it (selects RW) where nothing does.
# --------------------------------------------------------------------------- #
def _fc_rw(tr: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(tr[-1]))


def _fc_drift(tr: np.ndarray, h: int) -> np.ndarray:
    d = float(np.mean(np.diff(tr))) if tr.size > 1 else 0.0
    return float(tr[-1]) + d * np.arange(1, h + 1, dtype=float)


def _fc_meanrev(tr: np.ndarray, h: int) -> np.ndarray:
    """AR(1) mean reversion toward the sample mean (the level-series model for a
    persistent-but-stationary series: interest rates, spreads)."""
    mu = float(tr.mean())
    x = tr - mu
    if x.size < 3:
        return np.full(h, float(tr[-1]))
    phi = float((x[:-1] @ x[1:]) / (x[:-1] @ x[:-1] + 1e-12))
    phi = float(np.clip(phi, -0.999, 0.999))
    return mu + phi ** np.arange(1, h + 1, dtype=float) * (float(tr[-1]) - mu)


def _local_slope(tr: np.ndarray, period: float | None = None) -> float:
    """Slope of a line through the recent portion (so an accelerating trend like
    CO2 is extrapolated at its *current* rate, not the flatter global average)."""
    n = tr.size
    k = n if period is None else min(n, max(2 * int(period), n // 3, 10))
    k = int(max(10, min(k, n)))
    seg = tr[-k:]
    return float(np.polyfit(np.arange(seg.size, dtype=float), seg, 1)[0])


def _trend_fit_extrap(series: np.ndarray, h: int,
                      period: float | None = None) -> np.ndarray:
    """Curvature-aware trend extrapolation via a **dtfit LSI** quadratic fit over
    the recent window, evaluated at the future window indices and anchored at the
    **fitted** last value (NOT the raw last value).

    A local straight line mis-extrapolates a curved trend badly (an accelerating
    series like CO2): a quadratic LSI trend captures the curvature and forecasts
    it far better. Anchoring at the *fitted* value (rather than the raw last
    sample) is what keeps the forecast unbiased -- pinning it to the noisy last
    observation carries that residual forward as a constant offset (the seasonal
    forecast otherwise sits stably above / below the actual). Guarded: if the
    quadratic would run away beyond a sane multiple of the in-sample range it
    falls back to a fitted straight line.
    """
    steps = np.arange(1, h + 1, dtype=float)
    n = series.size
    k = n if period is None else min(n, max(2 * int(period) + 2, n // 3, 60))
    k = int(max(30, min(k, n)))
    seg = series[-k:]
    z = np.arange(seg.size, dtype=float)
    zf = (seg.size - 1) + steps
    if seg.size >= 12:
        try:
            r = fit_lsi(z, seg, "a0 + a1*z + a2*z**2", "z",
                        k_star=2, filter_data=True)
            a0, a1, a2 = (float(r.coeffs[0]), float(r.coeffs[1]),
                          float(r.coeffs[2]))
            pred = a0 + a1 * zf + a2 * zf ** 2
            base_last = a0 + a1 * (seg.size - 1) + a2 * (seg.size - 1) ** 2
            span = float(seg.max() - seg.min()) + 1e-9
            if np.max(np.abs(pred - base_last)) <= 6.0 * span:
                return pred
        except Exception:
            pass
    a1l, a0l = np.polyfit(z, seg, 1)             # fitted straight-line fallback
    return a0l + a1l * zf


def _trend_anchored_extrap(tr: np.ndarray, h: int,
                           period: float | None = None) -> np.ndarray:
    """Curvature-aware trend extrapolation anchored at the **raw last value** (the
    increment from a recent-window quadratic LSI fit added onto ``tr[-1]``). Best
    when the series is clean (low noise) so the last value is itself a good level
    estimate, e.g. CO2; paired with the unbiased :func:`_trend_fit_extrap` so the
    backtest picks per series."""
    steps = np.arange(1, h + 1, dtype=float)
    n = tr.size
    k = n if period is None else min(n, max(2 * int(period) + 2, n // 3, 60))
    k = int(max(30, min(k, n)))
    seg = tr[-k:]
    if seg.size >= 12:
        z = np.arange(seg.size, dtype=float)
        try:
            r = fit_lsi(z, seg, "a0 + a1*z + a2*z**2", "z",
                        k_star=2, filter_data=True)
            a0, a1, a2 = (float(r.coeffs[0]), float(r.coeffs[1]),
                          float(r.coeffs[2]))
            last = float(seg.size - 1)
            zf = last + steps
            pred = ((a0 + a1 * zf + a2 * zf ** 2)
                    - (a0 + a1 * last + a2 * last ** 2) + float(tr[-1]))
            span = float(seg.max() - seg.min()) + 1e-9
            if np.max(np.abs(pred - float(tr[-1]))) <= 6.0 * span:
                return pred
        except Exception:
            pass
    return float(tr[-1]) + _local_slope(tr, period) * steps


def _fc_trend(tr: np.ndarray, h: int) -> np.ndarray:
    """Linear local-slope trend (conservative: a straight line from the last
    value). Robust for a spurious/level-shift 'trend' (e.g. the Nile) that a
    curved fit would over-extrapolate."""
    return float(tr[-1]) + _local_slope(tr) * np.arange(1, h + 1, dtype=float)


def _make_seasonal_fc(period: float, max_harmonics: int, with_trend: bool,
                      window_periods: float | None = None):
    """A seasonal forecaster: the fitted **trend + multi-harmonic seasonal** model
    extrapolated forward -- NOT anchored at the raw last value.

    The harmonics are fit over the whole series (the amplitude shrinks toward what
    is reliably predictable -- RMSE-optimal when the cycle's amplitude / phase
    drift, as in the sunspot record). The series is then de-seasonalised and the
    **trend extrapolated as a fitted curve** (:func:`_trend_fit_extrap`, recent-
    window curvature for an accelerating trend like CO2); the forecast is
    ``trend(t_future) + seasonal(t_future)``. Anchoring at the *fitted* trend
    rather than the raw last sample removes the bias that otherwise leaves the
    seasonal forecast sitting stably above / below the actual (the last
    observation's residual was being carried forward as a constant offset, and the
    last point often falls at a seasonal extreme). ``window_periods`` restores the
    *visual* amplitude of the current cycles at some cost in RMSE."""
    def fc(tr: np.ndarray, h: int) -> np.ndarray:
        m = tr.size
        ts = np.arange(m, dtype=float)
        # seasonal harmonics on the (linearly de-trended) full series
        if with_trend:
            a1g, a0g = np.polyfit(ts, tr, 1)
            base = a0g + a1g * ts
        else:
            base = np.full(m, float(tr.mean()))
        win = (m if window_periods is None
               else int(min(m, max(int(window_periods * period),
                                    2 * int(period) + 2))))
        seg_t, seg_d = ts[-win:], (tr - base)[-win:]
        k, coef = _fit_seasonal(seg_t - seg_t[0], seg_d, period, max_harmonics)
        phase0 = seg_t[0]
        seas_in = _seasonal_design(ts - phase0, period, k) @ coef
        tf = (m - 1) + np.arange(1, h + 1, dtype=float)
        seas_f = _seasonal_design(tf - phase0, period, k) @ coef
        # de-seasonalise, then extrapolate the trend as a FITTED curve (no raw
        # last-value anchor -> unbiased); a flat-trend regime just holds the
        # recent de-seasonalised level.
        deseas = tr - seas_in
        if with_trend:
            trend_f = _trend_fit_extrap(deseas, h, period)
        else:
            lvl_win = min(m, max(2 * int(period) + 2, 30))
            trend_f = np.full(h, float(np.mean(deseas[-lvl_win:])))
        return trend_f + seas_f
    return fc


def _make_seasonal_fc_anchored(period: float, max_harmonics: int,
                               with_trend: bool):
    """The **anchored** seasonal forecaster -- continues the multi-harmonic
    seasonal pattern from the raw last value plus a curvature-aware anchored trend.
    Best for a clean trend+seasonal series (CO2) where the last sample carries no
    noise; offered alongside the unbiased :func:`_make_seasonal_fc` so the backtest
    keeps whichever forecasts better (the fitted one wins on a noisy series, where
    anchoring at the last value would leave the forecast sitting above/below it)."""
    def fc(tr: np.ndarray, h: int) -> np.ndarray:
        m = tr.size
        ts = np.arange(m, dtype=float)
        if with_trend:
            a1g, a0g = np.polyfit(ts, tr, 1)
            detr = tr - (a0g + a1g * ts)
        else:
            detr = tr - tr.mean()
        k, coef = _fit_seasonal(ts, detr, period, max_harmonics)
        last = float(m - 1)
        seas_last = float((_seasonal_design(np.array([last]), period, k) @ coef)[0])
        tf = last + np.arange(1, h + 1, dtype=float)
        seas_f = _seasonal_design(tf, period, k) @ coef
        trend_f = ((_trend_anchored_extrap(tr, h, period) - float(tr[-1]))
                   if with_trend else 0.0)
        return float(tr[-1]) + trend_f + (seas_f - seas_last)
    return fc


def _select_forecaster(
    y: np.ndarray,
    candidates: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]],
    *,
    max_h: int = 30,
    folds: int = 5,
    margin: float = 0.98,
) -> tuple[str, Callable[[np.ndarray, int], np.ndarray]]:
    """Rolling-origin backtest each candidate; choose the best non-RW candidate if
    its mean RMSE is within ``margin`` of the random walk's, else the random walk.

    ``margin < 1`` is strict / RMSE-optimal (a non-RW model must *beat* RW by the
    margin -- used for level regimes, parsimony toward RW). ``margin > 1`` is
    lenient (prefer to *show* a detected structure -- a seasonal cycle, a trend --
    unless it is clearly worse than RW): a per-fold backtest under-rates a
    phase-sensitive cyclical forecast that is actually RMSE-competitive over the
    full record, so a detected cycle should not be flattened to a line on its
    account."""
    n = y.size
    if n <= 50 or len(candidates) == 1:
        return candidates[0]
    hb = int(max(5, min(n // 6, max_h)))
    scores: dict[str, list[float]] = {name: [] for name, _ in candidates}
    for k in range(folds):
        end = n - k * hb
        if end - hb < 40:
            break
        train, test = y[:end - hb], y[end - hb:end]
        for name, fn in candidates:
            try:
                fc = np.asarray(fn(train, hb), dtype=float)
                scores[name].append(float(np.sqrt(np.mean((fc - test) ** 2))))
            except Exception:
                scores[name].append(float("inf"))
    means = {name: (float(np.mean(s)) if s else float("inf"))
             for name, s in scores.items()}
    rw = means.get("random walk", float("inf"))
    non_rw = {nm: v for nm, v in means.items() if nm != "random walk"}
    best = "random walk"
    if non_rw:
        struct = min(non_rw, key=lambda nm: non_rw[nm])
        if non_rw[struct] <= margin * rw:
            best = struct
    return next(c for c in candidates if c[0] == best)


# Built-in named forecasters the caller can force or compose into a candidate set.
FORECASTERS = ("random walk", "drift", "mean-reversion", "trend",
               "seasonal", "trend+seasonal")


def _build_named_forecaster(name: str, per: float, max_harmonics: int):
    table = {"random walk": _fc_rw, "drift": _fc_drift,
             "mean-reversion": _fc_meanrev, "trend": _fc_trend}
    if name in table:
        return name, table[name]
    if name in ("seasonal", "trend+seasonal"):
        if not np.isfinite(per):
            raise ValueError(
                f"forecaster {name!r} needs a seasonal period; pass period=...")
        return name, _make_seasonal_fc(per, max_harmonics, name == "trend+seasonal")
    raise ValueError(f"unknown forecaster {name!r}; choose from {FORECASTERS} "
                     "or pass a callable / list")


def _resolve_forecaster(forecaster, auto_candidates, *, per, max_harmonics, y,
                        margin=0.98):
    """Turn the user's ``forecaster=`` argument into a chosen ``(name, fn)``.

    ``"auto"`` backtest-selects over the auto candidate set (with the given
    ``margin``); a name forces that built-in; a callable ``(train, h) -> array``
    is used directly; a list of names / callables / ``(name, fn)`` pairs is a
    custom candidate set that is backtest-selected.
    """
    if forecaster is None or forecaster == "auto":
        return _select_forecaster(y, auto_candidates, margin=margin)
    if callable(forecaster):
        return "custom", forecaster
    if isinstance(forecaster, str):
        return _build_named_forecaster(forecaster, per, max_harmonics)
    cands: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]] = []
    for item in forecaster:
        if callable(item):
            cands.append((getattr(item, "__name__", "custom"),
                          cast("Callable[[np.ndarray, int], np.ndarray]", item)))
        elif isinstance(item, tuple) and len(item) == 2:
            cands.append((str(item[0]),
                          cast("Callable[[np.ndarray, int], np.ndarray]", item[1])))
        elif isinstance(item, str):
            cands.append(_build_named_forecaster(item, per, max_harmonics))
        else:
            raise TypeError(f"bad forecaster candidate: {item!r}")
    if not cands:
        raise ValueError("forecaster list is empty")
    return _select_forecaster(y, cands, margin=margin)


# --------------------------------------------------------------------------- #
# Generative residual simulators -- the stochastic half of the multipart model
# (StochasticModel.simulate). Each draws a *fresh* path of a second-order regime
# from the recovered parameter, so re-fitting it round-trips back to the same
# regime. They are the simulation twins of the domain's ground-truth generators
# (AR(1), GARCH, fractional noise), but parameterized by what fit_stochastic
# detected rather than by a known truth.
# --------------------------------------------------------------------------- #
def _sim_ar1(n: int, phi: float, sigma: float, rng: np.random.Generator,
             *, burn: int = 100) -> np.ndarray:
    """Stationary AR(1) residual ``x_t = phi x_{t-1} + sigma * eps`` (burned in,
    started from the stationary distribution so there is no transient)."""
    phi = float(np.clip(phi, -0.999, 0.999))
    e = rng.standard_normal(n + burn) * float(sigma)
    x = np.empty(n + burn)
    x[0] = e[0] / np.sqrt(max(1e-9, 1.0 - phi ** 2))
    for tt in range(1, n + burn):
        x[tt] = phi * x[tt - 1] + e[tt]
    return x[burn:]


def _sim_long_memory(n: int, hurst: float, sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Long-memory residual as ARFIMA(0, d, 0) with ``d = H - 1/2`` -- the SAME
    process family the long-memory route models, so re-fitting the simulated path
    recovers the long-memory regime (a generic fGn synthesis does not round-trip:
    the detector's AR(1) whitening absorbs it and lands on mean reversion). Built
    from the truncated MA(inf) expansion of ``(1 - B)^{-d}`` driven by innovations
    of std ``sigma``."""
    d = float(np.clip(hurst - 0.5, 0.0, 0.49))
    ntrunc = int(min(2000, max(500, n)))
    psi = np.empty(ntrunc)
    psi[0] = 1.0
    for j in range(1, ntrunc):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    e = rng.standard_normal(n + ntrunc) * float(sigma)
    return np.asarray(np.convolve(e, psi)[ntrunc:ntrunc + n], dtype=float)


def _sim_garch(n: int, persistence: float, sigma: float,
               rng: np.random.Generator, *, burn: int = 200) -> np.ndarray:
    """GARCH(1,1) innovations with the given ``alpha + beta`` persistence and
    unconditional std ``sigma`` (a conventional ``alpha`` / ``beta`` split, since
    only their sum -- the persistence -- is identified by the ACF route)."""
    persistence = float(np.clip(persistence, 0.0, 0.999))
    alpha = max(0.02, 0.1 * persistence)
    beta = max(0.0, persistence - alpha)
    var = float(sigma) ** 2
    omega = var * max(1e-6, 1.0 - persistence)
    z = rng.standard_normal(n + burn)
    s2 = np.empty(n + burn)
    r = np.empty(n + burn)
    s2[0] = var
    r[0] = np.sqrt(s2[0]) * z[0]
    for tt in range(1, n + burn):
        s2[tt] = omega + alpha * r[tt - 1] ** 2 + beta * s2[tt - 1]
        r[tt] = np.sqrt(s2[tt]) * z[tt]
    return r[burn:]
