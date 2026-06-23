"""High-level "just fit it" entry points distilled from the domain merged pipelines.

The per-domain validation suite showed that the single biggest lever is **picking
the structurally-correct model / estimator variant**, not the solver. Those
studies wrapped that routing in two "merged" pipelines -- a parameter-estimation
*selector* and an auto-composed *forecaster* -- that compose only the validated,
stable levers behind one call. This module promotes that composition into the
stable API:

* :func:`auto_estimate` -- recover physical parameters, routing by signal *shape*
  to the estimator variant that fits it (oscillatory -> the LSI oscillatory
  recipe; transient / peak -> curvature-window EAC; outliers -> robust-loss EAC;
  else the better of LSI / EAC by in-sample fit).
* :func:`auto_forecast` -- a structured fit-then-extrapolate forecaster that
  routes the model class (saturating growth -> logistic; a detected cycle -> a
  joint linear+seasonal fit; otherwise a quadratic level), with a **no-structure
  guard** (persist when the fit cannot beat a random walk on a held-out training
  tail) and a **divergence guard** (drop a runaway quadratic to linear).

Both compose only stable pieces (:func:`dtfit.fit_lsi`, :func:`dtfit.fit_eac`
with curvature-adaptive windows and the robust loss, :func:`dtfit.fft_frequency_seed`);
they are the conservative merges the studies validated, and they keep the honest
ceilings: near-random-walk series fall back to persistence, and ``auto_estimate``
matches but does not beat a well-initialised NLLS on clean bulk shapes.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import sympy as sp

from dtfit.types import FittingResult, InitialGuess
from dtfit.methods import (
    fit_lsi,
    fit_eac,
    fft_frequency_seed,
    model_params,
)


# shared helpers
def _dominant_period(y: np.ndarray, *, min_period: int = 4) -> tuple[float, float]:
    """``(period_samples, strength)`` of the strongest spectral peak of a
    linearly-detrended series; ``strength`` is the peak's share of detrended
    power in ``[0, 1]`` (small => no real cycle)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2 * min_period:
        return float("nan"), 0.0
    t = np.arange(n)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    spec = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0
    valid = (freqs > 1.0 / max(n, 1)) & (freqs <= 1.0 / min_period)
    if not valid.any() or spec[valid].sum() == 0:
        return float("nan"), 0.0
    k = int(np.argmax(np.where(valid, spec, 0.0)))
    strength = float(spec[k] / spec[1:].sum()) if spec[1:].sum() > 0 else 0.0
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else float("nan")
    return period, strength


def _ordered(expr: str, var: str, pmap: dict[str, tuple]) -> tuple[list[float], list[tuple[float, float]]]:
    """``(p0, bounds)`` ordered by the model's sorted parameter names -- the
    layout :func:`fit_lsi` uses -- from a ``{name: (p0, lo, hi)}`` map."""
    names = [str(s) for s in model_params(cast(sp.Expr, sp.sympify(expr)), sp.Symbol(var))]
    p0 = [pmap[n][0] for n in names]
    bounds = [(pmap[n][1], pmap[n][2]) for n in names]
    return p0, bounds


def _eac_bounds(bounds):
    return ([b[0] for b in bounds], [b[1] for b in bounds]) if bounds else None


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# auto_estimate -- the parameter-estimation merged selector
def auto_estimate(
    x: np.ndarray,
    y: np.ndarray,
    expr: str,
    var: str,
    *,
    shape: str = "auto",
    freq_param: str | None = None,
    p0: InitialGuess = None,
    bounds: list[tuple[float, float]] | None = None,
) -> FittingResult:
    """Recover the parameters of ``expr`` by routing to the estimator that fits
    the signal's *shape* (the parameter-estimation domain study's merged selector).

    Args:
        x, y: Observed samples.
        expr, var: Model expression and main variable.
        shape: One of ``"auto"`` (detect oscillation, else bulk), ``"oscillatory"``,
            ``"transient"`` / ``"peak"`` (curvature-window EAC), ``"robust"``
            (outlier-robust EAC via ``loss="soft_l1"``), or ``"bulk"`` (the better
            of LSI / EAC by in-sample fit). The variant-follows-shape mapping the
            study validated.
        freq_param: Name of the angular-frequency parameter, forwarded to the LSI
            oscillatory recipe (:func:`fit_lsi`); implies an oscillatory shape.
        p0: Optional initial guess.
        bounds: Optional per-parameter ``(min, max)`` bounds.

    Returns:
        FittingResult from the selected estimator.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if shape == "auto":
        if freq_param is not None:
            shape = "oscillatory"
        else:
            _, strength = _dominant_period(y)
            shape = "oscillatory" if strength > 0.10 else "bulk"

    if shape == "oscillatory":
        return fit_lsi(x, y, expr, var, p0=p0, bounds=bounds, oscillatory=True,
                       freq_param=freq_param)
    if shape in ("transient", "peak"):
        return fit_eac(x, y, expr, var, window_mode="curvature", p0=p0)
    if shape == "robust":
        return fit_eac(x, y, expr, var, p0=p0, bounds=_eac_bounds(bounds),
                       loss="soft_l1")
    if shape != "bulk":
        raise ValueError(
            f"unknown shape {shape!r}; expected auto/oscillatory/transient/peak/"
            "robust/bulk"
        )

    # bulk: fit both base methods and keep the lower in-sample residual.
    best: FittingResult | None = None
    best_rmse = np.inf
    for fitter in (
        lambda: fit_lsi(x, y, expr, var, p0=p0, bounds=bounds),
        lambda: fit_eac(x, y, expr, var, p0=p0, bounds=_eac_bounds(bounds)),
    ):
        try:
            res = fitter()
            r = _rmse(y, np.asarray(res.model(x), dtype=float))
        except Exception:
            continue
        if np.isfinite(r) and r < best_rmse:
            best, best_rmse = res, r
    if best is None:
        raise RuntimeError("both base fits (LSI and EAC) failed for this series")
    return best


# auto_forecast -- the forecasting merged pipeline
def _looks_like_growth(y: np.ndarray) -> bool:
    if np.any(y <= 0):
        return False
    d = np.diff(y)
    if d.size == 0:
        return False
    monotone = np.mean(np.sign(d) == np.sign(d[np.argmax(np.abs(d))])) > 0.9
    return bool(monotone and (abs(y[-1] / y[0]) > 3 or abs(y[0] / y[-1]) > 3))


def _auto_model(y: np.ndarray, seasonal: bool, season_strength: float) -> str:
    """Route the model class with no per-series tuning (the merged forecaster's
    router): saturating growth -> logistic; a detected cycle -> linear+seasonal;
    otherwise a quadratic level (caught by the divergence guard if it runs away)."""
    if _looks_like_growth(y) and np.all(y > 0):
        return "logistic"
    _, strength = _dominant_period(y)
    return "linear_seasonal" if (seasonal and strength > season_strength) else "poly"


def _poly_seed(y: np.ndarray, t: np.ndarray, deg: int) -> list[float]:
    pc = np.polyfit(t, y, deg)  # numpy: highest power first
    return [float(pc[deg - i]) for i in range(deg + 1)]


def _fit_model(model: str, t: np.ndarray, y: np.ndarray, t_all: np.ndarray,
               period: float | None) -> np.ndarray:
    """Fit one model class and evaluate it over ``t_all`` (train + future x)."""
    xspan = float(t[-1] - t[0]) or 1.0
    if model == "logistic":
        ylast = float(y[-1])
        # Growth rate scales with the time span; bracket the seed rather than
        # using fixed (0.1, 60) bounds. The old fixed bounds excluded gentle
        # slopes (seed 6/xspan can fall below 0.1) and let the global search
        # latch onto a near-vertical step -- a degenerate fit that matches
        # in-sample but extrapolates to garbage / overflow (NaN forecasts).
        k_seed = 6.0 / xspan
        p0, bounds = _ordered(
            "L/(1 + exp(-k*(x - x0)))", "x",
            {"L": (ylast * 1.5, ylast * 0.8, ylast * 12.0),
             "k": (k_seed, 0.2 * k_seed, 8.0 * k_seed),
             "x0": (t[0] + xspan, t[0], t[0] + 2.5 * xspan)})
        r = fit_lsi(t, y, "L/(1 + exp(-k*(x - x0)))", "x", p0=p0, bounds=bounds,
                    k_star=6)
        return np.asarray(r.model(t_all), dtype=float)
    if model == "linear":
        r = fit_lsi(t, y, "a0 + a1*x", "x", p0=_poly_seed(y, t, 1))
        return np.asarray(r.model(t_all), dtype=float)
    if model == "linear_seasonal":
        s = _poly_seed(y, t, 1)
        dx = float(np.mean(np.diff(t))) or 1.0
        if period is not None and period > 0:
            w0 = 2 * np.pi / (period * dx)
        else:
            w0 = fft_frequency_seed(t, y - np.polyval(np.polyfit(t, y, 1), t)) or (2 * np.pi / xspan)
        amp = float(np.std(y)) + 1e-3
        expr = "a0 + a1*x + A*sin(w*x + p)"
        p0, bounds = _ordered(expr, "x", {
            "a0": (s[0], -1e6, 1e6), "a1": (s[1], -1e6, 1e6),
            "A": (amp, 1e-3, 5 * amp), "p": (0.0, -np.pi, np.pi),
            "w": (w0, 0.7 * w0, 1.3 * w0)})
        r = fit_lsi(t, y, expr, "x", p0=p0, bounds=bounds, freq_param="w")
        return np.asarray(r.model(t_all), dtype=float)
    # poly (quadratic level)
    r = fit_lsi(t, y, "a0 + a1*x + a2*x**2", "x", p0=_poly_seed(y, t, 2))
    return np.asarray(r.model(t_all), dtype=float)


def _diverges(pred: np.ndarray, y: np.ndarray, k: float = 5.0) -> bool:
    rng = float(np.ptp(y)) or 1.0
    lo, hi = float(y.min()) - k * rng, float(y.max()) + k * rng
    return not np.all((pred >= lo) & (pred <= hi))


def _no_structure(model: str, t: np.ndarray, y: np.ndarray,
                  period: float | None, factor: float = 8.0) -> bool:
    """True when the structured model cannot get near naive persistence on a
    held-out tail of the *training* data -- the near-random-walk signature."""
    n = y.size
    if n < 24:
        return False
    k = int(n * 0.8)
    try:
        sp_tail = _fit_model(model, t[:k], y[:k], t, period)[k:n]
        s_rmse = _rmse(y[k:], sp_tail)
    except Exception:
        return True
    p_rmse = _rmse(y[k:], np.full(n - k, y[k - 1])) + 1e-12
    return bool(np.isfinite(s_rmse) and s_rmse > factor * p_rmse)


def auto_forecast(
    x: np.ndarray,
    y: np.ndarray,
    horizon: int,
    *,
    model: str = "auto",
    period: float | None = None,
    seasonal: bool = True,
    season_strength: float = 0.05,
) -> np.ndarray:
    """Structured fit-then-extrapolate forecast (the forecasting merged pipeline).

    Routes the model class, applies a no-structure guard (persist on a
    near-random-walk series) and a divergence guard (drop a runaway quadratic to
    linear), then extrapolates ``horizon`` steps past ``x`` on its uniform grid.

    Args:
        x, y: The observed series (``x`` (near-)uniformly sampled).
        horizon: Number of future steps to forecast.
        model: ``"auto"`` (route by structure) or one of ``"logistic"``,
            ``"linear"``, ``"poly"``, ``"linear_seasonal"``, ``"random_walk"``.
        period: Optional known seasonal period (in samples) for the seasonal fit.
        seasonal: Whether to consider a seasonal model under ``"auto"``.
        season_strength: Minimum detected cycle strength to pick a seasonal model.

    Returns:
        The length-``horizon`` forecast (the values at the extrapolated x grid).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if horizon <= 0:
        return np.empty(0)
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    future = x[-1] + dx * np.arange(1, horizon + 1)
    t_all = np.concatenate([x, future])

    chosen = _auto_model(y, seasonal, season_strength) if model == "auto" else model

    if chosen == "random_walk" or _no_structure(chosen, x, y, period):
        return np.full(horizon, float(y[-1]))

    try:
        pred = _fit_model(chosen, x, y, t_all, period)
    except Exception:
        pred = _fit_model("linear", x, y, t_all, period)

    if chosen == "poly" and _diverges(pred, y):
        try:
            pred = _fit_model("linear", x, y, t_all, period)
        except Exception:
            return np.full(horizon, float(y[-1]))

    return pred[x.size:]
