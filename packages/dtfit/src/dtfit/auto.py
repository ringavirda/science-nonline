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

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from dtfit.types import FittingResult, InitialGuess
from dtfit._signal import dominant_period
from dtfit._pandas import (
    HAS_PANDAS,
    as_series,
    capture_index,
    extend_index,
    to_1d_array,
)
from dtfit.methods import (
    fit_lsi,
    fit_eac,
    fft_frequency_seed,
)


# shared helpers
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# auto_estimate -- the parameter-estimation merged selector
def auto_estimate(
    x: np.ndarray,
    y: np.ndarray,
    expr: str | Callable[..., Any],
    var: str,
    *,
    shape: str = "auto",
    freq_param: str | None = None,
    p0: InitialGuess | Mapping[str, float] = None,
    bounds: (
        Sequence[tuple[float, float]]
        | Mapping[str, tuple[float, float]]
        | tuple[Any, Any]
        | None
    ) = None,
    param_names: tuple[str, ...] | None = None,
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
        p0: Optional initial guess, forwarded verbatim to the base fitters: a
            sequence in sorted-name order or a full ``{name: value}`` dict
            (see :func:`dtfit.methods.normalize_p0`).
        bounds: Optional parameter bounds, forwarded verbatim to the base
            fitters: a per-parameter ``(min, max)`` pair list in sorted-name
            order, a partial ``{name: (min, max)}`` dict (unnamed parameters
            stay unbounded), or a scipy-style ``(lo, hi)`` 2-tuple (see
            :func:`dtfit.methods.normalize_bounds`).
        param_names: Parameter names for a callable ``expr`` whose signature
            cannot be introspected; forwarded to the base fitters. Ignored (but
            validated) for a symbolic ``expr``.

    ``expr`` may be a SymPy-expression string or a Python callable ``f(x, *params)``
    (see :func:`dtfit.methods.resolve_model`); the callable is forwarded to
    whichever base fitter the shape routes to.

    Returns:
        FittingResult from the selected estimator.

    Accepts pandas ``Series`` / single-column ``DataFrame`` inputs for ``x`` /
    ``y`` (coerced to 1-D float arrays); an ``ndarray`` / list input is unchanged
    and bit-identical.
    """
    x = to_1d_array(x, "x")
    y = to_1d_array(y, "y")

    if shape == "auto":
        if freq_param is not None:
            shape = "oscillatory"
        else:
            _, strength = dominant_period(y)
            shape = "oscillatory" if strength > 0.10 else "bulk"

    if shape == "oscillatory":
        return fit_lsi(x, y, expr, var, p0=p0, bounds=bounds, oscillatory=True,
                       freq_param=freq_param, param_names=param_names)
    if shape in ("transient", "peak"):
        # Forward the (self-seeded) bounds: the curvature path honours them, and
        # peak/saturating families rely on them for their positivity / width
        # guards (e.g. a Gaussian's sigma > 0), so dropping them here would let
        # those families land on a degenerate negative-width fit.
        return fit_eac(x, y, expr, var, window_mode="curvature", p0=p0,
                       bounds=bounds, param_names=param_names)
    if shape == "robust":
        # active_ratio=0.8 is the study-tuned uniform-window recipe this
        # pipeline was validated with (fit_eac itself defaults to 1.0).
        return fit_eac(x, y, expr, var, p0=p0, bounds=bounds, loss="soft_l1",
                       active_ratio=0.8, param_names=param_names)
    if shape != "bulk":
        raise ValueError(
            f"unknown shape {shape!r}; expected auto/oscillatory/transient/peak/"
            "robust/bulk"
        )

    # bulk: fit both base methods and keep the lower in-sample residual.
    best: FittingResult | None = None
    best_rmse = np.inf
    failures: list[str] = []
    for name, fitter in (
        ("fit_lsi", lambda: fit_lsi(x, y, expr, var, p0=p0, bounds=bounds,
                                    param_names=param_names)),
        # active_ratio=0.8: the study-tuned uniform-window recipe (see above).
        ("fit_eac", lambda: fit_eac(x, y, expr, var, p0=p0, bounds=bounds,
                                    active_ratio=0.8, param_names=param_names)),
    ):
        try:
            res = fitter()
            r = _rmse(y, np.asarray(res.model(x), dtype=float))
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            warnings.warn(
                f"auto_estimate: bulk candidate {name} failed ({exc}); "
                "falling back to the remaining base fitter",
                UserWarning,
                stacklevel=2,
            )
            continue
        if not np.isfinite(r):
            failures.append(f"{name}: non-finite in-sample RMSE")
            continue
        if r < best_rmse:
            best, best_rmse = res, r
    if best is None:
        raise RuntimeError(
            "both base fits (LSI and EAC) failed for this series: "
            + "; ".join(failures)
        )
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
    if _looks_like_growth(y):  # already guarantees strictly positive y
        return "logistic"
    _, strength = dominant_period(y)
    return "linear_seasonal" if (seasonal and strength > season_strength) else "poly"


def _poly_seed(y: np.ndarray, t: np.ndarray, deg: int) -> list[float]:
    pc = np.polyfit(t, y, deg)  # numpy: highest power first
    return [float(pc[deg - i]) for i in range(deg + 1)]


def _fit_model(model: str, t: np.ndarray, y: np.ndarray, t_all: np.ndarray,
               period: float | None) -> tuple[np.ndarray, FittingResult]:
    """Fit one model class and evaluate it over ``t_all`` (train + future x).

    Returns the model values over ``t_all`` **and** the underlying
    :class:`FittingResult`, so :func:`auto_forecast` can attach the fit (and a
    prediction band) as forecast provenance without re-fitting.
    """
    xspan = float(t[-1] - t[0]) or 1.0
    if model == "logistic":
        ylast = float(y[-1])
        # Growth rate scales with the time span; bracket the seed rather than
        # using fixed (0.1, 60) bounds. The old fixed bounds excluded gentle
        # slopes (seed 6/xspan can fall below 0.1) and let the global search
        # latch onto a near-vertical step -- a degenerate fit that matches
        # in-sample but extrapolates to garbage / overflow (NaN forecasts).
        k_seed = 6.0 / xspan
        r = fit_lsi(
            t, y, "L/(1 + exp(-k*(x - x0)))", "x",
            p0={"L": ylast * 1.5, "k": k_seed, "x0": float(t[0]) + xspan},
            bounds={"L": (ylast * 0.8, ylast * 12.0),
                    "k": (0.2 * k_seed, 8.0 * k_seed),
                    "x0": (float(t[0]), float(t[0]) + 2.5 * xspan)},
            k_star=6)
        return np.asarray(r.model(t_all), dtype=float), r
    if model == "linear":
        r = fit_lsi(t, y, "a0 + a1*x", "x", p0=_poly_seed(y, t, 1))
        return np.asarray(r.model(t_all), dtype=float), r
    if model == "linear_seasonal":
        s = _poly_seed(y, t, 1)
        dx = float(np.mean(np.diff(t))) or 1.0
        if period is not None and period > 0:
            w0 = 2 * np.pi / (period * dx)
        else:
            w0 = fft_frequency_seed(t, y - np.polyval(np.polyfit(t, y, 1), t)) or (2 * np.pi / xspan)
        amp = float(np.std(y)) + 1e-3
        expr = "a0 + a1*x + A*sin(w*x + p)"
        r = fit_lsi(
            t, y, expr, "x",
            p0={"a0": s[0], "a1": s[1], "A": amp, "p": 0.0, "w": w0},
            bounds={"a0": (-1e6, 1e6), "a1": (-1e6, 1e6),
                    "A": (1e-3, 5 * amp), "p": (-np.pi, np.pi),
                    "w": (0.7 * w0, 1.3 * w0)},
            freq_param="w")
        return np.asarray(r.model(t_all), dtype=float), r
    # poly (quadratic level)
    r = fit_lsi(t, y, "a0 + a1*x + a2*x**2", "x", p0=_poly_seed(y, t, 2))
    return np.asarray(r.model(t_all), dtype=float), r


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
        sp_tail = _fit_model(model, t[:k], y[:k], t, period)[0][k:n]
        s_rmse = _rmse(y[k:], sp_tail)
    except Exception:
        return True
    p_rmse = _rmse(y[k:], np.full(n - k, y[k - 1])) + 1e-12
    return bool(np.isfinite(s_rmse) and s_rmse > factor * p_rmse)


class ForecastResult(np.ndarray):
    """The forecast values plus their provenance -- a :class:`numpy.ndarray`.

    Subclasses ``np.ndarray`` so it *is* the length-``horizon`` forecast: every
    existing caller keeps working unchanged (indexing, ``.shape``, ``len``,
    arithmetic, ``np.allclose``, ``np.isfinite``). It additionally carries where
    the numbers came from, so a caller can inspect the fit and an uncertainty
    band without a second call.

    Attributes:
        model_name: The model that actually produced the forecast, *including the
            fallback provenance* -- e.g. ``"logistic"`` for a clean fit,
            ``"linear (poly diverged)"`` when the divergence guard dropped a
            runaway quadratic, ``"linear (logistic failed)"`` when the primary
            fit raised, ``"random_walk"`` / ``"persistence (...)"`` on the
            persistence paths.
        result: The underlying :class:`FittingResult` when the forecast came from
            a real fit; ``None`` on the persistence / random-walk paths.
        std_band: A length-``horizon`` 1-sigma prediction band (delta method) when
            the fit exposed a covariance and the propagation succeeded; ``None``
            otherwise. Named ``std_band`` (not ``std``) so it does not shadow
            ``numpy.ndarray.std`` -- ``fc.std()`` and ``np.std(fc)`` keep working.
        index: The length-``horizon`` FUTURE pandas index continuing the ``x``
            passed to :func:`auto_forecast` (from ``extend_index(capture_index(x),
            horizon)``); ``None`` when ``x`` was not a pandas ``Series`` /
            ``DataFrame`` with an inferable step, or when pandas is absent. Purely
            additive -- the forecast values are unaffected. Use :meth:`to_series`
            for the pandas "in -> out" view.
    """

    model_name: str
    result: FittingResult | None
    std_band: np.ndarray | None
    index: Any

    def __new__(
        cls,
        values: np.ndarray | Sequence[float],
        *,
        model_name: str = "",
        result: FittingResult | None = None,
        std_band: np.ndarray | None = None,
        index: Any = None,
    ) -> ForecastResult:
        obj = np.asarray(values, dtype=float).view(cls)
        obj.model_name = model_name
        obj.result = result
        obj.std_band = None if std_band is None else np.asarray(std_band, dtype=float)
        obj.index = index
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        # Called on every construction path (view/slice/ufunc). Scalar
        # provenance (model_name/result) carries forward unconditionally, but
        # the per-step ``index`` and ``std_band`` are length-``horizon`` and
        # only valid while the array keeps that length. On a slice / reduction /
        # broadcast the derived array no longer aligns with them, so drop them
        # rather than carry a misaligned band or a wrong-length index (which
        # would make ``fc[:3].std_band`` silently wrong and ``fc[:3].to_series()``
        # raise a length error).
        if obj is None:
            return
        self.model_name = getattr(obj, "model_name", "")
        self.result = getattr(obj, "result", None)
        src_std = getattr(obj, "std_band", None)
        src_index = getattr(obj, "index", None)
        n = self.shape[0] if self.ndim == 1 else -1
        self.std_band = (
            src_std if src_std is not None and len(src_std) == n else None
        )
        self.index = (
            src_index if src_index is not None and len(src_index) == n else None
        )

    def to_series(self) -> Any:
        """Return the forecast values as a pandas ``Series`` indexed by :attr:`index`.

        This is the opt-in "pandas in -> pandas out" view: :func:`auto_forecast`
        always returns an ``ndarray`` subclass for back-compat, and this method
        wraps those same values in a ``Series`` carrying the future index.

        Returns:
            A pandas ``Series`` of the forecast values with index :attr:`index`.

        Raises:
            ValueError: when :attr:`index` is ``None`` (``x`` was not a pandas
                object with an extendable index, or pandas is not installed), so
                there is no future index to align to.
        """
        if not HAS_PANDAS:
            raise ValueError(
                "ForecastResult.to_series() requires pandas, which is not installed"
            )
        if self.index is None:
            raise ValueError(
                "ForecastResult.to_series() needs a future index, but .index is "
                "None: auto_forecast was not given a pandas Series/DataFrame whose "
                "index has an inferable frequency/step. Pass x as such an object."
            )
        return as_series(np.asarray(self), self.index)


def _persist(
    y: np.ndarray, horizon: int, model_name: str, index: Any = None
) -> ForecastResult:
    """A flat ``y[-1]`` persistence forecast, tagged with its provenance."""
    return ForecastResult(
        np.full(horizon, float(y[-1])), model_name=model_name, result=None,
        std_band=None, index=index,
    )


def _forecast_std(
    result: FittingResult | None, future: np.ndarray
) -> np.ndarray | None:
    """Best-effort 1-sigma band at the future grid from the fit's covariance.

    Returns ``None`` (never raises) when the fit has no covariance, when the
    delta-method propagation fails, or when it yields a non-finite / wrong-length
    band -- an uncertainty band is a bonus, not a contract of the forecast.
    """
    if result is None or result.cov is None:
        return None
    try:
        _, std = result.predict(future, return_std=True)
    except Exception:
        # Any numeric/model failure in the band is non-fatal: the forecast values
        # are still valid, we simply report no uncertainty for them.
        return None
    std = np.asarray(std, dtype=float)
    if std.shape != future.shape or not np.all(np.isfinite(std)):
        return None
    return std


def auto_forecast(
    x: np.ndarray,
    y: np.ndarray,
    horizon: int,
    *,
    model: str = "auto",
    period: float | None = None,
    seasonal: bool = True,
    season_strength: float = 0.05,
) -> ForecastResult:
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

    ``x`` / ``y`` may be pandas ``Series`` / single-column ``DataFrame`` inputs
    (coerced to 1-D float arrays); an ``ndarray`` / list input is unchanged and
    the forecast is bit-identical. When ``x`` is a pandas object whose index is
    extendable (a ``DatetimeIndex`` with an inferable frequency, or an
    integer-stepped index), the result carries a length-``horizon`` FUTURE
    ``.index`` and :meth:`ForecastResult.to_series` returns the pandas view.

    Returns:
        A :class:`ForecastResult` -- an ``np.ndarray`` of the length-``horizon``
        forecast (the values at the extrapolated x grid) that also carries
        ``.model_name`` (the model that produced it, with fallback provenance),
        ``.result`` (the underlying :class:`FittingResult`, or ``None`` on the
        persistence paths), ``.std_band`` (a 1-sigma band when available) and
        ``.index`` (the future pandas index, or ``None`` for a non-pandas ``x``).
    """
    allowed = {"auto", "logistic", "linear", "poly", "linear_seasonal", "random_walk"}
    if model not in allowed:
        raise ValueError(
            f"unknown model {model!r}; expected one of {sorted(allowed)}"
        )
    x_index = capture_index(x)
    x = to_1d_array(x, "x")
    y = to_1d_array(y, "y")
    fut_index = extend_index(x_index, horizon)
    if horizon <= 0:
        return ForecastResult(np.empty(0), model_name=model, result=None,
                              std_band=None, index=fut_index)
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    future = x[-1] + dx * np.arange(1, horizon + 1)
    t_all = np.concatenate([x, future])

    chosen = _auto_model(y, seasonal, season_strength) if model == "auto" else model

    # Persistence paths: an explicit random walk, or a structured model that
    # cannot beat naive persistence on a held-out training tail. (The `or`
    # short-circuits exactly as before -- an explicit random walk never runs the
    # no-structure probe.)
    if chosen == "random_walk":
        return _persist(y, horizon, "random_walk", index=fut_index)
    if _no_structure(chosen, x, y, period):
        return _persist(
            y, horizon, f"persistence ({chosen} no structure)", index=fut_index
        )

    result: FittingResult | None = None
    produced = chosen
    try:
        pred, result = _fit_model(chosen, x, y, t_all, period)
    except Exception as exc:
        warnings.warn(
            f"auto_forecast: {chosen} fit failed ({exc}); falling back to linear",
            UserWarning,
            stacklevel=2,
        )
        pred, result = _fit_model("linear", x, y, t_all, period)
        produced = f"linear ({chosen} failed)"

    if chosen == "poly" and _diverges(pred, y):
        try:
            pred, result = _fit_model("linear", x, y, t_all, period)
            produced = "linear (poly diverged)"
        except Exception as exc:
            warnings.warn(
                f"auto_forecast: linear fit failed ({exc}); "
                "falling back to persistence",
                UserWarning,
                stacklevel=2,
            )
            return _persist(
                y, horizon, "persistence (linear failed)", index=fut_index
            )

    return ForecastResult(
        pred[x.size:],
        model_name=produced,
        result=result,
        std_band=_forecast_std(result, future),
        index=fut_index,
    )
