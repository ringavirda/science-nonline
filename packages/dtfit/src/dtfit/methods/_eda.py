"""EDA -- Equal Differential Areas (equal-areas) fitting.

Numeric successor to the symbolic DSBE method. Identifies model parameters by
matching integral areas of the model and the data over a set of windows, rather
than balancing differential spectra. Integration smooths noise, so this is
markedly more robust than spectral/derivative-based approaches and works
directly on raw ``(x, y)`` data.

Overdetermined by default
-------------------------
The original EDA used exactly ``n`` windows for ``n`` parameters -- an
exactly-determined system with no redundancy, which throws away the very noise
averaging that integration buys. Here the active region is split into
``n_windows >= n`` windows (default ``2n``), giving an **overdetermined**
area-matching system solved by Levenberg-Marquardt / trust-region least squares
with an analytic (integrated) Jacobian. More equations than unknowns means the
random per-window integration errors partly cancel, and a parameter covariance
can be estimated from the residual Jacobian. Bounds and a robust loss are
exposed for constrained / outlier-prone fits.
"""

from typing import Any, Callable, cast

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from dtfit.log import echo
from dtfit._core._kernels import simpson_windows, simpson_windows_rows
from dtfit.types import FittingResult, InitialGuess
from ._common import model_params, _covariance


def _dominant_cycles(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate how many full periods of the dominant tone the record spans.

    Uses the FFT peak above DC. Non-oscillatory shapes (a trend, a single peak,
    a sigmoid) concentrate their energy at/near DC and return ~0-1; a genuine
    oscillation returns roughly its cycle count. Used to auto-scale the window
    count so windows stay sub-period (see ``fit_eda``).
    """
    if x.size < 8:
        return 0.0
    yd = np.asarray(y, dtype=float) - float(np.mean(y))
    duration = float(x[-1] - x[0])
    if duration <= 0.0 or not np.any(yd):
        return 0.0
    freqs = np.fft.rfftfreq(x.size, d=duration / (x.size - 1))
    ps = np.abs(np.fft.rfft(yd))
    if ps.size < 2:
        return 0.0
    k_peak = 1 + int(np.argmax(ps[1:]))  # skip the DC bin
    return float(freqs[k_peak] * duration)


def fit_eda(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    active_ratio: float = 0.8,
    n_windows: int | None = None,
    bounds: tuple | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
    p0: InitialGuess = None,
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` with the equal-areas criterion.

    Args:
        data_x, data_y: Observed samples.
        expr: Model expression, e.g. ``"a * atan(w * t)"``.
        var: Main variable name in ``expr``.
        active_ratio: Fraction of the (leading) data used for window placement;
            the informative transient usually lives here.
        n_windows: Number of integration windows (area equations). Defaults to
            ``2 * n_params`` for an overdetermined, noise-averaging fit. Must be
            ``>= n_params``; clamped so each window keeps at least 3 samples.
        bounds: Optional ``(lower, upper)`` parameter bounds (as accepted by
            ``scipy.optimize.least_squares``); switches the solver to
            trust-region.
        loss: Least-squares loss (e.g. ``"linear"`` or ``"soft_l1"`` for outlier
            robustness). The loss acts on the *window-area* residuals, so it can
            only down-weight whole contaminated windows -- give it enough windows
            (``n_windows``) that outliers stay localized for it to bite.
        f_scale: Soft margin of the robust ``loss`` (``scipy``'s ``f_scale``):
            residuals below it stay quadratic, above it are down-weighted. The
            default of ``1.0`` is far larger than typical small window-area
            residuals, which would leave a robust ``loss`` in its quadratic
            (i.e. ``"linear"``) regime; lower it to the scale of a clean window's
            area residual to actually engage the robustness. Ignored when
            ``loss="linear"``.
        p0: Optional initial guess (defaults to ones).

    Returns:
        FittingResult with the fitted coefficients, callable model and (when
        overdetermined) a parameter covariance estimate.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    n = len(params)
    if n == 0:
        raise RuntimeError("Model expression has no free parameters to fit.")

    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)
    if x.size < 2 * n:
        raise RuntimeError(
            f"Need at least {2 * n} samples to fit {n} parameters via EDA."
        )

    model_func = sp.lambdify((t, *params), f_sym, "numpy")
    jac_funcs = [
        sp.lambdify((t, *params), sp.diff(f_sym, p), "numpy") for p in params
    ]

    # Split the active region into integration windows (>= n for solvability,
    # default 2n for redundancy), each with at least 3 samples for Simpson.
    idx_max = max(int(x.size * active_ratio), n + 1)
    requested = 2 * n if n_windows is None else int(n_windows)
    if n_windows is None:
        # Oscillatory data: a window spanning whole periods integrates to ~0, so
        # its area is blind to amplitude/phase and the criterion loses
        # conditioning as the cycle count grows. Auto-scale to keep windows
        # sub-period (~3 per dominant cycle). Non-oscillatory shapes have their
        # FFT peak at/near DC (cycles ~ 0-1) and are left at the 2n default.
        cycles = _dominant_cycles(x[:idx_max], y[:idx_max])
        if cycles >= 2.0:
            requested = max(requested, int(np.ceil(3.0 * cycles)))
    m = max(n, min(requested, idx_max // 3))
    window = max(idx_max // m, 2)

    # Contiguous window spans [start, stop) over the active region. The model
    # and its sensitivities are evaluated once over the whole active region per
    # solver step and integrated per window by the (compiled) Simpson kernel,
    # rather than re-evaluated window by window.
    x_active = np.ascontiguousarray(x[:idx_max])
    starts = np.array([i * window for i in range(m)], dtype=np.intp)
    stops = np.array(
        [(i + 1) * window if i < m - 1 else idx_max for i in range(m)],
        dtype=np.intp,
    )
    data_areas_arr = simpson_windows(y[:idx_max], x_active, starts, stops)
    echo(f"EDA windows: {m} (params: {n})")

    def _eval(func: Callable[..., Any], c: np.ndarray) -> np.ndarray:
        v = func(x_active, *c)
        if np.ndim(v) == 0:  # constant model/derivative -> broadcast
            v = np.full_like(x_active, float(v))
        v = np.ascontiguousarray(v, dtype=float)
        # A transcendental sensitivity can be singular at an isolated sample
        # (e.g. d/dn of x**n is x**n*log(x), which is NaN at x=0) while its
        # integral over the window is finite -- the limit there is 0. A raw NaN
        # would poison the whole Simpson area and silently stall LM (it returns
        # the seed unchanged) or crash TRF; replace such measure-zero blow-ups
        # with their finite contribution so the area equation stays well-posed.
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def residuals(c: np.ndarray) -> np.ndarray:
        mv = _eval(model_func, c)
        return simpson_windows(mv, x_active, starts, stops) - data_areas_arr

    def jacobian(c: np.ndarray) -> np.ndarray:
        rows = np.vstack([_eval(jac_funcs[j], c) for j in range(n)])
        return simpson_windows_rows(rows, x_active, starts, stops).T

    guess = np.ones(n) if p0 is None else np.asarray(p0, float)
    if bounds is not None or loss != "linear":
        method = "trf"
        kwargs: dict[str, Any] = {"loss": loss, "f_scale": f_scale}
        if bounds is not None:
            kwargs["bounds"] = bounds
        # cast: scipy's stub types `jac` as a str literal, omitting the
        # callable form the runtime accepts.
        sol = least_squares(
            residuals, guess, jac=cast(Any, jacobian), method=method, **kwargs
        )
    else:
        sol = least_squares(
            residuals, guess, jac=cast(Any, jacobian), method="lm"
        )
    coeffs = np.asarray(sol.x, dtype=np.float64)
    echo("EDA fitted coefficients:", coeffs)

    cov = _covariance(sol.jac, sol.fun, n)

    model = sp.lambdify(t, f_sym.subs(dict(zip(params, coeffs))), "numpy")
    return FittingResult(model=model, coeffs=coeffs, cov=cov,
                         expr=expr, var=var, names=tuple(str(p) for p in params))


def _curvature_edges(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
    """Index edges so each window holds ~equal cumulative |curvature|."""
    d2 = np.abs(np.gradient(np.gradient(y, x), x))
    d2 = d2 + 1e-9  # floor so flat regions still get covered
    cum = np.concatenate([[0.0], np.cumsum(d2)])
    cum /= cum[-1]
    targets = np.linspace(0, 1, m + 1)
    edges = np.searchsorted(cum, targets)
    edges[0], edges[-1] = 0, x.size
    # enforce >= 3 samples per window for Simpson
    for k in range(1, m + 1):
        if edges[k] - edges[k - 1] < 3:
            edges[k] = min(edges[k - 1] + 3, x.size)
    return np.unique(edges)


def fit_eda_adaptive(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    n_windows: int | None = None,
    window_mode: str = "curvature",
    p0: InitialGuess = None,
) -> FittingResult:
    """EDA with information-adaptive (curvature-weighted) window placement.

    Where :func:`fit_eda` splits the active region into **equal** windows, this
    variant places window edges so each carries roughly equal *information*
    (cumulative absolute curvature): narrow windows where the signal bends and
    wide ones where it is smooth. For signals with a localized transient (a step
    take-off, a sharp turn, a peak's rise) this is a better-conditioned
    area-matching system -- validated as the best estimator on concentrated
    transients and rational-saturating shapes (Michaelis-Menten / Hill) in the
    parameter-estimation domain study.

    Args:
        data_x, data_y: Observed samples.
        expr, var: Model expression and main variable.
        n_windows: Number of area windows (default ``2 * n_params``).
        window_mode: ``"curvature"`` (default, curvature-adaptive edges) or
            ``"equal"`` (uniform edges, the :func:`fit_eda` baseline placement).
        p0: Optional initial guess (defaults to ones).

    Returns:
        FittingResult with coefficients, callable model and covariance.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    n = len(params)
    if n == 0:
        raise RuntimeError("Model expression has no free parameters to fit.")

    x = np.ascontiguousarray(np.asarray(data_x, dtype=float))
    y = np.asarray(data_y, dtype=float)
    m = 2 * n if n_windows is None else int(n_windows)
    m = max(n, m)

    if window_mode == "curvature":
        edges = _curvature_edges(x, y, m)
    else:
        edges = np.linspace(0, x.size, m + 1).astype(np.intp)
    starts = np.ascontiguousarray(edges[:-1], dtype=np.intp)
    stops = np.ascontiguousarray(edges[1:], dtype=np.intp)

    model_func = sp.lambdify((t, *params), f_sym, "numpy")
    jac_funcs = [sp.lambdify((t, *params), sp.diff(f_sym, p), "numpy") for p in params]
    data_areas = simpson_windows(np.ascontiguousarray(y), x, starts, stops)

    def _eval(func: Callable[..., Any], c: np.ndarray) -> np.ndarray:
        v = func(x, *c)
        if np.ndim(v) == 0:
            v = np.full_like(x, float(v))
        v = np.ascontiguousarray(v, dtype=float)
        # See fit_eda: replace a measure-zero singular sample (e.g. x**n*log(x)
        # at x=0) with its finite integral contribution so it cannot poison the
        # window area / Jacobian and stall the solver.
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def residual(c: np.ndarray) -> np.ndarray:
        mv = _eval(model_func, c)
        return simpson_windows(mv, x, starts, stops) - data_areas

    def jacobian(c: np.ndarray) -> np.ndarray:
        rows = np.vstack([_eval(jac_funcs[j], c) for j in range(n)])
        return simpson_windows_rows(rows, x, starts, stops).T

    guess = np.ones(n) if p0 is None else np.asarray(p0, float)
    sol = least_squares(residual, guess, jac=cast(Any, jacobian), method="lm")
    coeffs = np.asarray(sol.x, dtype=np.float64)
    cov = _covariance(sol.jac, sol.fun, n)
    model = sp.lambdify(t, f_sym.subs(dict(zip(params, coeffs))), "numpy")
    return FittingResult(model=model, coeffs=coeffs, cov=cov,
                         expr=expr, var=var, names=tuple(str(p) for p in params))

