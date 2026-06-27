"""EAC -- Equal-Areas Criterion fitting.

Numeric successor to the symbolic DSBE method. Identifies model parameters by
matching integral areas of the model and the data over a set of windows, rather
than balancing differential spectra. Integration smooths noise, so this is
markedly more robust than spectral/derivative-based approaches and works
directly on raw ``(x, y)`` data.

Overdetermined by default
-------------------------
The original EAC used exactly ``n`` windows for ``n`` parameters -- an
exactly-determined system with no redundancy, which throws away the very noise
averaging that integration buys. Here the active region is split into
``n_windows >= n`` windows (default ``2n``), giving an **overdetermined**
area-matching system solved by Levenberg-Marquardt / trust-region least squares
with an analytic (integrated) Jacobian. More equations than unknowns means the
random per-window integration errors partly cancel, and a parameter covariance
can be estimated from the residual Jacobian. Bounds and a robust loss are
exposed for constrained / outlier-prone fits.

Window placement (``window_mode``)
----------------------------------
The windows are placed either ``"uniform"`` (equal spans over the active region,
the default) or ``"curvature"`` (edges carrying roughly equal cumulative
absolute curvature -- narrow where the signal bends, wide where it is smooth).
The curvature placement is the better-conditioned area-matching system for
signals with a localized transient (a step take-off, a sharp turn, a peak's
rise); it was validated as the best estimator on concentrated transients and
rational-saturating shapes (Michaelis-Menten / Hill) in the parameter-estimation
domain study.

Robustness
----------
Two complementary outlier defences are available. A **robust least-squares
loss** (``loss=`` / ``f_scale=``) down-weights contaminated *window-area*
residuals within a single overdetermined fit -- the mechanism characterised in
the EAC paper. For heavier contamination, :func:`dtfit.ensemble_fit` aggregates
fits over overlapping windows by median, rejecting whole corrupted windows
without per-problem ``f_scale`` tuning. Use the robust loss when a few windows
are mildly contaminated; reach for the ensemble when outliers are dense or the
loss is hard to scale.
"""

from typing import Any, Callable, cast

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from dtfit.log import echo
from dtfit._core._kernels import simpson_windows, simpson_windows_rows
from dtfit.types import FittingResult, InitialGuess
from ._common import model_params, _covariance, _validate_xy


def _dominant_cycles(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate how many full periods of the dominant tone the record spans.

    Uses the FFT peak above DC. Non-oscillatory shapes (a trend, a single peak,
    a sigmoid) concentrate their energy at/near DC and return ~0-1; a genuine
    oscillation returns roughly its cycle count. Used to auto-scale the window
    count so windows stay sub-period (see ``fit_eac``).
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


def fit_eac(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    active_ratio: float = 0.8,
    n_windows: int | None = None,
    window_mode: str = "uniform",
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
        active_ratio: Fraction of the (leading) data used for window placement in
            ``window_mode="uniform"``; the informative transient usually lives
            here. Ignored by ``"curvature"`` placement (which spans all the data).
        n_windows: Number of integration windows (area equations). Defaults to
            ``2 * n_params`` for an overdetermined, noise-averaging fit. Must be
            ``>= n_params``; clamped so each window keeps at least 3 samples.
        window_mode: Window placement -- ``"uniform"`` (equal spans over the
            active region, the default) or ``"curvature"`` (edges carrying equal
            cumulative absolute curvature: narrow where the signal bends, wide
            where it is smooth). ``"curvature"`` is better-conditioned for
            signals with a localized transient (step take-off, sharp turn, a
            peak's rise) -- validated as the best estimator on concentrated
            transients and rational-saturating shapes in the parameter-estimation
            domain study.
        bounds: Optional ``(lower, upper)`` parameter bounds (as accepted by
            ``scipy.optimize.least_squares``); switches the solver to
            trust-region.
        loss: Least-squares loss (e.g. ``"linear"`` or ``"soft_l1"`` for outlier
            robustness, as in the EAC paper). The loss acts on the *window-area*
            residuals, so it down-weights whole contaminated windows -- give it
            enough windows (``n_windows``) that outliers stay localized for it to
            bite.
        f_scale: Soft margin of the robust ``loss`` (``scipy``'s ``f_scale``):
            residuals below it stay quadratic, above it are down-weighted. The
            default ``1.0`` is far larger than typical small window-area
            residuals and so leaves a robust ``loss`` in its quadratic regime;
            **set it to the scale of a clean window's area residual** to actually
            engage the robustness (the paper tunes it per problem). Ignored when
            ``loss="linear"``.
        p0: Optional initial guess (defaults to ones).

    For dense contamination where ``f_scale`` is hard to tune, prefer
    :func:`dtfit.ensemble_fit` (median over overlapping windows) as a
    complementary robust path.

    Returns:
        FittingResult with the fitted coefficients, callable model and (when
        overdetermined) a parameter covariance estimate.
    """
    if window_mode not in ("uniform", "curvature"):
        raise ValueError(
            f"window_mode must be 'uniform' or 'curvature', got {window_mode!r}"
        )
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    n = len(params)
    if n == 0:
        raise RuntimeError("Model expression has no free parameters to fit.")

    x, y = _validate_xy(data_x, data_y, min_size=2 * n)

    model_func = sp.lambdify((t, *params), f_sym, "numpy")
    jac_funcs = [
        sp.lambdify((t, *params), sp.diff(f_sym, p), "numpy") for p in params
    ]

    # Contiguous window spans [start, stop) over the (active) region. The model
    # and its sensitivities are evaluated once over the whole region per solver
    # step and integrated per window by the (compiled) Simpson kernel, rather
    # than re-evaluated window by window.
    if window_mode == "curvature":
        # Information-adaptive edges over all the data: each window carries
        # roughly equal cumulative |curvature| (>= 3 samples for Simpson).
        x_active = np.ascontiguousarray(x)
        m = max(n, 2 * n if n_windows is None else int(n_windows))
        edges = _curvature_edges(x_active, y, m)
        starts = np.ascontiguousarray(edges[:-1], dtype=np.intp)
        stops = np.ascontiguousarray(edges[1:], dtype=np.intp)
        data_areas_arr = simpson_windows(
            np.ascontiguousarray(y), x_active, starts, stops
        )
        m = starts.size
    else:
        # Uniform spans over the leading active region (>= n for solvability,
        # default 2n for redundancy), each with at least 3 samples for Simpson.
        idx_max = max(int(x.size * active_ratio), n + 1)
        requested = 2 * n if n_windows is None else int(n_windows)
        if n_windows is None:
            # Oscillatory data: a window spanning whole periods integrates to
            # ~0, so its area is blind to amplitude/phase and the criterion
            # loses conditioning as the cycle count grows. Auto-scale to keep
            # windows sub-period (~3 per dominant cycle). Non-oscillatory shapes
            # have their FFT peak at/near DC (cycles ~ 0-1) and stay at 2n.
            cycles = _dominant_cycles(x[:idx_max], y[:idx_max])
            if cycles >= 2.0:
                requested = max(requested, int(np.ceil(3.0 * cycles)))
        m = max(n, min(requested, idx_max // 3))
        window = max(idx_max // m, 2)
        x_active = np.ascontiguousarray(x[:idx_max])
        starts = np.array([i * window for i in range(m)], dtype=np.intp)
        stops = np.array(
            [(i + 1) * window if i < m - 1 else idx_max for i in range(m)],
            dtype=np.intp,
        )
        data_areas_arr = simpson_windows(y[:idx_max], x_active, starts, stops)
    echo(f"EAC windows: {m} (params: {n}, mode: {window_mode})")

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
        kwargs: dict[str, Any] = {"loss": loss, "f_scale": f_scale}
        if bounds is not None:
            kwargs["bounds"] = bounds
        # cast: scipy's stub types `jac` as a str literal, omitting the
        # callable form the runtime accepts.
        sol = least_squares(
            residuals, guess, jac=cast(Any, jacobian), method="trf", **kwargs
        )
    else:
        sol = least_squares(
            residuals, guess, jac=cast(Any, jacobian), method="lm"
        )
    coeffs = np.asarray(sol.x, dtype=np.float64)
    echo("EAC fitted coefficients:", coeffs)

    cov = _covariance(sol.jac, sol.fun, n)

    model = sp.lambdify(t, f_sym.subs(dict(zip(params, coeffs))), "numpy")
    return FittingResult(model=model, coeffs=coeffs, cov=cov,
                         expr=expr, var=var, names=tuple(str(p) for p in params),
                         converged=bool(sol.success), message=str(sol.message),
                         x_range=(float(np.min(x)), float(np.max(x))))

