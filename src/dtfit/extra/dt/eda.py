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

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.integrate import simpson

from dtfit.helpers import FittingResult, FittingOptions, echo_if
from .taylor import model_params


def fit_eda(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    options: FittingOptions = FittingOptions(),
    *,
    active_ratio: float = 0.8,
    n_windows: int | None = None,
    bounds: tuple | None = None,
    loss: str = "linear",
    p0: np.ndarray | None = None,
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` with the equal-areas criterion.

    Args:
        data_x, data_y: Observed samples.
        expr: Model expression, e.g. ``"a * atan(w * t)"``.
        var: Main variable name in ``expr``.
        options: Fitting options (used for echo logging).
        active_ratio: Fraction of the (leading) data used for window placement;
            the informative transient usually lives here.
        n_windows: Number of integration windows (area equations). Defaults to
            ``2 * n_params`` for an overdetermined, noise-averaging fit. Must be
            ``>= n_params``; clamped so each window keeps at least 3 samples.
        bounds: Optional ``(lower, upper)`` parameter bounds (as accepted by
            ``scipy.optimize.least_squares``); switches the solver to
            trust-region.
        loss: Least-squares loss (e.g. ``"linear"`` or ``"soft_l1"`` for outlier
            robustness).
        p0: Optional initial guess (defaults to ones).

    Returns:
        FittingResult with the fitted coefficients, callable model and (when
        overdetermined) a parameter covariance estimate.
    """
    t = sp.Symbol(var)
    f_sym = sp.sympify(expr)
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
    m = max(n, min(requested, idx_max // 3))
    window = max(idx_max // m, 2)

    windows: list[np.ndarray] = []
    data_areas: list[float] = []
    for i in range(m):
        start = i * window
        end = (i + 1) * window if i < m - 1 else idx_max
        xi, yi = x[start:end], y[start:end]
        windows.append(xi)
        data_areas.append(float(simpson(y=yi, x=xi)))
    data_areas_arr = np.asarray(data_areas)
    echo_if(options, f"EDA windows: {m} (params: {n})")

    def residuals(c: np.ndarray) -> np.ndarray:
        return np.array(
            [
                simpson(y=model_func(windows[i], *c), x=windows[i])
                for i in range(m)
            ]
        ) - data_areas_arr

    def jacobian(c: np.ndarray) -> np.ndarray:
        jac = np.zeros((m, n))
        for i in range(m):
            xi = windows[i]
            for j in range(n):
                d = jac_funcs[j](xi, *c)
                if np.isscalar(d):
                    d = np.full_like(xi, float(d))
                jac[i, j] = simpson(y=d, x=xi)
        return jac

    guess = np.ones(n) if p0 is None else np.asarray(p0, float)
    if bounds is not None or loss != "linear":
        method = "trf"
        kwargs = {"loss": loss}
        if bounds is not None:
            kwargs["bounds"] = bounds
        sol = least_squares(residuals, guess, jac=jacobian, method=method, **kwargs)
    else:
        sol = least_squares(residuals, guess, jac=jacobian, method="lm")
    coeffs = np.asarray(sol.x, dtype=np.float64)
    echo_if(options, "EDA fitted coefficients:", coeffs)

    cov = _covariance(sol.jac, sol.fun, n)

    model = sp.lambdify(t, f_sym.subs(dict(zip(params, coeffs))), "numpy")
    return FittingResult(model=model, coeffs=coeffs, cov=cov)


def _covariance(
    jac: np.ndarray, res: np.ndarray, n_params: int
) -> np.ndarray | None:
    """Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` from the area-residual
    Jacobian at the solution (overdetermined case only)."""
    m = res.size
    if m <= n_params:
        return None
    jtj = jac.T @ jac
    try:
        jtj_inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return None
    sigma2 = float(res @ res) / (m - n_params)
    return sigma2 * jtj_inv
