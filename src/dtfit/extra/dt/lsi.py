"""LSI -- Least Squares Integral.

Numeric successor to the symbolic DSBI method. Fits a model that is nonlinear in
its parameters by minimizing the integral least-squares discrepancy between the
data and the model over the observation interval.

Reconditioned formulation
--------------------------
The original LSI matched **monomial Maclaurin** coefficients with an integral
weighting ``M[i,j] = w_i w_j H^(i+j+1)/(i+j+1)`` -- a (weighted) *Hilbert*
matrix, whose condition number explodes with the order, and it extracted the
empirical spectrum with a raw-Vandermonde ``numpy.polyfit``, also ill-
conditioned. Both are replaced here by an **orthogonal-polynomial (Legendre)
spectrum** on the data interval:

* the empirical spectrum is the Legendre fit of the data (well-conditioned
  least squares on an orthogonal basis, not a raw Vandermonde);
* the model spectrum is the model's Legendre projection, evaluated by
  Gauss-Legendre quadrature (so the model is integrated exactly rather than
  Taylor-truncated);
* because the Legendre polynomials are orthogonal on the interval, the integral
  criterion ``∫ (data - model)^2 dt`` becomes a **diagonal** sum of squared
  coefficient residuals ``Σ_j H/(2j+1) (β_j^data - β_j^model)^2`` -- the Hilbert
  matrix collapses to a diagonal weight, so the problem is perfectly
  conditioned.

This works directly on raw ``(x, y)`` data and returns a parameter covariance
estimate alongside the coefficients.
"""

import numpy as np
import sympy as sp
from numpy.polynomial import Legendre
from scipy.optimize import least_squares, differential_evolution, minimize
from scipy.signal import savgol_filter

from dtfit.helpers import FittingResult, FittingOptions, echo_if
from .taylor import model_params


def _auto_order(x: np.ndarray, y: np.ndarray, max_order: int = 8) -> int:
    """Pick the Legendre degree that minimizes BIC of the data fit -- enough
    spectral resolution to capture the signal without fitting noise."""
    n = y.size
    best_order, best_bic = 1, np.inf
    for k in range(1, min(max_order, n - 1) + 1):
        resid = y - Legendre.fit(x, y, k)(x)
        rss = float(resid @ resid)
        if rss <= 0:
            return k
        bic = n * np.log(rss / n) + (k + 1) * np.log(n)
        if bic < best_bic:
            best_order, best_bic = k, bic
    return best_order


def fit_lsi(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    options: FittingOptions = FittingOptions(),
    *,
    k_star: int | str = 5,
    alpha: float = 0.0,
    filter_data: bool = True,
    bounds: list[tuple[float, float]] | None = None,
    p0: np.ndarray | None = None,
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` using the integral least-squares
    criterion in the reconditioned (Legendre) differential-transformation scheme.

    Args:
        data_x, data_y: Observed samples.
        expr: Model expression, e.g. ``"a0 + a1*x + a2*exp(a3*x)"``.
        var: Main variable name in ``expr``.
        options: Fitting options (used for echo logging).
        k_star: Number of spectral coefficients (Legendre order) to match, or
            ``"auto"`` to select it by BIC of the data fit.
        alpha: Extra exponential down-weight ``exp(-alpha*j)`` on high-order
            coefficients, on top of the built-in ``1/(2j+1)`` orthonormal weight.
            Defaults to 0 (the orthogonal basis already tames high orders).
        filter_data: Apply a Savitzky-Golay pre-filter to the data.
        bounds: Optional per-parameter ``(min, max)`` bounds. When given a global
            search (differential evolution) is run before local refining.
        p0: Optional initial guess (defaults to ones).

    Returns:
        FittingResult with the fitted coefficients, callable model and a
        parameter covariance estimate.
    """
    t = sp.Symbol(var)
    f_sym = sp.sympify(expr)
    params = model_params(f_sym, t)
    if not params:
        raise RuntimeError("Model expression has no free parameters to fit.")

    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)

    # 1. Optional smoothing to tame noise before the spectral projection.
    if filter_data and y.size >= 5:
        window = min(11, y.size if y.size % 2 == 1 else y.size - 1)
        if window > 3:
            y = savgol_filter(y, window, polyorder=3)

    order = _auto_order(x, y) if k_star == "auto" else int(k_star)
    order = max(1, min(order, y.size - 2))
    echo_if(options, f"LSI Legendre spectral order: {order}")

    x0, xn = float(x[0]), float(x[-1])
    h = xn - x0
    domain = [x0, xn]

    # 2. Empirical spectrum: Legendre coefficients of the data (conditioned).
    beta_data = Legendre.fit(x, y, order, domain=domain).coef

    # 3. Model spectrum via Gauss-Legendre quadrature (model integrated exactly).
    n_quad = max(2 * (order + 1), 16)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    t_quad = x0 + h * (nodes + 1.0) / 2.0
    # P_j(node) for j = 0..order:  legvander gives columns of Legendre polys.
    legvander = np.polynomial.legendre.legvander(nodes, order)  # (n_quad, order+1)
    norm = (2.0 * np.arange(order + 1) + 1.0) / 2.0  # standard-Legendre scaling
    f_func = sp.lambdify((t, *params), f_sym, "numpy")

    def model_spectrum(c: np.ndarray) -> np.ndarray:
        fv = np.asarray(f_func(t_quad, *c), dtype=float)
        if fv.ndim == 0:
            fv = np.full_like(t_quad, float(fv))
        return norm * ((weights * fv) @ legvander)

    # 4. Diagonal orthonormal weight: ∫(data-model)^2 dt = Σ H/(2j+1) Δβ_j^2,
    #    with an optional extra exponential down-weight.
    j = np.arange(order + 1)
    sqrt_w = np.sqrt((h / (2.0 * j + 1.0)) * np.exp(-alpha * j))

    def residual(c: np.ndarray) -> np.ndarray:
        spec = model_spectrum(c)
        if not np.all(np.isfinite(spec)):
            return np.full(order + 1, 1e6)
        return sqrt_w * (beta_data - spec)

    guess = np.ones(len(params)) if p0 is None else np.asarray(p0, float)

    # 5. Solve. With bounds use a global search; otherwise weighted NLLS.
    if bounds is not None:
        def cost(c: np.ndarray) -> float:
            r = residual(c)
            return float(r @ r)

        res_g = differential_evolution(
            cost, bounds, strategy="best1bin", popsize=15, seed=0
        )
        res = minimize(cost, res_g.x, method="L-BFGS-B", bounds=bounds)
        coeffs = np.asarray(res.x, dtype=np.float64)
        jac = _numeric_jac(residual, coeffs)
    else:
        sol = least_squares(residual, guess, method="lm")
        coeffs = np.asarray(sol.x, dtype=np.float64)
        jac = sol.jac

    echo_if(options, "LSI fitted coefficients:", coeffs)
    cov = _covariance(jac, residual(coeffs), len(params))

    model = sp.lambdify(t, f_sym.subs(dict(zip(params, coeffs))), "numpy")
    return FittingResult(model=model, coeffs=coeffs, cov=cov)


def _numeric_jac(residual, c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Forward-difference Jacobian of a residual vector at ``c``."""
    r0 = residual(c)
    jac = np.empty((r0.size, c.size))
    for k in range(c.size):
        step = eps * max(1.0, abs(c[k]))
        cp = c.copy()
        cp[k] += step
        jac[:, k] = (residual(cp) - r0) / step
    return jac


def _covariance(
    jac: np.ndarray, res: np.ndarray, n_params: int
) -> np.ndarray | None:
    """Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` from the residual
    Jacobian, scaled by the reduced chi-square."""
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
