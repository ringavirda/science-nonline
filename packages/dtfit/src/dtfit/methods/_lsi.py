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

from typing import Any, cast

import numpy as np
import sympy as sp
from numpy.polynomial import Legendre
from scipy.optimize import least_squares, differential_evolution, minimize
from scipy.signal import savgol_filter

from dtfit.log import echo
from dtfit._core._kernels import legendre_project
from dtfit.types import FittingResult, InitialGuess
from ._common import model_params, _covariance, information_criteria


def fft_frequency_seed(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant **angular** frequency of ``y`` over uniform grid ``x``.

    The peak of the (mean-removed) real FFT, returned as ``2*pi*f`` -- the seed
    for the angular-frequency parameter of an oscillatory model. This is the
    ``freq_param`` initial guess used by :func:`fit_lsi`'s oscillatory recipe;
    a sinusoid's frequency cannot be recovered by a smoothed low-order spectral
    fit without it (see the parameter-estimation / forecasting domain studies).
    Assumes (near-)uniform sampling; the spacing is read from ``x[1] - x[0]``.
    """
    yy = np.asarray(y, dtype=float) - float(np.mean(y))
    spec = np.abs(np.fft.rfft(yy))
    if spec.size:
        spec[0] = 0.0  # ignore the DC bin
    freqs = np.fft.rfftfreq(yy.size, d=float(x[1] - x[0]))
    return 2.0 * np.pi * float(freqs[int(np.argmax(spec))])


def _osc_order(x: np.ndarray, y: np.ndarray, max_order: int = 200) -> int:
    """Spectral order high enough to resolve the dominant cycle: ~1.4 cycles
    over the window plus headroom (the recipe the domain studies validated)."""
    w0 = fft_frequency_seed(x, y)
    cycles = w0 * float(x[-1] - x[0]) / (2.0 * np.pi)
    return min(int(1.4 * cycles) + 10, max_order)


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
        _, bic = information_criteria(rss, n, k + 1)
        if bic < best_bic:
            best_order, best_bic = k, bic
    return best_order


def fit_lsi(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    k_star: int | str = 5,
    alpha: float = 0.0,
    filter_data: bool = True,
    bounds: list[tuple[float, float]] | None = None,
    p0: InitialGuess = None,
    oscillatory: bool = False,
    freq_param: str | None = None,
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` using the integral least-squares
    criterion in the reconditioned (Legendre) differential-transformation scheme.

    Args:
        data_x, data_y: Observed samples.
        expr: Model expression, e.g. ``"a0 + a1*x + a2*exp(a3*x)"``.
        var: Main variable name in ``expr``.
        k_star: Number of spectral coefficients (Legendre order) to match, or
            ``"auto"`` to select it by BIC of the data fit.
        alpha: Extra exponential down-weight ``exp(-alpha*j)`` on high-order
            coefficients, on top of the built-in ``1/(2j+1)`` orthonormal weight.
            Defaults to 0 (the orthogonal basis already tames high orders).
        filter_data: Apply a Savitzky-Golay pre-filter to the data.
        bounds: Optional per-parameter ``(min, max)`` bounds. When given a global
            search (differential evolution) is run before local refining.
        p0: Optional initial guess (defaults to ones).
        oscillatory: Apply the **oscillatory recipe** validated across the
            forecasting and parameter-estimation domain studies: a smoothed,
            low-order spectral fit erases a cycle, so this forces
            ``filter_data=False`` and -- unless ``k_star`` is given explicitly
            (left at its default ``5``) -- raises the spectral order to resolve
            the dominant cycle (:func:`_osc_order`). A sinusoid recovers to <1%
            with this recipe versus ~50% without it.
        freq_param: Name of the model's **angular-frequency** parameter. When
            given, its initial guess is seeded from the data's FFT peak
            (:func:`fft_frequency_seed`) -- the seed the local (no-bounds) solve
            needs to lock onto the right cycle. Implies the oscillatory recipe.

    Returns:
        FittingResult with the fitted coefficients, callable model and a
        parameter covariance estimate.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    if not params:
        raise RuntimeError("Model expression has no free parameters to fit.")

    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)

    oscillatory = oscillatory or freq_param is not None
    if oscillatory:
        # The cycle lives in the high-order spectrum and is destroyed by
        # smoothing; force the recipe (raise order only if k_star is default).
        filter_data = False
        if k_star == 5:
            k_star = _osc_order(x, y)

    # 1. Optional smoothing to tame noise before the spectral projection.
    if filter_data and y.size >= 5:
        window = min(11, y.size if y.size % 2 == 1 else y.size - 1)
        if window > 3:
            y = np.asarray(savgol_filter(y, window, polyorder=3), dtype=float)

    order = _auto_order(x, y) if k_star == "auto" else int(k_star)
    # The spectral residual has order+1 entries and must carry at least as many
    # equations as parameters, or the least-squares is underdetermined and LM
    # raises. Floor the order at n_params-1 so every catalogue model is solvable
    # at the default k_star (e.g. an 8-parameter Fourier series needs order>=7).
    order = max(order, len(params) - 1)
    order = max(1, min(order, y.size - 2))
    echo(f"LSI Legendre spectral order: {order}")

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
        return legendre_project(fv, weights, legvander, norm)

    # 4. Diagonal orthonormal weight: ∫(data-model)^2 dt = Σ H/(2j+1) Δβ_j^2,
    #    with an optional extra exponential down-weight.
    j = np.arange(order + 1)
    sqrt_w = np.sqrt((h / (2.0 * j + 1.0)) * np.exp(-alpha * j))

    def residual(c: np.ndarray) -> np.ndarray:
        spec = model_spectrum(c)
        if not np.all(np.isfinite(spec)):
            return np.full(order + 1, 1e6)
        return sqrt_w * (beta_data - spec)

    guess = np.ones(len(params)) if p0 is None else np.asarray(p0, float).copy()

    # Seed the angular-frequency parameter from the data's FFT peak so the local
    # solve locks onto the right cycle (the bounds path runs a global search and
    # ignores the seed, which is fine -- there the bounds bracket the frequency).
    if freq_param is not None:
        names = [str(p) for p in params]
        if freq_param not in names:
            raise ValueError(
                f"freq_param {freq_param!r} is not a parameter of the model "
                f"(have {names})."
            )
        guess[names.index(freq_param)] = fft_frequency_seed(x, y)

    # 5. Solve. Without bounds: weighted NLLS from the seed. With bounds: a fast
    #    bounded *local* solve from a supplied seed first -- the catalog seeders
    #    and the auto/oscillatory paths give good, data-driven seeds (an FFT peak
    #    for the frequency), so a trust-region solve from there lands on the same
    #    optimum as the global search at ~10-50x less cost. The global
    #    differential-evolution search is kept as the fallback for when no seed is
    #    supplied or the local solve lands on a poor basin (large residual).
    if bounds is not None:
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        local = None
        if p0 is not None:
            loc = least_squares(
                residual, np.clip(guess, lo, hi), bounds=(lo, hi), method="trf"
            )
            denom = float(np.linalg.norm(sqrt_w * beta_data)) + 1e-30
            rel = float(np.linalg.norm(loc.fun)) / denom
            if loc.success and rel < 0.5:  # converged + explains the spectrum
                local = loc
        if local is not None:
            coeffs = np.asarray(local.x, dtype=np.float64)
            jac = local.jac
        else:
            def cost(c: np.ndarray) -> float:
                r = residual(c)
                return float(r @ r)

            # cast: `seed` is the portable arg across scipy versions (newer stubs
            # only expose its `rng` successor).
            res_g = cast(Any, differential_evolution)(
                cost, bounds, strategy="best1bin", popsize=15, seed=0
            )
            res = minimize(cost, res_g.x, method="L-BFGS-B", bounds=bounds)
            coeffs = np.asarray(res.x, dtype=np.float64)
            jac = _numeric_jac(residual, coeffs)
    else:
        sol = least_squares(residual, guess, method="lm")
        coeffs = np.asarray(sol.x, dtype=np.float64)
        jac = sol.jac

    echo("LSI fitted coefficients:", coeffs)
    cov = _covariance(jac, residual(coeffs), len(params))

    model = sp.lambdify(t, f_sym.subs(dict(zip(params, coeffs))), "numpy")
    return FittingResult(model=model, coeffs=coeffs, cov=cov,
                         expr=expr, var=var, names=tuple(str(p) for p in params))


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
