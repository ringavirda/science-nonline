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

from typing import cast

import numpy as np
import sympy as sp
from numpy.polynomial import Legendre
from scipy.optimize import least_squares

from dtfit.log import echo
from dtfit._core._kernels import legendre_project
from dtfit._core._spectral import solve_weighted_nlls
from dtfit.types import FittingResult, InitialGuess
from ._common import (
    model_params, _covariance, information_criteria, _validate_xy, _validate_p0,
    _savgol_prefilter,
)


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
    """Spectral order high enough to resolve the dominant cycle. A degree-``k``
    polynomial only starts to represent a sinusoid once ``k`` exceeds its mapped
    wavenumber ``pi * cycles`` (the classical Legendre/Chebyshev resolution
    threshold; below it the data and model spectra both collapse toward zero and
    the amplitude is left under-constrained). The earlier ``1.4 * cycles``
    heuristic fell under this threshold for mid-range cycle counts, leaving a
    band of degraded amplitude recovery; ``pi * cycles`` plus headroom clears it
    for every cycle count."""
    w0 = fft_frequency_seed(x, y)
    cycles = w0 * float(x[-1] - x[0]) / (2.0 * np.pi)
    return min(int(np.ceil(np.pi * cycles)) + 8, max_order)


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
    k_star: int | str | None = None,
    alpha: float = 0.0,
    filter_data: bool = True,
    bounds: list[tuple[float, float]] | None = None,
    p0: InitialGuess = None,
    oscillatory: bool = False,
    freq_param: str | None = None,
    random_state: int | None = 0,
    robust: bool = False,
    huber_c: float = 3.0,
    nan_policy: str = "raise",
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` using the integral least-squares
    criterion in the reconditioned (Legendre) differential-transformation scheme.

    Args:
        data_x, data_y: Observed samples.
        expr: Model expression, e.g. ``"a0 + a1*x + a2*exp(a3*x)"``.
        var: Main variable name in ``expr``.
        k_star: Number of spectral coefficients (Legendre order) to match. An int
            sets the order explicitly; ``"auto"`` selects it by BIC of the data
            fit; ``None`` (default) uses order 5, or -- under the oscillatory
            recipe -- an order high enough to resolve the dominant cycle
            (:func:`_osc_order`).
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
            ``filter_data=False`` and -- unless ``k_star`` is given explicitly --
            raises the spectral order to resolve the dominant cycle
            (:func:`_osc_order`). A sinusoid recovers to <1% with this recipe
            versus ~50% without it. Passing ``freq_param`` implies this recipe;
            pass ``oscillatory=True`` to enable it without naming a frequency
            parameter.
        freq_param: Name of the model's **angular-frequency** parameter. When
            given, its initial guess is seeded from the data's FFT peak
            (:func:`fft_frequency_seed`) -- the seed the local (no-bounds) solve
            needs to lock onto the right cycle. Implies the oscillatory recipe.
        random_state: Seed for the global (differential-evolution) search used on
            the bounded path, for reproducibility; ``None`` draws from NumPy's
            global RNG (non-deterministic). Defaults to ``0`` (deterministic). Has
            no effect on the unbounded local solve, which is already deterministic.
        robust: If True, robustify the **empirical spectrum** via IRLS: each
            sample's residual to the current model is winsorized within ``huber_c``
            robust sigmas (MAD) before re-projecting, so an outlier sample cannot
            distort the Legendre coefficients. Forces ``filter_data=False`` (a
            linear pre-smoother would smear outliers first). The robust integral
            lever for LSI -- a handful of cheap re-solves, no tuning.
        huber_c: Robust winsorization threshold in residual sigmas for
            ``robust=True`` (~3 leaves clean samples untouched).

    Returns:
        FittingResult with the fitted coefficients, callable model and a
        parameter covariance estimate.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    if not params:
        raise RuntimeError("Model expression has no free parameters to fit.")

    x, y = _validate_xy(data_x, data_y, nan_policy=nan_policy)

    oscillatory = oscillatory or freq_param is not None
    if oscillatory:
        # The cycle lives in the high-order spectrum and is destroyed by
        # smoothing; force the recipe (raise the order only when the caller left
        # it unset, i.e. ``k_star is None``).
        filter_data = False
        if k_star is None:
            k_star = _osc_order(x, y)
    if robust:
        # Robust mode winsorizes each sample toward the model (below); a linear
        # pre-smoother would smear an outlier across its neighbours first, so turn
        # it off and let the winsorization handle the noise + outliers.
        filter_data = False

    # 1. Optional smoothing to tame noise before the spectral projection.
    if filter_data:
        y = _savgol_prefilter(y)

    if k_star is None:
        order = 5  # the conditioned default Legendre order
    elif k_star == "auto":
        order = _auto_order(x, y)
    else:
        order = int(k_star)
    # The spectral residual has order+1 entries and must carry at least as many
    # equations as parameters, or the least-squares is underdetermined and LM
    # raises. Floor the order at n_params-1 so every catalogue model is solvable
    # at the default k_star (e.g. an 8-parameter Fourier series needs order>=7).
    order = max(order, len(params) - 1)
    order = max(1, min(order, y.size - 2))
    if order + 1 < len(params):
        # The sample count clamped the order below n_params-1, so the spectral
        # residual (order+1 entries) carries fewer equations than parameters and
        # LM would raise a cryptic 'm < n'. Fail clearly instead (mirrors EAC's
        # 2*n_params guard).
        raise ValueError(
            f"fit_lsi needs at least {len(params) + 1} samples to fit "
            f"{len(params)} parameters (got {y.size}); the spectral match "
            "would be underdetermined."
        )
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

    guess = _validate_p0(p0, params)

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

    # 5. Solve via the shared weighted-NLLS driver (the same bounded-local /
    #    global-DE-fallback / unbounded-LM logic behind the promoted map-reduce
    #    fitters -- one implementation, so the two cannot drift). Without bounds it
    #    is a weighted LM from the seed; with bounds a fast bounded local solve
    #    from a good seed (catalog/oscillatory paths give data-driven seeds, e.g.
    #    an FFT peak for the frequency) at ~10-50x less cost than the global
    #    differential-evolution fallback that guards the multimodal cases.
    coeffs, jac, converged, message = solve_weighted_nlls(
        residual, sqrt_w, beta_data, guess, p0=p0, bounds=bounds, seed=random_state
    )

    if robust:
        # Robust integral via IRLS: winsorize each sample's residual to the
        # current model (within huber_c robust sigmas) and recompute the empirical
        # Legendre spectrum, so an outlier sample cannot distort the projection.
        # ``beta_data`` is reassigned and the ``residual`` closure reads it lazily.
        for _ in range(3):
            mv = np.asarray(f_func(x, *coeffs), dtype=float)
            if mv.ndim == 0:
                mv = np.full_like(x, float(mv))
            r = y - mv
            med = float(np.median(r))
            sigma = 1.4826 * float(np.median(np.abs(r - med)))
            if sigma <= 0.0:
                break
            clip = huber_c * sigma
            y_eff = mv + (med + np.clip(r - med, -clip, clip))
            beta_data = Legendre.fit(x, y_eff, order, domain=domain).coef
            if bounds is not None:
                sol = least_squares(residual, coeffs,
                                    bounds=([b[0] for b in bounds],
                                            [b[1] for b in bounds]), method="trf")
            else:
                sol = least_squares(residual, coeffs, method="lm")
            coeffs = np.asarray(sol.x, dtype=np.float64)
            jac = sol.jac
        converged, message = True, "robust IRLS"

    echo("LSI fitted coefficients:", coeffs)
    cov = _covariance(jac, residual(coeffs), len(params))

    # FittingResult lambdifies the fitted model lazily from expr+coeffs (and
    # drops it on pickling); skip the eager compile here.
    return FittingResult(coeffs=coeffs, cov=cov,
                         expr=expr, var=var, names=tuple(str(p) for p in params),
                         converged=converged, message=message,
                         x_range=(float(np.min(x)), float(np.max(x))))
