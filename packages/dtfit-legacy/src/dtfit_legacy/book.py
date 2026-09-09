"""The three criteria as the sources state them, before any improvement.

Sources: Pysarchuk, "Нелінійні та багатокритеріальні моделі", section 2.5
(spectrum balance, pp. 110-117), 2.6 (integral least squares in the
differential-transformation scheme, pp. 118-126) and 2.7 (equal areas in
shifted differential transformations, pp. 127-140), and the dissertation's
chapter 2, eqs 2.12 to 2.14 and 2.23 to 2.27.

All three share the intermediate polynomial: the least-squares polynomial
of the data in the monomial basis on the window shifted to start at zero.
Its ascending coefficients are the data's Maclaurin coefficients, the
differential spectrum with the scale ``H`` cancelled. The model is given in
the shifted variable ``t = x - x[0]`` and its own spectrum is the truncated
Maclaurin expansion. Every method needs a start for a model nonlinear in
its parameters; the balance is the one that can be solved symbolically.
"""

from __future__ import annotations

import functools

import numpy as np
import sympy as sp
from numpy.polynomial import polynomial as P
from scipy.linalg import LinAlgError, cholesky
from scipy.optimize import least_squares

from dtfit._input import _validate_p0, resolve_model, result_kwargs
from dtfit.types import FittingResult
from .dsb import fit_dsb
from .integral._common import model_params, taylor_coeffs


def intermediate_polynomial(
    x: np.ndarray, y: np.ndarray, degree: int
) -> tuple[np.ndarray, float]:
    """The least-squares polynomial of ``(x, y)`` on the window shifted to
    start at ``x[0]``.

    Args:
        x: sample positions, shape ``(n,)``, increasing.
        y: sample values, shape ``(n,)``.
        degree: polynomial degree, at least 0 and below ``n``.

    Returns:
        ``(d, H)``: ascending monomial coefficients ``d[k]`` of ``t**k`` with
        ``t = x - x[0]``, and the window length ``H = x[-1] - x[0]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = x - x[0]
    d = P.polyfit(t, y, int(degree))
    return np.asarray(d, dtype=float), float(t[-1])


def monomial_weight_matrix(h: float, order: int) -> np.ndarray:
    """The weight matrix of the integral criterion in the monomial basis,
    ``M[i, j] = H**(i+j+1) / (i+j+1)`` (dissertation eq 2.13), shape
    ``(order+1, order+1)``. Its condition number grows as ``exp(3.5 K)``."""
    i = np.arange(int(order) + 1)
    s = i[:, None] + i[None, :] + 1
    return float(h) ** s / s


@functools.lru_cache(maxsize=64)
def _spectrum(expr: str, var: str, order: int):
    """The model's parameter list and its truncated Maclaurin spectrum as a
    vectorized function of the parameters; the symbolic work is cached per
    ``(expr, var, order)``."""
    t = sp.Symbol(var)
    f = sp.sympify(expr)
    params = model_params(f, t)
    coeffs = taylor_coeffs(f, t, int(order))
    fn = sp.lambdify(params, coeffs, "numpy")

    def phi(theta: np.ndarray) -> np.ndarray:
        return np.asarray(fn(*theta), dtype=float).reshape(-1)

    return params, phi


def _result(
    coeffs: np.ndarray, expr: str, var: str, x: np.ndarray, y: np.ndarray,
    converged: bool, message: str, nfev: int,
) -> FittingResult:
    spec = resolve_model(expr, var)
    t = np.asarray(x, dtype=float) - float(np.asarray(x, dtype=float)[0])
    yv = np.asarray(y, dtype=float)
    f = spec.eval(t, coeffs)
    rss = float(np.sum((yv - f) ** 2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    return FittingResult(
        coeffs=np.asarray(coeffs, dtype=float), cov=None,
        converged=bool(converged), message=str(message),
        x_range=(0.0, float(t[-1])), n_obs=int(t.size), rss=rss, tss=tss,
        nfev=int(nfev), cost=0.5 * rss, **result_kwargs(spec, coeffs),
    )


def dsb_balance(
    x: np.ndarray, y: np.ndarray, expr: str, var: str, *, p0=None
) -> FittingResult:
    """Spectrum balance, the square system of the source (monograph 2.5).

    Steps as the source lists them: the model's differential spectrum to
    as many discretes as it has parameters; the data's spectrum from the
    intermediate polynomial of that degree; the discrete-by-discrete
    balance solved for the parameters (symbolically, with a numeric fall
    back from ``p0``).

    Args:
        x, y: samples; ``var`` is measured from ``x[0]``.
        expr: the model, a SymPy expression string in ``var``.
        p0: start for the numeric fallback; ``None`` is all ones.

    Returns:
        The balance's ``FittingResult``; ``cov`` is ``None`` for a square
        balance.
    """
    t = sp.Symbol(var)
    m = len(model_params(sp.sympify(expr), t))
    d, _ = intermediate_polynomial(x, y, m - 1)
    return fit_dsb(d, expr, var, rank=m, p0=p0)


def lsi_integral_monomial(
    x: np.ndarray, y: np.ndarray, expr: str, var: str, *,
    order: int | None = None, p0=None,
) -> FittingResult:
    """The integral least-squares criterion in the monomial basis
    (monograph 2.6; dissertation eq 2.13).

    Minimizes ``sum_ij M_ij (d_i - phi_i(c)) (d_j - phi_j(c))`` over the
    parameters ``c``, where ``d`` is the intermediate polynomial of degree
    ``order`` and ``phi(c)`` the model's truncated Maclaurin spectrum.

    Args:
        order: expansion order ``K``; ``None`` is twice the parameter count.
        p0: start; ``None`` is all ones.

    Raises:
        ValueError: the weight matrix is not positive definite at
            ``order``, the source form's breakdown near order 11.
    """
    t = sp.Symbol(var)
    m = len(model_params(sp.sympify(expr), t))
    k = 2 * m if order is None else int(order)
    d, h = intermediate_polynomial(x, y, k)
    params, phi = _spectrum(expr, var, k)
    mat = monomial_weight_matrix(h, k)
    try:
        lower = cholesky(mat, lower=True)
    except LinAlgError as exc:
        raise ValueError(
            f"the monomial weight matrix is not positive definite at order "
            f"{k} (condition {np.linalg.cond(mat):.1e}); the source form "
            "breaks down near order 11"
        ) from exc

    def residual(theta: np.ndarray) -> np.ndarray:
        return lower.T @ (d - phi(theta))

    guess = _validate_p0(p0, params)
    sol = least_squares(residual, guess, method="lm")
    return _result(sol.x, expr, var, x, y, sol.success, sol.message,
                   sol.nfev)


def eac_areas(
    x: np.ndarray, y: np.ndarray, expr: str, var: str, *,
    order: int | None = None, p0=None,
) -> FittingResult:
    """The equal-areas criterion as the source states it (monograph 2.7;
    dissertation 2.3.1 and 2.3.2 before the direct reformulation).

    One window per parameter, equal windows over ``[0, H]``. The data's
    area on a window comes from the intermediate polynomial; the model's
    from its truncated Maclaurin spectrum, integrated term by term. The
    square system of equal areas is solved from ``p0``. The truncation is
    the source of the bias outside the spectrum's convergence radius
    (dissertation figure 2.4).

    Args:
        order: expansion order ``K``; ``None`` is twice the parameter count.
        p0: start; ``None`` is all ones.
    """
    t = sp.Symbol(var)
    m = len(model_params(sp.sympify(expr), t))
    k = 2 * m if order is None else int(order)
    d, h = intermediate_polynomial(x, y, k)
    params, phi = _spectrum(expr, var, k)
    edges = np.linspace(0.0, h, m + 1)
    powers = np.arange(k + 1) + 1
    areas = (edges[1:, None] ** powers - edges[:-1, None] ** powers) / powers
    data_areas = areas @ d

    def residual(theta: np.ndarray) -> np.ndarray:
        return areas @ phi(theta) - data_areas

    guess = _validate_p0(p0, params)
    sol = least_squares(residual, guess, method="lm")
    return _result(sol.x, expr, var, x, y, sol.success, sol.message,
                   sol.nfev)
