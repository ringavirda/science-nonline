"""Shared internals of the differential-transformation fitting methods.

Collected here so the method modules (``lsi`` / ``eda`` / ``dsb``) draw on one
small toolbox rather than a scatter of one-function files:

- symbolic spectrum helpers: :func:`model_params`, :func:`taylor_coeffs`;
- numeric statistics: :func:`_covariance`, :func:`information_criteria`;
- polynomial-degree selection (the DSB pre-fit support): :func:`find_degree`.
"""

from typing import cast

import numpy as np
import sympy as sp

from dtfit.log import echo


# symbolic (Taylor / Maclaurin) spectrum helpers
#
# The differential transform of ``f`` about ``t0=0`` with sampling interval ``H``
# is ``F(k) = (H**k / k!) * f^(k)(0)``. In a *spectra balance* every equation sets
# a model discrete equal to the data discrete at the same order ``k``, so the
# common ``H**k`` factor cancels and the balance reduces to matching plain Taylor
# (Maclaurin) coefficients ``f^(k)(0)/k!`` -- available for any expression SymPy
# can differentiate, with no hand-written per-function discrete rules.
def model_params(f_sym: sp.Expr, t: sp.Symbol) -> list[sp.Symbol]:
    """Return the free parameters of ``f_sym`` (all symbols except ``t``),
    ordered by name for a stable coefficient layout."""
    params = sorted((s for s in f_sym.free_symbols if s != t), key=str)
    return cast("list[sp.Symbol]", params)


def taylor_coeffs(f_sym: sp.Expr, t: sp.Symbol, order: int) -> list[sp.Expr]:
    """Symbolic Maclaurin coefficients ``a_k = f^(k)(0) / k!`` for
    ``k = 0 .. order`` (inclusive) -- the ``H``-free differential spectrum."""
    coeffs: list[sp.Expr] = []
    deriv = f_sym
    for k in range(order + 1):
        coeffs.append(sp.simplify(deriv.subs(t, 0) / sp.factorial(k)))
        if k < order:
            deriv = sp.diff(deriv, t)
    return coeffs


# numeric statistics
def _covariance(
    jac: np.ndarray, res: np.ndarray, n_params: int
) -> np.ndarray | None:
    """Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` from the residual
    Jacobian, scaled by the reduced chi-square.

    Returns ``None`` for an exactly- or under-determined system (``m <=
    n_params``) or when ``J^T J`` is singular.
    """
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


def information_criteria(rss: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian-likelihood ``(AIC, BIC)`` from a residual sum of squares.

    ``AIC = n*ln(rss/n) + 2k`` and ``BIC = n*ln(rss/n) + k*ln(n)`` for ``n``
    samples and ``k`` parameters. A perfect fit (``rss <= 0``) returns ``-inf``.
    The single source of truth for the criteria used by degree selection, the
    LSI spectral-order pick and the diagnostics report.
    """
    if rss <= 0:
        return float("-inf"), float("-inf")
    base = n * np.log(rss / n)
    return float(base + 2 * k), float(base + k * np.log(n))


# polynomial-degree selection (DSB pre-fit support)
def find_degree(
    data_x: np.ndarray,
    data_y: np.ndarray,
    method: str = "bic",
    max_degree: int = 12,
) -> int:
    """Select a polynomial degree for ``(data_x, data_y)`` by ``"bic"``/``"aic"``.

    Returns the degree minimizing the chosen information criterion over
    ``0..max_degree`` (a parsimonious fit-vs-complexity trade-off).
    """
    if method not in ("bic", "aic"):
        raise ValueError(
            f"Unsupported degree-selection method {method!r}; use 'bic' or 'aic'."
        )
    degree = _find_degree_direct(data_x, data_y, max_degree, method)
    if degree == max_degree:
        echo(f"Warning: maximum degree {max_degree} reached.")
    echo(f"Best polynomial degree selected by {method}: {degree}")
    return degree


def _find_degree_direct(
    data_x: np.ndarray,
    data_y: np.ndarray,
    max_degree: int,
    criterion: str,
) -> int:
    """Degree minimizing AIC/BIC computed from the residual sum of squares."""
    if data_y is None or data_y.size == 0:
        raise ValueError("data must be a non-empty 1-D array")

    n = data_y.size
    max_degree = int(min(max_degree, max(0, n - 1)))
    best_degree, best_score = 0, np.inf

    for deg in range(0, max_degree + 1):
        try:
            coeffs = np.polyfit(data_x, data_y, deg)
        except Exception:
            continue
        model = np.poly1d(coeffs)(data_x)
        rss = float(np.sum((data_y - model) ** 2))
        if rss <= 0:
            return deg
        aic, bic = information_criteria(rss, n, deg + 1)
        score = aic if criterion == "aic" else bic
        if score < best_score:
            best_degree, best_score = deg, score

    return int(best_degree)
