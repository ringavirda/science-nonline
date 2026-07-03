"""Shared internals of the differential-transformation fitting methods.

Collected here so the method modules (``lsi`` / ``eac`` / ``dsb``) draw on one
small toolbox rather than a scatter of one-function files:

- symbolic spectrum helpers: :func:`model_params`, :func:`taylor_coeffs`;
- numeric statistics: :func:`_covariance`, :func:`information_criteria`;
- polynomial-degree selection (the DSB pre-fit support): :func:`find_degree`.
"""

from typing import cast

import numpy as np
import sympy as sp
from scipy.signal import savgol_filter

from dtfit.log import echo


def _savgol_prefilter(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay pre-smoother applied before an LSI spectral projection
    (window <= 11, cubic polyorder); a no-op for series shorter than 5 samples.

    Shared by :func:`dtfit.fit_lsi` and the experimental basis-LSI fitter so the
    smoothing heuristic lives in one place.
    """
    y = np.asarray(y, dtype=float)
    if y.size >= 5:
        window = min(11, y.size if y.size % 2 == 1 else y.size - 1)
        if window > 3:
            return np.asarray(savgol_filter(y, window, polyorder=3), dtype=float)
    return y


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


# input validation (shared by the batch fitters)
def _validate_xy(
    data_x: np.ndarray,
    data_y: np.ndarray,
    *,
    min_size: int = 2,
    nan_policy: str = "raise",
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and validate a 1-D ``(x, y)`` sample pair.

    Returns float arrays. Raises :class:`ValueError` with a clear message on a
    shape mismatch, a non-1-D input, or fewer than ``min_size`` samples -- so
    every batch fitter (``fit_lsi`` / ``fit_eac``) rejects malformed input the
    same way instead of failing obscurely deeper in.

    ``nan_policy`` controls non-finite handling: ``"raise"`` (default) rejects any
    NaN/inf; ``"omit"`` drops the offending ``(x, y)`` pairs before fitting, so
    gappy real-world telemetry (dropped GPS/sensor samples) can be fit directly
    without external masking. The finite-count floor still applies after omission.
    """
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"data_x and data_y must be 1-D; got shapes {x.shape} and {y.shape}."
        )
    if x.size != y.size:
        raise ValueError(
            f"data_x and data_y must have the same length; got {x.size} and {y.size}."
        )
    if nan_policy not in ("raise", "omit"):
        raise ValueError(
            f"nan_policy must be 'raise' or 'omit', got {nan_policy!r}."
        )
    if nan_policy == "omit":
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
    if x.size < min_size:
        raise ValueError(
            f"need at least {min_size} samples to fit; got {x.size}."
            + (" (after dropping non-finite pairs)" if nan_policy == "omit" else "")
        )
    if nan_policy == "raise":
        if not np.all(np.isfinite(x)):
            raise ValueError("data_x contains non-finite values (NaN/inf).")
        if not np.all(np.isfinite(y)):
            raise ValueError("data_y contains non-finite values (NaN/inf).")
    return x, y


def _validate_p0(p0, params: list) -> np.ndarray:
    """Coerce an initial guess to a float vector and length-check it against the
    parameter list.

    ``None`` yields all-ones. A wrong-length ``p0`` raises :class:`ValueError`
    naming the expected count *and order* -- parameters are laid out sorted by
    name (:func:`model_params`), which surprises callers passing a positional
    ``p0`` in source order, so the message spells the order out.
    """
    n = len(params)
    if p0 is None:
        return np.ones(n)
    guess = np.array(p0, dtype=float).reshape(-1)  # copy: callers mutate it
    if guess.size != n:
        names = [str(p) for p in params]
        raise ValueError(
            f"p0 must have length {n} (one per parameter, in order {names}); "
            f"got length {guess.size}."
        )
    return guess


# numeric statistics
def _covariance(
    jac: np.ndarray, res: np.ndarray, n_params: int
) -> np.ndarray | None:
    """Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` from the residual
    Jacobian, scaled by the reduced chi-square.

    Computed from the SVD of ``J`` directly rather than inverting ``J^T J``:
    forming ``J^T J`` squares the condition number, so ``inv(J^T J)`` is both
    slower and far less accurate exactly when the parameters are near-degenerate
    -- the case a covariance is most needed. With ``J = U S V^T``,
    ``(J^T J)^-1 = V diag(1/s^2) V^T``; singular values below a relative
    tolerance are treated as null directions (Moore-Penrose), matching
    ``scipy.optimize.curve_fit``'s SVD-based covariance.

    Returns ``None`` for an exactly- or under-determined system (``m <=
    n_params``) or when ``J`` is entirely singular.
    """
    m = res.size
    if m <= n_params:
        return None
    try:
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or s[0] == 0.0:
        return None
    tol = s[0] * max(jac.shape) * float(np.finfo(float).eps)
    inv_s2 = np.where(s > tol, 1.0 / (s * s), 0.0)
    jtj_inv = (vt.T * inv_s2) @ vt
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
