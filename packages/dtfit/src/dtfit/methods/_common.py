"""Shared internals of the differential-transformation fitting methods.

Collected here so the method modules (``lsi`` / ``eac`` / ``dsb``) draw on one
small toolbox rather than a scatter of one-function files:

- symbolic spectrum helpers: :func:`model_params`, :func:`taylor_coeffs`;
- user-input normalizers shared by the batch fitters: :func:`normalize_p0`,
  :func:`normalize_bounds`;
- numeric statistics: :func:`_covariance`, :func:`information_criteria`;
- polynomial-degree selection (the DSB pre-fit support): :func:`find_degree`.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

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
            f"data_x and data_y must be 1-D; got shapes {x.shape} and {y.shape}. "
            "dtfit's integral criteria (LSI project onto Legendre polynomials over "
            "a scalar interval; EAC integrates windows on a scalar axis) are "
            "one-dimensional, so multivariate X (several predictors) is not "
            "supported. If instead you have a 1-D signal that is a sum of "
            "components along one axis (e.g. trend + cycle), compose 1-D models "
            "with `+` (models.linear() + models.sine()); see the 'Multivariate "
            "data' note in the docs."
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


def normalize_p0(
    p0: Sequence[float] | np.ndarray | Mapping[str, float] | None,
    param_names: Sequence[str],
) -> np.ndarray | None:
    """Normalize a user initial guess to a float vector in sorted-name order.

    Accepted forms:

    - ``None`` -- no guess supplied; returned unchanged so callers can still
      distinguish "seeded" from "unseeded" paths;
    - a positional sequence -- one value per parameter, laid out in the
      alphabetically-sorted parameter-name order (:func:`model_params`),
      length-checked like :func:`_validate_p0`;
    - a ``{name: value}`` mapping -- must cover **all** parameters; a missing
      or unknown name raises :class:`ValueError` listing the valid names in
      sorted order.

    Returns a fresh float array (callers may mutate it) or ``None``.
    """
    names = [str(n) for n in param_names]
    if p0 is None:
        return None
    if isinstance(p0, Mapping):
        keys = {str(k) for k in p0}
        missing = sorted(set(names) - keys)
        unknown = sorted(keys - set(names))
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"missing {missing}")
            if unknown:
                problems.append(f"unknown {unknown}")
            raise ValueError(
                f"p0 dict must give one value per parameter "
                f"(valid names, in order: {sorted(names)}): "
                + "; ".join(problems) + "."
            )
        return np.array([float(p0[n]) for n in names], dtype=float)
    return _validate_p0(p0, list(names))


def _is_pair(v: Any) -> bool:
    """True for a non-string 2-sequence (a candidate ``(lo, hi)`` pair)."""
    if isinstance(v, str):
        return False
    try:
        return len(v) == 2
    except TypeError:
        return False


def normalize_bounds(
    bounds: (
        Sequence[tuple[float, float]]
        | Mapping[str, tuple[float, float]]
        | tuple[Any, Any]
        | None
    ),
    param_names: Sequence[str],
) -> list[tuple[float, float]] | None:
    """Normalize user bounds to a per-parameter ``[(lo, hi), ...]`` list.

    Accepted forms (``n`` = number of parameters, laid out in the
    alphabetically-sorted name order of :func:`model_params`):

    - ``None`` -- unbounded; returned unchanged;
    - a ``{name: (lo, hi)}`` mapping -- may be **partial**: parameters not
      named get ``(-inf, inf)``; an unknown name raises :class:`ValueError`
      listing the valid names in sorted order;
    - a sequence of ``n`` ``(lo, hi)`` pairs in sorted-name order;
    - the scipy-style 2-tuple ``(lo, hi)`` with ``lo``/``hi`` scalars or
      length-``n`` arrays (:func:`scipy.optimize.least_squares`'s convention).

    Ambiguity note: for ``n == 2`` a 2-tuple of two 2-sequences (e.g.
    ``([0, 0], [10, 10])``) could be read either way; it is interpreted as
    **per-parameter pairs** -- pass scalars or a dict for the scipy reading.

    Each pair is validated ``lo < hi`` (strictly); a violation raises
    :class:`ValueError` naming the offending parameter. To pin a parameter to
    a constant, substitute the value into the model expression instead of
    passing a degenerate ``lo == hi`` box (scipy's bounded solvers reject it).
    """
    names = [str(n) for n in param_names]
    n = len(names)
    if bounds is None:
        return None
    if isinstance(bounds, Mapping):
        keys = {str(k) for k in bounds}
        unknown = sorted(keys - set(names))
        if unknown:
            raise ValueError(
                f"bounds dict names unknown parameters {unknown} "
                f"(valid names, in order: {sorted(names)})."
            )
        out = [
            (float(bounds[nm][0]), float(bounds[nm][1]))
            if nm in bounds else (-np.inf, np.inf)
            for nm in names
        ]
        return _check_bounds(out, names)
    seq = list(bounds)
    if len(seq) == n and all(_is_pair(v) for v in seq):
        # n (lo, hi) pairs in sorted-name order. This branch also resolves the
        # documented n == 2 ambiguity in favour of per-parameter pairs.
        out = [(float(v[0]), float(v[1])) for v in seq]
        return _check_bounds(out, names)
    if len(seq) == 2:
        # scipy-style (lo, hi): scalars broadcast, arrays must be length n.
        lo = np.asarray(seq[0], dtype=float)
        hi = np.asarray(seq[1], dtype=float)
        lo = np.full(n, float(lo)) if lo.ndim == 0 else lo.reshape(-1)
        hi = np.full(n, float(hi)) if hi.ndim == 0 else hi.reshape(-1)
        if lo.size != n or hi.size != n:
            raise ValueError(
                f"scipy-style bounds must give scalars or length-{n} arrays "
                f"(one per parameter, in order {names}); got lengths "
                f"{lo.size} and {hi.size}."
            )
        out = list(zip(lo.tolist(), hi.tolist()))
        return _check_bounds(out, names)
    raise ValueError(
        f"bounds must be a dict, {n} (lo, hi) pairs (one per parameter, in "
        f"order {names}), or a scipy-style (lo, hi) 2-tuple; got a "
        f"length-{len(seq)} sequence."
    )


def _check_bounds(
    out: list[tuple[float, float]], names: list[str]
) -> list[tuple[float, float]]:
    """Validate ``lo < hi`` (strictly) per parameter, naming the offender.

    Strict, not ``<=``: scipy's trf solver rejects a degenerate ``lo == hi``
    box with an error that names no parameter, and whether such a box even
    reaches trf depends on the solver path taken -- validate it consistently
    up front instead.
    """
    for nm, (lo, hi) in zip(names, out):
        if not lo < hi:
            raise ValueError(
                f"invalid bounds for parameter {nm!r}: lower {lo} must be "
                f"strictly less than upper {hi}. To pin a parameter to a "
                "constant, substitute the value into the model expression."
            )
    return out


# per-sample measurement weights
def _resolve_sigma(
    sigma: np.ndarray | Sequence[float] | None,
    data_x: np.ndarray,
    data_y: np.ndarray,
    x: np.ndarray,
    nan_policy: str,
) -> np.ndarray | None:
    """Validate ``sigma`` and align it with the post-validation samples ``x``.

    ``sigma`` is the per-sample measurement standard deviation of ``data_y``,
    the **same length as the raw input** (matching :func:`scipy.optimize.curve_fit`).
    Returns a float array aligned with the validated ``x`` -- with the same
    non-finite ``(x, y)`` pairs dropped when ``nan_policy="omit"`` -- or ``None``
    when no ``sigma`` was supplied. Raises :class:`ValueError` on a length
    mismatch or a non-finite / non-positive entry among the retained samples.

    Shared by :func:`dtfit.fit_lsi` and :func:`dtfit.fit_eac` so the two fitters
    cannot drift on the sigma-length contract (they once disagreed under
    ``nan_policy="omit"``).
    """
    if sigma is None:
        return None
    # ``_validate_xy`` has already run, so the raw inputs are 1-D and equal
    # length; ``reshape(-1)`` is an identity here and just fixes the dtype.
    raw_x = np.asarray(data_x, dtype=float).reshape(-1)
    raw_y = np.asarray(data_y, dtype=float).reshape(-1)
    s = np.asarray(sigma, dtype=float).reshape(-1)
    if s.size != raw_y.size:
        raise ValueError(
            f"sigma must have the same length as data_y; got {s.size} and "
            f"{raw_y.size}."
        )
    if nan_policy == "omit":
        s = s[np.isfinite(raw_x) & np.isfinite(raw_y)]
    if s.size != x.size:  # defensive: cannot happen once the mask matches
        raise ValueError(
            f"sigma length {s.size} does not match the {x.size} fitted samples."
        )
    if not np.all(np.isfinite(s)):
        raise ValueError("sigma contains non-finite values (NaN/inf).")
    if np.any(s <= 0.0):
        raise ValueError("sigma must be strictly positive.")
    return s


# numeric statistics
def _covariance(
    jac: np.ndarray, res: np.ndarray, n_params: int, *, absolute_sigma: bool = False
) -> np.ndarray | None:
    """Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` from the residual Jacobian.

    Computed from the SVD of ``J`` directly rather than inverting ``J^T J``:
    forming ``J^T J`` squares the condition number, so ``inv(J^T J)`` is both
    slower and far less accurate exactly when the parameters are near-degenerate
    -- the case a covariance is most needed. With ``J = U S V^T``,
    ``(J^T J)^-1 = V diag(1/s^2) V^T``; singular values below a relative
    tolerance are treated as null directions (Moore-Penrose), matching
    ``scipy.optimize.curve_fit``'s SVD-based covariance.

    With ``absolute_sigma=False`` (the default) the raw ``(J^T J)^-1`` is scaled
    by the reduced chi-square ``res @ res / (m - n_params)`` -- the residual
    carries the (unknown) noise scale. With ``absolute_sigma=True`` no such
    rescaling is applied (``sigma^2 = 1``): the residual is assumed already
    scaled by ``1/sigma`` (absolute measurement errors), so the covariance
    reflects those absolute values -- matching ``scipy.optimize.curve_fit``'s
    ``absolute_sigma`` flag.

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
    sigma2 = 1.0 if absolute_sigma else float(res @ res) / (m - n_params)
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
