"""Adaptation #3 -- overlapping-window ensemble + robust aggregation.

Fitting one model to the whole record gives one estimate with no spread and full
exposure to outliers. Instead, fit the model on many **overlapping subwindows**
and aggregate the per-window coefficients robustly: the **median** of the
coefficients (median-of-estimates) rejects windows corrupted by outliers, and
the inter-window spread is a cheap empirical uncertainty band. This is bagging
over the time axis, applicable to both EDA and LSI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dtfit.methods import fit_lsi, fit_eda
from dtfit.types import InitialGuess, FittingResult

_FITTERS: dict[str, Callable[..., FittingResult]] = {"lsi": fit_lsi, "eda": fit_eda}


@dataclass
class EnsembleResult:
    """Aggregated estimate from an overlapping-window ensemble."""

    coeffs: np.ndarray         # aggregated parameter estimate
    spread: np.ndarray         # per-parameter inter-window std (uncertainty)
    members: np.ndarray        # (n_windows_fitted, n_params) raw per-window fits
    expr: str
    var: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        import sympy as sp
        from dtfit.methods._common import model_params

        t = sp.Symbol(self.var)
        f_sym = sp.sympify(self.expr)
        params = model_params(f_sym, t)
        model = sp.lambdify(t, f_sym.subs(dict(zip(params, self.coeffs))), "numpy")
        v = model(np.asarray(x, dtype=float))
        return np.full(np.shape(x), float(v)) if np.ndim(v) == 0 else np.asarray(v, float)


def ensemble_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    method: str = "eda",
    n_windows: int = 8,
    overlap: float = 0.5,
    aggregate: str = "median",
    p0: InitialGuess = None,
    **kwargs,
) -> EnsembleResult:
    """Robustly aggregate fits over overlapping subwindows.

    Args:
        data_x, data_y: Observed samples.
        expr, var: Model expression and main variable.
        method: ``"eda"`` (default) or ``"lsi"``.
        n_windows: Number of overlapping subwindows.
        overlap: Fractional overlap between consecutive windows (0..0.9).
        aggregate: ``"median"`` (robust, default) or ``"wmean"`` (mean).
        p0: Initial guess forwarded to each window fit.
        **kwargs: Extra args forwarded to the underlying fitter.

    Returns:
        EnsembleResult with the aggregated coefficients, their inter-window
        spread (uncertainty), and the raw per-window members.
    """
    fitter = _FITTERS.get(method)
    if fitter is None:
        raise ValueError(f"method must be 'lsi' or 'eda', got {method!r}")
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)
    n = x.size

    step = max(1, int(n / n_windows * (1.0 - overlap)))
    win = max(int(n / n_windows / (1.0 - overlap)) if overlap < 1 else n, 8)
    win = min(win, n)

    members: list[np.ndarray] = []
    start = 0
    while start + win <= n and len(members) < n_windows * 3:
        sl = slice(start, start + win)
        try:
            res = fitter(x[sl], y[sl], expr, var, p0=p0, **kwargs)
            members.append(np.asarray(res.coeffs, dtype=float))
        except Exception:
            pass
        start += step
        if step == 0:
            break

    if not members:
        # fall back to a single whole-record fit
        res = fitter(x, y, expr, var, p0=p0, **kwargs)
        members.append(np.asarray(res.coeffs, dtype=float))

    M = np.vstack(members)
    if aggregate == "median":
        coeffs = np.median(M, axis=0)
    else:
        coeffs = np.mean(M, axis=0)
    spread = np.std(M, axis=0)
    return EnsembleResult(coeffs=coeffs, spread=spread, members=M, expr=expr, var=var)
