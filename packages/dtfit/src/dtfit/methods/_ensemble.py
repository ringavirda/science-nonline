"""Overlapping-window ensemble -- robust aggregation for outlier-prone data.

Promoted from the experimental adaptations (#3). Fitting one model to the whole
record gives a single estimate with full exposure to outliers. ``ensemble_fit``
instead fits the model on many **overlapping subwindows** and aggregates the
per-window coefficients robustly: the **median** of the estimates rejects windows
corrupted by outliers, and the inter-window spread is a cheap empirical
uncertainty band. This is bagging over the time axis, applicable to both EAC and
LSI.

When to use it: **densely outlier-contaminated** data. The median-of-windows
aggregation rejects whole corrupted windows without the per-problem ``f_scale``
tuning that ``fit_eac(loss="soft_l1", ...)`` needs -- and stays stable where that
robust loss can diverge. For lighter contamination the robust loss on a single
``fit_eac`` is the cheaper path (see :func:`dtfit.fit_eac`); the ensemble is the
heavier-duty complement. On clean (Gaussian-noise) data prefer a single
whole-record fit: the ensemble trades a little accuracy there for the outlier
robustness, so it is a specialised tool rather than the default path.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dtfit.types import FittingResult, InitialGuess
from ._lsi import fit_lsi
from ._eac import fit_eac

_FITTERS: dict[str, Callable[..., FittingResult]] = {"lsi": fit_lsi, "eac": fit_eac}


class EnsembleResult(FittingResult):
    """A :class:`FittingResult` aggregated from an overlapping-window ensemble.

    Behaves like any fitted result (named ``params``, ``model``, ``predict``,
    ``to_dict``) and additionally exposes the raw per-window fits
    (:attr:`members`) and their inter-window standard deviation (:attr:`spread`).
    The spread also fills a diagonal empirical covariance, so ``stderr()`` and
    ``predict(return_std=True)`` report the ensemble's uncertainty.

    Attributes:
        spread: Per-parameter inter-window standard deviation (uncertainty).
        members: ``(n_windows_fitted, n_params)`` raw per-window coefficients.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        spread: np.ndarray,
        members: np.ndarray,
        *,
        expr: str,
        var: str,
        names: tuple[str, ...],
    ) -> None:
        spread = np.asarray(spread, dtype=float)
        super().__init__(
            coeffs=coeffs, cov=np.diag(spread**2), expr=expr, var=var, names=names
        )
        self.spread = spread
        self.members = np.asarray(members, dtype=float)


def ensemble_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    method: str = "eac",
    n_windows: int = 8,
    overlap: float = 0.5,
    aggregate: str = "median",
    p0: InitialGuess = None,
    **kwargs,
) -> EnsembleResult:
    """Robustly aggregate fits over overlapping subwindows (bagging in time).

    Args:
        data_x, data_y: Observed samples.
        expr, var: Model expression and main variable.
        method: Underlying batch fitter, ``"eac"`` (default) or ``"lsi"``.
        n_windows: Target number of overlapping subwindows.
        overlap: Fractional overlap between consecutive windows (``0..0.9``).
        aggregate: ``"median"`` (robust, default) or ``"mean"``.
        p0: Initial guess forwarded to each window fit.
        **kwargs: Extra arguments forwarded to the underlying fitter (e.g.
            ``bounds``).

    Returns:
        :class:`EnsembleResult` -- a :class:`FittingResult` carrying the
        aggregated coefficients plus the per-window ``members`` and their
        ``spread`` (which also populates the covariance).
    """
    fitter = _FITTERS.get(method)
    if fitter is None:
        raise ValueError(f"method must be 'lsi' or 'eac', got {method!r}")
    if aggregate not in ("median", "mean"):
        raise ValueError(f"aggregate must be 'median' or 'mean', got {aggregate!r}")
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)
    n = x.size

    step = max(1, int(n / n_windows * (1.0 - overlap)))
    win = max(int(n / n_windows / (1.0 - overlap)) if overlap < 1 else n, 8)
    win = min(win, n)

    members: list[np.ndarray] = []
    names: tuple[str, ...] = ()
    last_res: FittingResult | None = None
    start = 0
    while start + win <= n and len(members) < n_windows * 3:
        sl = slice(start, start + win)
        try:
            res = fitter(x[sl], y[sl], expr, var, p0=p0, **kwargs)
            members.append(np.asarray(res.coeffs, dtype=float))
            last_res = res
            names = res.names or names
        except Exception:  # noqa: BLE001 - a corrupted window is simply skipped
            pass
        start += step
        if step == 0:
            break

    if not members:  # every subwindow failed -> one whole-record fit
        res = fitter(x, y, expr, var, p0=p0, **kwargs)
        members.append(np.asarray(res.coeffs, dtype=float))
        last_res = res
        names = res.names or names

    M = np.vstack(members)
    coeffs = np.median(M, axis=0) if aggregate == "median" else np.mean(M, axis=0)
    if M.shape[0] < 2:
        # A single surviving window gives no inter-window spread -- np.std would
        # be exactly 0, which masquerades as *zero* uncertainty (a dangerously
        # overconfident stderr / predict(return_std=True) precisely when the
        # ensemble has degraded to one fit). Fall back to that fit's own analytic
        # covariance; if even that is unavailable, report NaN rather than lie.
        cov = last_res.cov if last_res is not None else None
        if cov is not None:
            spread = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        else:
            spread = np.full(M.shape[1], np.nan)
    else:
        spread = np.std(M, axis=0)
    return EnsembleResult(coeffs, spread, M, expr=expr, var=var, names=names)
