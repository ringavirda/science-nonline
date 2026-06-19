"""Adaptation #2 -- pluggable orthogonal basis for LSI.

``fit_lsi`` matches spectra on the Legendre basis. For a periodic signal the
Legendre (polynomial) spectrum needs many high orders to express an oscillation,
whereas a **Fourier** basis captures it in a couple of harmonics; for a pure
decay a **Laguerre** basis is natural. :func:`fit_lsi_basis` keeps the exact LSI
criterion (diagonal-weighted spectral match) but lets the caller pick the basis.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from dtfit.types import FittingResult, InitialGuess
from dtfit._core._spectral import make_basis, solve_spectral


def fit_lsi_basis(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    basis: str = "fourier",
    order: int = 5,
    filter_data: bool = True,
    period: float | None = None,
    bounds: list[tuple[float, float]] | None = None,
    p0: InitialGuess = None,
) -> FittingResult:
    """LSI fit with a chosen orthogonal basis.

    Args:
        data_x, data_y: Observed samples.
        expr, var: Model expression and main variable.
        basis: ``"legendre"`` | ``"chebyshev"`` | ``"fourier"`` | ``"laguerre"``.
        order: Spectral order (number of harmonics K for ``"fourier"``).
        filter_data: Savitzky-Golay pre-smoothing before projection.
        period: Fundamental period for ``"fourier"`` (defaults to the domain
            length).
        p0: Optional initial guess.

    Returns:
        FittingResult with coefficients, callable model and covariance.
    """
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)

    if filter_data and y.size >= 5:
        window = min(11, y.size if y.size % 2 == 1 else y.size - 1)
        if window > 3:
            y = np.asarray(savgol_filter(y, window, polyorder=3), dtype=float)

    domain = (float(x[0]), float(x[-1]))
    kwargs = {"period": period} if basis == "fourier" else {}
    b = make_basis(basis, order, domain, **kwargs)
    beta_data = b.empirical(x, y)
    guess = None if p0 is None else np.asarray(p0, float)
    return solve_spectral(expr, var, b, beta_data, p0=guess, bounds=bounds)
