"""GEMM-batched, backend-pluggable LSI projection (promoted from the experiment suite).

The data side of LSI is an integral ``β_j = ∫ y·φ_j dx`` which factors as a
matrix product ``β = Dᵀ·(w⊙y)`` (design matrix ``D`` and trapezoid weights ``w``;
see :func:`dtfit._core._spectral._trapz_weights`). Stacking ``B`` channels
that share a sampling grid into the columns of ``Y`` turns the whole batch into a
single GEMM ``S = Dᵀ·(w⊙Y)``:

* on CPU it dispatches to multithreaded BLAS instead of a Python per-channel loop;
* on a GPU backend (``cupy`` / ``torch``) it runs on cuBLAS, where the projection
  amortizes the kernel launch / transfer over all ``B`` channels.

This is the form that makes the projection scale: it raises the arithmetic work
done per byte read (``B`` outputs per input column), which is exactly what a
bandwidth-bound reduction needs to benefit from a GPU.

The reduction stays *exact and additive* (it is the same projection as
:class:`dtfit.PartitionedLSI`), so a batched projection can still be summed across
a domain partition. Validated across the big-data domain study (~50x the
per-channel loop on the 321-channel real panel, bit-identical) and **promoted to
the stable API**.
"""

from __future__ import annotations

import numpy as np

from dtfit.types import FittingResult, InitialGuess
from dtfit._core._backend import Backend, resolve_backend
from dtfit._core._spectral import make_basis, solve_spectral


def project_spectra(
    x: np.ndarray,
    Y: np.ndarray,
    *,
    order: int = 6,
    basis: str = "legendre",
    backend: str | Backend = "auto",
    **basis_kwargs: object,
) -> np.ndarray:
    """Empirical spectra of ``B`` channels sharing grid ``x``, in one GEMM.

    ``Y`` is ``(n, B)`` (a column per channel) or ``(n,)`` for one channel;
    returns ``(B, n_coef)`` (or ``(n_coef,)`` for a single channel). ``backend``
    is a name (``"auto"``/``"numpy"``/``"cupy"``/``"torch"``) or a :class:`Backend`.
    """
    x = np.asarray(x, float)
    Y = np.asarray(Y)  # preserve dtype; the backend controls compute precision
    single = Y.ndim == 1
    if single:
        Y = Y[:, None]
    if Y.shape[0] != x.shape[0]:
        raise ValueError(
            f"Y must have shape (len(x), n_channels); got {Y.shape} for len(x)={x.size}"
        )
    b = make_basis(basis, order, (float(x[0]), float(x[-1])), **basis_kwargs)
    bk = backend if isinstance(backend, Backend) else resolve_backend(backend)
    spectra = b.empirical_batched(x, Y, bk)
    return spectra[0] if single else spectra


def fit_lsi_batched(
    x: np.ndarray,
    Y: np.ndarray,
    expr: str,
    var: str,
    *,
    order: int = 6,
    basis: str = "legendre",
    backend: str | Backend = "auto",
    p0: InitialGuess = None,
    bounds: list[tuple[float, float]] | None = None,
    **basis_kwargs: object,
) -> FittingResult | list[FittingResult]:
    """Fit one LSI model per channel of ``Y`` (shared grid ``x``); projections batched.

    All channels' empirical spectra are computed in a single GEMM via ``backend``;
    each channel's small spectral-match solve then runs on the host (it is
    ``len(params)``-dimensional and negligible). ``Y`` is ``(n, B)`` or ``(n,)``;
    returns a list of :class:`FittingResult` (or one for a single channel).
    """
    x = np.asarray(x, float)
    Y = np.asarray(Y)  # preserve dtype; the backend controls compute precision
    single = Y.ndim == 1
    if single:
        Y = Y[:, None]
    if Y.shape[0] != x.shape[0]:
        raise ValueError(
            f"Y must have shape (len(x), n_channels); got {Y.shape} for len(x)={x.size}"
        )
    b = make_basis(basis, order, (float(x[0]), float(x[-1])), **basis_kwargs)
    bk = backend if isinstance(backend, Backend) else resolve_backend(backend)
    spectra = b.empirical_batched(x, Y, bk)  # (B, n_coef)
    p0a = None if p0 is None else np.asarray(p0, float)
    results = [
        solve_spectral(expr, var, b, spectra[i], p0=p0a, bounds=bounds)
        for i in range(spectra.shape[0])
    ]
    return results[0] if single else results
