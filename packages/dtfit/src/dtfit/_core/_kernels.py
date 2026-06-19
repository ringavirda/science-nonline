"""Numeric kernels with an optional compiled (C) backend.

The integral-based fitting methods repeat a few small numeric inner loops --
composite-Simpson window integrals and Gauss-Legendre spectral projections --
thousands of times inside the optimizers and the streaming filters. Those loops
carry heavy per-call Python / SciPy overhead.

This module exposes them behind a thin API. When the compiled extension
``dtfit._native`` (built by ``build_native.py`` with clang) is importable, the
calls dispatch to it; otherwise they fall back to pure NumPy / SciPy. The two
paths are numerically identical -- the C Simpson reproduces
``scipy.integrate.simpson`` exactly -- so results do not depend on whether the
extension is built. ``HAVE_NATIVE`` reports which backend is active.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

try:  # compiled backend is optional
    from dtfit._core import _native

    HAVE_NATIVE = True
except Exception:  # pragma: no cover - exercised only without the build
    _native = None  # type: ignore[assignment]
    HAVE_NATIVE = False


def _as_idx(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.intp)


def simpson_windows(
    y: np.ndarray, x: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """Composite-Simpson integral of ``y(x)`` over each index window.

    Window ``k`` spans the half-open index range ``[starts[k], stops[k])`` of the
    shared samples ``(x, y)``. Returns an array of ``len(starts)`` areas.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    starts = _as_idx(starts)
    stops = _as_idx(stops)
    if HAVE_NATIVE and _native is not None:
        return _native.simpson_windows(y, x, starts, stops)
    return np.array(
        [simpson(y=y[s:e], x=x[s:e]) for s, e in zip(starts, stops)],
        dtype=np.float64,
    )


def simpson_windows_rows(
    Y: np.ndarray, x: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """Per-row composite-Simpson integral of a 2-D ``Y`` over the windows.

    ``Y`` has shape ``(nrows, len(x))``; returns ``(nrows, len(starts))`` areas.
    Used to integrate a stack of parameter sensitivities (Jacobian rows) at once.
    """
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    starts = _as_idx(starts)
    stops = _as_idx(stops)
    if HAVE_NATIVE and _native is not None:
        return _native.simpson_windows_rows(Y, x, starts, stops)
    return np.array(
        [
            [simpson(y=row[s:e], x=x[s:e]) for s, e in zip(starts, stops)]
            for row in Y
        ],
        dtype=np.float64,
    )


def legendre_project(
    fv: np.ndarray, qw: np.ndarray, legvander: np.ndarray, norm: np.ndarray
) -> np.ndarray:
    """Fused Gauss-Legendre spectral projection.

    Computes ``norm * ((qw * fv) @ legvander)`` -- the order-``k`` Legendre
    coefficients of a function sampled at the quadrature nodes. ``legvander`` is
    ``(nq, k)``; ``fv`` and ``qw`` are ``(nq,)``; ``norm`` is ``(k,)``.
    """
    fv = np.ascontiguousarray(fv, dtype=np.float64)
    qw = np.ascontiguousarray(qw, dtype=np.float64)
    legvander = np.ascontiguousarray(legvander, dtype=np.float64)
    norm = np.ascontiguousarray(norm, dtype=np.float64)
    if HAVE_NATIVE and _native is not None:
        return _native.legendre_project(fv, qw, legvander, norm)
    return norm * ((qw * fv) @ legvander)
