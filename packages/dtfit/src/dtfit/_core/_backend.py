"""Pluggable array backend for the GEMM-batched projection.

The batched LSI/EDA projection is a single matrix product ``Dᵀ·(w⊙Y)`` (see
:mod:`dtfit.scale._batched`). That primitive runs unchanged on NumPy/BLAS
(CPU) or on a GPU array library, so we keep the math in one place and swap only
*where the arrays live*:

* ``numpy``  -- always available; multithreaded BLAS GEMM on the CPU.
* ``cupy``   -- drop-in NumPy-API GPU arrays (cuBLAS), if installed + a GPU.
* ``torch``  -- CUDA tensors, if installed + ``torch.cuda.is_available()``.

A :class:`Backend` only has to move an array to/from its device; the projection
code uses plain ``@``, ``*`` and ``.T``, which every backend supports. This is
why GPU support is a *backend choice*, not a rewrite -- the projection is a GEMM
with very low arithmetic intensity, so the GPU pays off only when the data is
already resident (see the throughput experiment), but the code path is identical.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class Backend:
    """Moves arrays on/off a compute device; arithmetic stays generic (``@``/``*``)."""

    def __init__(
        self,
        name: str,
        asarray: Callable[[Any], Any],
        to_host: Callable[[Any], np.ndarray],
    ) -> None:
        self.name = name
        self._asarray = asarray
        self._to_host = to_host

    def asarray(self, a: Any) -> Any:
        """Place ``a`` on this backend's device with the backend dtype."""
        return self._asarray(a)

    def to_host(self, a: Any) -> np.ndarray:
        """Bring a device array back to a host NumPy array."""
        return self._to_host(a)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Backend({self.name!r})"


def _numpy_backend(dtype: Any) -> Backend:
    npd = np.dtype(dtype)
    return Backend("numpy", lambda a: np.asarray(a, dtype=npd), lambda a: np.asarray(a))


def _cupy_backend(dtype: Any) -> Backend:  # pragma: no cover - requires a GPU
    import cupy as cp

    cpd = cp.dtype(dtype)
    return Backend("cupy", lambda a: cp.asarray(a, dtype=cpd), lambda a: cp.asnumpy(a))


def _torch_backend(dtype: Any) -> Backend:  # pragma: no cover - requires a GPU
    import torch

    td = torch.float32 if np.dtype(dtype) == np.float32 else torch.float64
    return Backend(
        "torch",
        lambda a: torch.as_tensor(np.asarray(a), dtype=td, device="cuda"),
        lambda a: a.detach().cpu().numpy(),
    )


def available_backends() -> list[str]:
    """Backends usable here: always ``numpy``, plus GPU ones that import + have a device."""
    out = ["numpy"]
    try:  # pragma: no cover - depends on environment
        import cupy  # noqa: F401

        out.append("cupy")
    except Exception:
        pass
    try:  # pragma: no cover - depends on environment
        import torch

        if torch.cuda.is_available():
            out.append("torch")
    except Exception:
        pass
    return out


def resolve_backend(name: str = "auto", *, dtype: Any = "float64") -> Backend:
    """Build a :class:`Backend` by name; ``"auto"`` prefers a GPU when present."""
    avail = available_backends()
    if name == "auto":
        name = "cupy" if "cupy" in avail else ("torch" if "torch" in avail else "numpy")
    if name == "numpy":
        return _numpy_backend(dtype)
    if name == "cupy":
        return _cupy_backend(dtype)  # pragma: no cover - requires a GPU
    if name == "torch":
        return _torch_backend(dtype)  # pragma: no cover - requires a GPU
    raise ValueError(f"unknown backend {name!r}; available: {avail}")
