"""Backend infrastructure for the GEMM-batched projection throughput experiment.

This module is the **single source of truth for the benchmarking and compute
code** behind ``08_gpu_batched_projection.ipynb``; the notebook imports it and
does all the presentation (tables, figures, narrative). Keeping the infra here
means the data generator, timers and benchmark kernels are defined once and the
notebook stays a thin, rerunnable layer over them.

The data side of LSI/EAC is an integral ``beta_j = integral y*phi_j dx`` that
factors into a single matrix product ``beta = D^T*(w (x) y)`` (design matrix
``D``, trapezoid weights ``w``). Stacking ``B`` channels that share a grid into
the columns of ``Y`` makes the whole batch one GEMM ``S = D^T*(w (x) Y)`` -- the
recommendation-#2 reframe. This module measures what that buys:

* the **data generator** -- :func:`make_data` (a bank of ``exp(b*t)`` channels);
* the **roofline anchor** -- :func:`host_copy_bandwidth` (host memory copy GB/s);
* the **CPU win** -- :func:`bench_loop_vs_batched` (one BLAS GEMM vs a per-channel
  Python loop, with an exactness check);
* the **precision sweep** -- :func:`bench_dtype` (fp32 vs fp64, bandwidth-bound);
* the **honest GPU story** -- :func:`bench_resident_streamed` (every available
  backend, timed resident vs streamed) plus :func:`gpu_name` for the device.

The GPU rows are driven entirely by :func:`dtfit_experimental.available_backends`:
on a CUDA box (``cupy`` / ``torch`` present and a device reachable) the
resident/streamed columns are filled from measurement; with no usable GPU only
the ``numpy`` (CPU) backend appears and the notebook reports the CPU numbers plus
the PCIe-bound model -- so the notebook completes either way.
"""

from __future__ import annotations

import time

import numpy as np

from dtfit.scale import project_spectra
from dtfit._core._spectral import make_basis
from dtfit_experimental import available_backends, resolve_backend

from dtfit_experimental.experiments.common import fmt

__all__ = [
    "ORDER",
    "gpu_name",
    "make_data",
    "host_copy_bandwidth",
    "bench_loop_vs_batched",
    "bench_dtype",
    "bench_resident_streamed",
    "has_gpu_backend",
    "available_backends",
    "fmt",
]

ORDER = 6  # k = order + 1 = 7 coefficients


def gpu_name() -> str:
    """Best-effort device name of the active GPU backend (for the report)."""
    try:  # pragma: no cover - only on a CUDA box
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # pragma: no cover
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:
            return "the GPU"


def has_gpu_backend() -> bool:
    """True iff a non-``numpy`` (GPU) backend is present and usable.

    Tries every non-numpy backend with a tiny GEMM; if none resolves and runs we
    are CPU-only and the GPU rows degrade to "n/a" in the notebook.
    """
    for name in available_backends():
        if name == "numpy":
            continue
        try:  # pragma: no cover - depends on the host
            bk = resolve_backend(name, dtype="float32")
            a = bk.asarray(np.ones((2, 2), dtype="float32"))
            bk.to_host(a.T @ a)
            return True
        except Exception:  # pragma: no cover
            continue
    return False


def _time(fn, reps: int) -> float:
    """Best (min) wall time of ``fn`` over ``reps`` runs."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_data(N: int, B: int, seed: int = 0):
    """A bank of ``B`` channels ``y = exp(b*t)`` on a shared ``N``-point grid."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, N)
    rates = np.linspace(0.1, 0.5, B)
    Y = np.exp(np.outer(x, rates)) + rng.normal(0.0, 0.01, (N, B))
    return x, np.ascontiguousarray(Y)


def host_copy_bandwidth(nbytes: int = 256 << 20, reps: int = 5) -> float:
    """Reference host memory copy bandwidth (GB/s) -- the CPU roofline anchor."""
    a = np.ones(nbytes // 8, dtype=np.float64)
    b = np.empty_like(a)
    t = _time(lambda: np.copyto(b, a), reps)
    return 2.0 * a.nbytes / t / 1e9  # read + write


# --- 1. batched GEMM vs per-channel loop --------------------------------- #
def bench_loop_vs_batched(N, Bs, reps):
    """Per-channel Python loop vs a single batched GEMM through NumPy/BLAS.

    Returns a list of rows ``[B, loop_ms, batched_ms, speedup, Melem/s, GB/s,
    max|delta|]`` (numeric strings via :func:`fmt`). ``max|delta|`` is the largest
    difference between the batched and looped spectra -- ~machine-epsilon proves
    the batched path is exact, not a corner cut.
    """
    rows = []
    for B in Bs:
        x, Y = make_data(N, B)
        b = make_basis("legendre", ORDER, (float(x[0]), float(x[-1])))
        loop = lambda: [b.integral_to_spectrum(b.project_integral(x, Y[:, i]))  # noqa: E731
                        for i in range(B)]
        batch = lambda: project_spectra(x, Y, order=ORDER, backend="numpy")  # noqa: E731
        loop(); batch()  # warm
        t_loop = _time(loop, max(2, reps // 2))  # loop is slow + stable
        t_batch = _time(batch, reps)
        # correctness: batched must equal the loop (no corner cut). Check a
        # capped subset of channels so this stays cheap at large B.
        nchk = min(B, 32)
        sb = np.atleast_2d(np.asarray(batch()))[:nchk]
        sl = np.array([b.integral_to_spectrum(b.project_integral(x, Y[:, i]))
                       for i in range(nchk)])
        max_diff = float(np.max(np.abs(sb - sl)))
        gbps = N * B * 8 / t_batch / 1e9
        rows.append([B, fmt(t_loop * 1e3, "{:.2f}"), fmt(t_batch * 1e3, "{:.2f}"),
                     fmt(t_loop / t_batch, "{:.1f}"), fmt(N * B / t_batch / 1e6, "{:.0f}"),
                     fmt(gbps, "{:.1f}"), f"{max_diff:.1e}"])
    return rows


# --- 2. fp32 vs fp64 ----------------------------------------------------- #
def bench_dtype(N, B, reps):
    """Same batched projection in fp64 vs fp32. Returns rows ``[dtype, time_ms,
    Melem/s, GB/s]``. The kernel is bandwidth-bound, so halving the bytes per
    element should lift throughput roughly proportionally."""
    x, Y0 = make_data(N, B)
    rows = []
    for dt in ["float64", "float32"]:
        Y = Y0.astype(dt)  # data *stored* in the target precision (no in-loop cast)
        bk = resolve_backend("numpy", dtype=dt)
        run = lambda: project_spectra(x, Y, order=ORDER, backend=bk)  # noqa: E731
        run()
        t = _time(run, reps)
        rows.append([dt, fmt(t * 1e3, "{:.2f}"), fmt(N * B / t / 1e6, "{:.0f}"),
                     fmt(N * B * np.dtype(dt).itemsize / t / 1e9, "{:.1f}")])
    return rows


# --- 3. backend resident vs streamed (per dtype) ------------------------- #
def bench_resident_streamed(N, B, reps):
    """The same GEMM per **available** backend, timed two ways: **resident**
    (arrays already on the device -- only the matmul is timed) and **streamed**
    (arrays transferred in every call, as for host/disk-resident data).

    Returns rows ``[backend, dtype, resident_ms, streamed_ms,
    streamed/resident, Melem/s(resident)]``. On a CPU-only host only ``numpy``
    appears (resident == streamed, no transfer); on a CUDA box the GPU backends
    fill in too. Any backend that fails to run is skipped, so the notebook never
    errors on a missing GPU."""
    x, Y0 = make_data(N, B)
    b = make_basis("legendre", ORDER, (float(x[0]), float(x[-1])))
    D, w = b._gemm_factors(x)
    Dw0 = np.ascontiguousarray(w[:, None] * D)  # weights folded into small design
    rows = []
    for name in available_backends():
        try:
            bk = resolve_backend(name, dtype="float64")
        except Exception:  # pragma: no cover - backend not usable on this host
            continue
        for dt in ("float64", "float32"):
            try:
                bk = resolve_backend(name, dtype=dt)
                Dw = Dw0.astype(dt); Y = Y0.astype(dt)

                def streamed(bk=bk, Dw=Dw, Y=Y):  # transfer big Y each call, GEMM, fetch
                    return bk.to_host(bk.asarray(Dw).T @ bk.asarray(Y))

                Dd = bk.asarray(Dw); Yd = bk.asarray(Y)  # resident: transfer once

                def resident(bk=bk, Dd=Dd, Yd=Yd):
                    return bk.to_host(Dd.T @ Yd)

                streamed(); resident()  # warm
                t_s = _time(streamed, reps)
                t_r = _time(resident, reps)
            except Exception:  # pragma: no cover - GPU absent / out of memory
                continue
            rows.append([name, dt, fmt(t_r * 1e3, "{:.2f}"), fmt(t_s * 1e3, "{:.2f}"),
                         fmt(t_s / t_r, "{:.1f}"), fmt(N * B / t_r / 1e6, "{:.0f}")])
    return rows
