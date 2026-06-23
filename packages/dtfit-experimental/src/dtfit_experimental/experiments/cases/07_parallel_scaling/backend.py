"""Backend infrastructure for the parallel-scaling / architecture-adaptability experiment.

This module is the **single source of truth for the benchmark code** behind
``07_parallel_scaling.ipynb``; the notebook imports it and does all the
presentation (tables, figures, narrative). Keeping the infra here means the
workloads, parallel drivers and scaling/speedup measurements are defined once
and the notebook stays a thin, rerunnable layer over them.

The experiment quantifies how dtfit turns a multi-core box into throughput --
the **acceleration factor vs the rank of parallelism** ``P`` -- measured three
ways, each honest about its bottleneck:

* **Compiled-kernel threading** -- :func:`kernel_scaling`. The native numeric
  kernels release the GIL, so many concurrent threads run the compiled
  Simpson/Legendre loops truly in parallel; cache-resident data makes it
  compute-bound and near-linearly scaling -- the clean validation of the
  GIL-release refactor.
* **``fit_many`` process backend** -- :func:`fitmany_scaling`. Embarrassingly
  parallel independent fits via loky; per-task dispatch/spawn overhead and the
  per-fit SymPy lambdify cap the practical speedup of fine-grained fits.
* **Threaded map-reduce streaming** -- :func:`mapreduce_scaling` (``PartitionedLSI``,
  adaptation #1). Numpy releases the GIL on the bulk array ops, so partitions
  run concurrently, but the workload is memory-bandwidth-bound, which caps the
  speedup.

Plus the :func:`amdahl_serial_fraction` fit for the kernel curve and a small
:func:`_timed` helper. Functions submitted to the ``fit_many`` process pool stay
picklable because the loky backend handles them; the work closures in the
threaded benchmarks are local (threads, no pickling needed).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dtfit import FittingProblem, fit_many
from dtfit import PartitionedLSI
from dtfit._core import _kernels

__all__ = [
    "N_CORES", "PHYS", "HAVE_NATIVE",
    "amdahl_serial_fraction", "kernel_scaling", "fitmany_scaling",
    "mapreduce_scaling",
]

N_CORES = os.cpu_count() or 8
PHYS = N_CORES // 2
HAVE_NATIVE = _kernels.HAVE_NATIVE


def amdahl_serial_fraction(Ps, speedups):
    """Fit Amdahl's serial fraction ``s`` to an observed speedup curve, i.e. the
    ``s`` for which ``1 / (s + (1 - s)/P)`` best matches the measured speedups."""
    from scipy.optimize import least_squares
    Ps = np.asarray(Ps, float)
    sp = np.asarray(speedups, float)
    sol = least_squares(lambda s: 1.0 / (s + (1 - s) / Ps) - sp, 0.05,
                        bounds=(0, 1))
    return float(sol.x[0])


# --- 1. compiled-kernel threaded throughput (weak scaling) --------------- #
def kernel_scaling(Ps, rep_per_thread=4000):
    """Weak-scaling throughput of the native Simpson kernel under P threads.

    Each of ``P`` threads runs ``rep_per_thread`` native ``simpson_windows`` calls
    on cache-resident data (compute-bound). Because the kernels release the GIL,
    P threads do P x the work in nearly the same wall time. Returns
    ``({P: throughput-multiplier}, {P: wall-time})``."""
    from dtfit._core import _native
    x = np.ascontiguousarray(np.linspace(0, 10, 40_000))
    y = np.ascontiguousarray(np.sin(x))
    starts = np.arange(0, 39_000, 200, dtype=np.intp)
    stops = starts + 200

    def work(_):
        acc = 0.0
        for _ in range(rep_per_thread):
            acc += _native.simpson_windows(y, x, starts, stops).sum()
        return acc

    def run(P):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=P) as ex:
            list(ex.map(work, range(P)))
        return time.perf_counter() - t0

    run(1)  # warm
    times = {P: run(P) for P in Ps}
    # weak scaling: P threads do P x the calls; throughput multiplier vs 1 thread
    return {P: (P * times[1]) / times[P] for P in Ps}, times


# --- 2. fit_many process strong scaling ---------------------------------- #
def fitmany_scaling(Ps, n_problems):
    """Strong-scaling speedup of ``fit_many`` (EAC) across loky processes.

    Builds ``n_problems`` independent ``a*exp(b*t)`` fits and times them at each
    worker count ``P``. Returns ``({P: speedup vs P=1}, {P: wall-time})``."""
    rng = np.random.default_rng(0)
    probs = []
    for i in range(n_problems):
        x = np.linspace(0, 1.5, 400)
        y = (1 + 0.2 * (i % 5)) * np.exp((0.6 + 0.05 * (i % 7)) * x) + rng.normal(0, 0.03, 400)
        probs.append(FittingProblem(x=x, y=y, expr="a*exp(b*t)", var="t",
                                method="eac", kwargs={"p0": [1.0, 1.0]}))
    fit_many(probs[:16], n_jobs=max(Ps), backend="loky")  # warm pool
    times = {P: _timed(lambda: fit_many(probs, n_jobs=P, backend="loky")) for P in Ps}
    return {P: times[1] / times[P] for P in Ps}, times


# --- 3. threaded map-reduce streaming ------------------------------------ #
def mapreduce_scaling(Ps, total):
    """Strong-scaling speedup of a threaded ``PartitionedLSI`` map-reduce stream.

    A ``total``-sample exponential stream is split into ``P`` partitions, each
    processed by a thread (numpy releases the GIL on the bulk ops). Returns
    ``({P: speedup vs P=1}, {P: wall-time})``."""
    def work(args):
        t0, t1, n, seed = args
        rng = np.random.default_rng(seed)
        acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
        CH = 2_000_000
        nc = max(1, n // CH)
        span = (t1 - t0) / nc
        for c in range(nc):
            x = np.linspace(t0 + c * span, t0 + (c + 1) * span, CH, endpoint=False)
            y = np.exp(0.2 * x) + rng.normal(0, 0.05, CH)
            acc.update(x, y)
        return acc._s

    def run(P):
        parts = [(10.0 * i / P, 10.0 * (i + 1) / P, total // P, i) for i in range(P)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=P) as ex:
            list(ex.map(work, parts))
        return time.perf_counter() - t0

    run(2)  # warm
    times = {P: run(P) for P in Ps}
    return {P: times[1] / times[P] for P in Ps}, times


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
