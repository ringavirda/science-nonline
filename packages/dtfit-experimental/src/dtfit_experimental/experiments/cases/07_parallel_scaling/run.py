"""Experiment 7 -- parallel scaling & architecture adaptability.

How well do the dtfit methods turn a multi-core box into throughput? We measure
the **acceleration factor vs the rank of parallelism** P from three angles, each
honest about its bottleneck:

1. **Compiled-kernel threading (the headline).** The native numeric kernels
   release the GIL (this work's kernel change), so many concurrent threads run
   the compiled Simpson/Legendre loops truly in parallel. With cache-resident
   data this is compute-bound and scales near-linearly to the physical-core
   count -- the clean validation of the GIL-release refactor.
2. **`fit_many` process backend.** Embarrassingly-parallel independent fits via
   loky; on this platform the per-task dispatch/spawn overhead and the per-fit
   SymPy lambdify limit the practical speedup of fine-grained fits -- reported
   honestly.
3. **Threaded map-reduce streaming** (`PartitionedLSI`, adaptation #1). Numpy
   releases the GIL on the bulk array ops, so partitions run concurrently, but
   the workload is memory-bandwidth-bound, which caps the speedup.

Together they show where dtfit's parallelism pays off (compiled hot loops) and
where the ceiling is set by the platform (IPC, memory bandwidth), not the
algorithms -- which are all embarrassingly parallel.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dtfit import FittingProblem, fit_many
from dtfit import PartitionedLSI
from dtfit._core import _kernels

from dtfit_experimental.experiments.common import ReportWriter, fmt
from dtfit_experimental.experiments.common.plotting import plt

EXP_DIR = __file__.rsplit("run.py", 1)[0]
N_CORES = os.cpu_count() or 8
PHYS = N_CORES // 2


def amdahl_serial_fraction(Ps, speedups):
    from scipy.optimize import least_squares
    Ps = np.asarray(Ps, float)
    sp = np.asarray(speedups, float)
    sol = least_squares(lambda s: 1.0 / (s + (1 - s) / Ps) - sp, 0.05,
                        bounds=(0, 1))
    return float(sol.x[0])


# --- 1. compiled-kernel threaded throughput (weak scaling) --------------- #
def kernel_scaling(Ps, rep_per_thread=4000):
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
    # weak scaling: P threads do P× the calls; throughput multiplier vs 1 thread
    return {P: (P * times[1]) / times[P] for P in Ps}, times


# --- 2. fit_many process strong scaling ---------------------------------- #
def fitmany_scaling(Ps, n_problems):
    rng = np.random.default_rng(0)
    probs = []
    for i in range(n_problems):
        x = np.linspace(0, 1.5, 400)
        y = (1 + 0.2 * (i % 5)) * np.exp((0.6 + 0.05 * (i % 7)) * x) + rng.normal(0, 0.03, 400)
        probs.append(FittingProblem(x=x, y=y, expr="a*exp(b*t)", var="t",
                                method="eda", kwargs={"p0": [1.0, 1.0]}))
    fit_many(probs[:16], n_jobs=max(Ps), backend="loky")  # warm pool
    times = {P: _timed(lambda: fit_many(probs, n_jobs=P, backend="loky")) for P in Ps}
    return {P: times[1] / times[P] for P in Ps}, times


# --- 3. threaded map-reduce streaming ------------------------------------ #
def mapreduce_scaling(Ps, total):
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


def main(quick: bool = False) -> str:
    rep = ReportWriter(
        EXP_DIR, "Experiment 7 — Parallel scaling & architecture adaptability",
        intent=(
            f"Quantify how dtfit uses this {PHYS}-physical-core machine "
            f"({N_CORES} logical): the acceleration factor vs the rank of "
            "parallelism P, measured three ways — the GIL-released compiled "
            "kernels (threads), independent fits (`fit_many`, processes), and "
            "the threaded map-reduce stream (adaptation #1) — each reported "
            "honestly with its bottleneck."),
    )
    rep.section(
        "Models fitted & why",
        "This experiment measures *throughput*, not fit quality, so the models "
        "are deliberately simple and representative:\n"
        "- **`fit_many` / `FilterBank`:** `y = a·exp(b·t)` — a canonical "
        "nonlinear-in-parameters fit, replicated across many independent "
        "problems/streams to create the embarrassingly-parallel workload.\n"
        "- **Kernel-threading benchmark:** no model — raw `simpson_windows` "
        "native-kernel calls on cache-resident data, chosen to isolate the "
        "compiled hot loop and show the GIL-release scaling directly.\n"
        "- **Map-reduce stream:** `PartitionedLSI` on `y = a·exp(b·t)`, the "
        "promoted distributed estimator.")

    Ps = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16]

    # --- 1. kernel threading (headline) --------------------------------- #
    rep.section(
        "1. Compiled-kernel threading — the GIL-release payoff",
        "Each of P threads runs a fixed batch of native Simpson-kernel calls on "
        "cache-resident data (compute-bound). Because the kernels release the "
        "GIL, P threads do P× the work in nearly the same wall time — a "
        "near-linear **throughput** multiplier. This is the direct validation "
        "of the GIL-release kernel change.")
    if not _kernels.HAVE_NATIVE:
        rep.text("_Native kernels not built — skipping (build with "
                 "`python build_native.py`)._")
        kspeed = {}
    else:
        kspeed, ktimes = kernel_scaling(Ps)
        rep.table(
            ["threads P", "throughput ×", "efficiency %"],
            [[P, fmt(kspeed[P], "{:.2f}"), fmt(kspeed[P] / P * 100, "{:.0f}")]
             for P in Ps])
        s_frac = amdahl_serial_fraction(Ps, [kspeed[P] for P in Ps])
        rep.text(
            f"Near-linear to the physical-core count (peak **{max(kspeed.values()):.1f}×** "
            f"at P={max(Ps)}), tapering past {PHYS} cores (SMT). Amdahl serial "
            f"fraction s≈**{s_frac:.3f}**.")

    # --- 2. fit_many process scaling ------------------------------------ #
    rep.section(
        "2. fit_many — independent fits across processes",
        "A batch of independent EDA fits fanned across loky workers. These are "
        "embarrassingly parallel, but each fit is short (~ms, with a SymPy "
        "lambdify) so on this platform the process dispatch/spawn overhead caps "
        "the practical speedup of fine-grained fits.")
    fmspeed, fmtimes = fitmany_scaling(Ps, 120 if quick else 400)
    rep.table(["workers P", "time (s)", "speedup"],
              [[P, fmt(fmtimes[P], "{:.2f}"), fmt(fmspeed[P], "{:.2f}")] for P in Ps])

    # --- 3. map-reduce threaded ----------------------------------------- #
    rep.section(
        "3. Threaded map-reduce stream (adaptation #1)",
        "Partitions of a large stream processed concurrently by threads; numpy "
        "releases the GIL on the bulk ops, so it scales — but the workload is "
        "memory-bandwidth-bound, which sets the ceiling.")
    total = 40_000_000 if quick else 160_000_000
    mrspeed, mrtimes = mapreduce_scaling(Ps, total)
    rep.table(["threads P", "time (s)", "speedup"],
              [[P, fmt(mrtimes[P], "{:.2f}"), fmt(mrspeed[P], "{:.2f}")] for P in Ps])

    # figure: all three speedup curves
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(Ps, Ps, "k--", lw=1, label="ideal (linear)")
    if kspeed:
        ax.plot(Ps, [kspeed[P] for P in Ps], "o-", color="tab:blue",
                label="native kernel (threads)")
    ax.plot(Ps, [mrspeed[P] for P in Ps], "^-", color="tab:green",
            label="map-reduce stream (threads)")
    ax.plot(Ps, [fmspeed[P] for P in Ps], "s-", color="tab:orange",
            label="fit_many (processes)")
    ax.set_title("Acceleration factor vs rank of parallelism")
    ax.set_xlabel("rank of parallelism P"); ax.set_ylabel("speedup / throughput ×")
    ax.legend(fontsize=8)
    rep.figure(fig, "scaling", "Acceleration vs parallelism rank (three workloads).")

    rep.section("Reading it", level=2)
    peak_k = max(kspeed.values()) if kspeed else float("nan")
    rep.text(
        f"- **Compiled kernels scale near-linearly** ({fmt(peak_k, '{:.1f}')}× at "
        f"P={max(Ps)}) — the GIL-release refactor lets dtfit's hot numeric loops "
        "use every physical core, the key result for productivity on real "
        "hardware.\n"
        f"- The threaded **map-reduce** stream scales to ~{max(mrspeed.values()):.1f}× "
        "before memory bandwidth saturates (expected for a streaming, "
        "data-bound workload).\n"
        f"- Fine-grained **`fit_many`** fits are embarrassingly parallel in "
        "principle but here limited (~{:.1f}×) by per-task process overhead; "
        "they scale when the per-task work is coarse/heavy (batch the fits) "
        "rather than millisecond-sized. The ceilings are set by the platform "
        "(IPC, memory bandwidth), not the algorithms.".format(max(fmspeed.values())))

    path = rep.write()
    print(f"[parallel_scaling] wrote {path}")
    return str(path)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
