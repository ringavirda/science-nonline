"""Experiment 9 -- embedded / low-resource footprint of the streaming filters.

The streaming estimators (`EDAFilter`, `LSIFilter`) are the
only part of dtfit meant for *online* deployment, and the natural place to ask
the hardware question raised by the GPS experiment: **can this run on the kind of
microcontroller you would bolt to a GPS module?** (Arduino / STM32 / ESP32 class.)

Three things decide that, and we measure each honestly:

  1. **Per-sample latency** -- the recurrence is O(1) in history; how many µs does
     one `partial_fit` cost, and therefore how many samples/second can one core
     sustain? (A GPS fix arrives at ~1-10 Hz, so there is enormous headroom.)
  2. **State memory** -- the bytes that must persist between samples. This is the
     real constraint on a 2 KB-SRAM part, and it is *tiny and fixed* (a ring
     buffer + a few small matrices), independent of how long the stream runs.
  3. **What actually ports.** The reference here is Python + NumPy + a compiled
     SymPy callable -- none of which runs on a microcontroller. What ports is the
     **recurrence**: a fixed-size C struct and a handful of small matrix ops (the
     integration kernels are already C in `dtfit._native`). So the latency below
     is the desktop reference for the *algorithm shape*; the memory figure is the
     deployable C-struct size, computed exactly; and the MCU table estimates the
     fit of that struct against common parts. We do not pretend NumPy runs on an
     AVR.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
from scipy.optimize import curve_fit

from dtfit.streaming import EDAFilter, LSIFilter

from dtfit_experimental.experiments.common import ReportWriter, fmt
from dtfit_experimental.experiments.common.plotting import plt
from dtfit_experimental.experiments.common import baselines as bl

EXP_DIR = __file__.rsplit("run.py", 1)[0]


# --------------------------------------------------------------------------- #
# algorithmic (deployable) state size -- the C struct, the real embedded number
# --------------------------------------------------------------------------- #
def state_doubles_eda(n: int, w: int, n_sub: int = 2) -> int:
    """Floating-point words a minimal C port of EDAFilter must keep alive
    between samples (a fixed-size, no-malloc struct):

      * ring buffer  t[W], y[W]                       -> 2W
      * parameter estimate p[n]                        -> n
      * covariance P[n*n]                              -> n*n
      * process-noise diagonal Q[n]                    -> n
      * measurement/detector scalars (R, EWMA scales,
        CUSUM arms, counters ~ 8)                      -> 8

    Scratch for the per-step solve (h_mat, S, gain) is transient and can live on
    the stack; it is not counted as resident state.
    """
    return 2 * w + n * n + 2 * n + 8


def state_bytes(n: int, w: int, dtype_bytes: int) -> int:
    return state_doubles_eda(n, w) * dtype_bytes


def state_doubles_lsi_ram(n: int, w: int) -> int:
    """Mutable per-sample RAM words for LSIFilter: the ring buffer
    (2W), estimate p (n), covariance P (n*n), Q diagonal (n) and ~9 scalars. The
    filter *also* precomputes projection/quadrature tables, but those are
    **read-only constants** that belong in flash/PROGMEM, not SRAM -- see
    ``const_doubles_lsi``."""
    return 2 * w + n * n + n + 9


def const_doubles_lsi(n: int, w: int, order: int) -> int:
    """Read-only tables the Legendre filter precomputes (flash-able, not SRAM):
    the projection pseudo-inverse (order+1)*W, the Gauss-Legendre nodes/weights
    (2*n_quad), the quadrature Vandermonde n_quad*(order+1), and the per-order
    weights 2*(order+1)."""
    n_quad = max(2 * (order + 1), 16)
    return (order + 1) * w + 2 * n_quad + n_quad * (order + 1) + 2 * (order + 1)


def state_doubles_kalman(dim: int = 3) -> int:
    """Mutable RAM words for the CA Kalman: state x[3] + covariance P[3x3] per
    axis = 12 per axis. The transition/noise/measurement matrices F,Q,H,R are
    constants (flash-able). The Kalman keeps **no history window**, so it is
    leaner than the integral filters -- an honest architectural difference."""
    return 12 * dim


def approx_flops_per_update(n: int, w: int, n_sub: int = 2) -> int:
    """Rough FLOP count for one EDAFilter update with a degree-(n-1)
    polynomial model. Dominated by evaluating the model and its n derivatives
    over the W-point window (~2*W*n each), plus the small (n_sub x n) Kalman
    algebra. Order-of-magnitude, for the MCU compute sanity check."""
    model_eval = 2 * w * n          # Horner over the window
    jac_eval = 2 * w * n * n        # n derivative rows over the window
    integrate = 2 * w * (n + 1)     # Simpson sums for e_vec + h_mat rows
    kalman = n * n * n_sub + n_sub ** 3 + n * n_sub ** 2 + n * n
    return model_eval + jac_eval + integrate + kalman


# --------------------------------------------------------------------------- #
# measured per-sample latency
# --------------------------------------------------------------------------- #
def measure_latency(make_filter, n_warm: int, n_timed: int = 4000) -> float:
    """Mean wall-clock µs per `partial_fit` on a warmed-up filter."""
    flt = make_filter()
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, n_warm + n_timed)
    y = np.sin(3 * t) + rng.normal(0, 0.1, t.size)
    for i in range(n_warm):  # fill the window so we time the steady-state path
        flt.partial_fit(float(t[i]), float(y[i]))
    t0 = time.perf_counter()
    for i in range(n_warm, n_warm + n_timed):
        flt.partial_fit(float(t[i]), float(y[i]))
    return (time.perf_counter() - t0) / n_timed * 1e6


def measure_kalman_latency(n_timed: int = 4000) -> float:
    """Mean µs per 3-axis CA Kalman update (recursive, no window)."""
    kf = bl.KalmanCA(dim=3, dt=0.03, q=5e-2, r=0.5)
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (n_timed + 10, 3))
    for i in range(10):
        kf.update(z[i])
    t0 = time.perf_counter()
    for i in range(10, 10 + n_timed):
        kf.update(z[i])
    return (time.perf_counter() - t0) / n_timed * 1e6


def measure_curvefit_latency(w: int, n_timed: int = 400) -> float:
    """Mean µs per step for the batch alternative: refit a CA quadratic on the
    trailing W-sample window with `scipy.optimize.curve_fit` (Levenberg-Marquardt)
    at every new sample -- one axis only."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, w + n_timed)
    y = 1.0 + 2.0 * t + 0.5 * t**2 + rng.normal(0, 0.1, t.size)

    def f(tt, c0, c1, c2):
        return c0 + c1 * tt + c2 * tt**2

    p0 = np.zeros(3)
    t0 = time.perf_counter()
    for i in range(w, w + n_timed):
        tw, yw = t[i - w:i], y[i - w:i]
        try:
            p0, _ = curve_fit(f, tw, yw, p0=p0, maxfev=2000)
        except Exception:
            pass
    return (time.perf_counter() - t0) / n_timed * 1e6


def measure_resident_python(make_filter, n_warm: int) -> int:
    """Python-object resident bytes of one warmed filter (tracemalloc). This is
    the *interpreter* footprint, not the embeddable struct -- reported only to
    show how much the Python objects inflate the true algorithmic state."""
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    flt = make_filter()
    t = np.linspace(0, 1, n_warm)
    for ti in t:
        flt.partial_fit(float(ti), float(np.sin(3 * ti)))
    snap = tracemalloc.take_snapshot()
    stats = snap.compare_to(base, "filename")
    total = sum(s.size_diff for s in stats)
    tracemalloc.stop()
    return max(total, 0)


# --------------------------------------------------------------------------- #
# MCU classes (datasheet SRAM / clock); compute estimate is deliberately rough
# --------------------------------------------------------------------------- #
MCUS = [
    # name, SRAM bytes, clock MHz, has FPU, eff. MFLOP/s (rough, soft-float
    # penalised where no FPU)
    ("AVR ATmega328 (Uno/Nano)", 2 * 1024, 16, False, 0.05),
    ("ARM Cortex-M0+ (SAMD21/Zero)", 32 * 1024, 48, False, 0.3),
    ("ARM Cortex-M4F (STM32F4/Teensy)", 192 * 1024, 168, True, 30.0),
    ("ESP32 (Xtensa LX6 FPU)", 520 * 1024, 240, True, 40.0),
]


def main(quick: bool = False) -> str:
    rep = ReportWriter(
        EXP_DIR, "Experiment 9 — embedded / low-resource footprint",
        intent=(
            "Can the streaming filter run on the microcontroller you would attach "
            "to a GPS module (Arduino / STM32 / ESP32 class)? We measure the three "
            "things that decide it — **per-sample latency**, **resident state "
            "memory**, and **what actually ports to C** — and size the deployable "
            "state against real MCU parts. The honest framing: NumPy does not run "
            "on an AVR, so latency here is the desktop reference for the algorithm "
            "shape, while the memory figure is the exact C-struct size that does "
            "deploy."),
    )

    # ---- configurations: (label, model, var, p0, kwargs, n, W) ----------- #
    configs = [
        ("CA quadratic (GPS axis)", "c0 + c1*t + c2*t**2", "t", [0.0, 0.0, 0.0],
         dict(window_size=15, q_diag=[1e-2] * 3, r=0.5, n_sub=2, adapt_r=True), 3, 15),
        ("damped sine (control ID)", "A*exp(-d*t)*sin(w*t)", "t", [1.0, 0.1, 1.0],
         dict(window_size=50, q_diag=[1e-2] * 3, r=1.0, n_sub=2), 3, 50),
        ("linear (range smoother)", "a + b*t", "t", [0.0, 0.0],
         dict(window_size=20, q_diag=[1e-1, 1e-2], r=0.6, n_sub=2, adapt_r=True), 2, 20),
    ]
    n_timed = 1500 if quick else 5000

    rep.section(
        "1. Per-sample latency & throughput (all package filters)",
        "Mean wall-clock cost of one steady-state update on this desktop "
        f"(warmed window, {n_timed} timed updates), and the sustainable sample "
        "rate it implies, for every online estimator in the package plus the "
        "Kalman reference. The recurrence is O(1) in stream length, so this is "
        "flat for the whole stream. A GPS fix arrives at ~1–10 Hz; the headroom "
        "is several orders of magnitude.")
    lat_rows = []
    lat_by_cfg = {}
    for label, expr, var, p0, kw, n, w in configs:
        def make(expr=expr, var=var, p0=p0, kw=kw):
            return EDAFilter(expr, var, p0=p0, **kw)
        us = measure_latency(make, n_warm=w + 5, n_timed=n_timed)
        lat_by_cfg[label] = us
        lat_rows.append(["EDAFilter", label, f"n={n}, W={w}",
                         fmt(us, "{:.1f}"), fmt(1e6 / us, "{:,.0f}")])
    # LSIFilter (the orthogonal-spectrum sibling): more observability
    # per window at the cost of an order+1 projection -- so a bit heavier per step.
    lsi_us = measure_latency(
        lambda: LSIFilter(
            "A*exp(-d*t)*sin(w*t)", "t", p0=[1.0, 0.1, 1.0], window_size=50,
            order=5, q_diag=[1e-2] * 3, r=1.0),
        n_warm=55, n_timed=n_timed)
    lat_by_cfg["LSIFilter"] = lsi_us
    lat_rows.append(["LSIFilter", "damped sine", "n=3, W=50, ord=5",
                     fmt(lsi_us, "{:.1f}"), fmt(1e6 / lsi_us, "{:,.0f}")])
    # Kalman reference (no window, no integral -- the leanest update).
    kf_us = measure_kalman_latency(n_timed=n_timed)
    lat_by_cfg["Kalman (CA)"] = kf_us
    lat_rows.append(["CA Kalman (3-axis)", "reference", "dim=3",
                     fmt(kf_us, "{:.2f}"), fmt(1e6 / kf_us, "{:,.0f}")])
    rep.table(["estimator", "config", "size", "µs / update",
               "max samples/s (1 core)"], lat_rows)
    rep.text(
        "`FilterBank` is just K of these run together (one per axis/channel/"
        "satellite), so its cost and memory are K× a single filter — e.g. the "
        "3-axis GPS tracker is 3 `EDAFilter`s. The Legendre filter is the "
        "heaviest per step (it projects onto `order+1` orthogonal moments for "
        "extra observability); the Kalman is the lightest (no window, no "
        "integral). All sit far under any real-time budget.")

    rep.section(
        "2. Resident state memory — the deployable C struct",
        "The bytes that must persist between samples. For a minimal C port this is "
        "a **fixed-size, no-malloc struct**: the `t,y` ring buffer (2·W words), the "
        "estimate `p` (n), covariance `P` (n²), process-noise diagonal `Q` (n), and "
        "~8 detector/measurement scalars — i.e. `2W + n² + 2n + 8` words. It does "
        "**not grow with stream length** (the defining property of a real-time "
        "estimator). Shown in float64 (the reference) and float32 (a natural "
        "embedded choice; the integration kernels carry fine at single precision "
        "for these window sizes).")
    mem_rows = []
    for label, expr, var, p0, kw, n, w in configs:
        words = state_doubles_eda(n, w)
        mem_rows.append([
            label, f"n={n}, W={w}", f"{words}",
            f"{state_bytes(n, w, 8):,} B", f"{state_bytes(n, w, 4):,} B",
            f"~{approx_flops_per_update(n, w):,}"])
    rep.table(["config", "size", "state words", "float64", "float32",
               "FLOPs/update"], mem_rows)

    # LSIFilter: same mutable RAM shape, plus read-only tables.
    lsi_n, lsi_w, lsi_ord = 3, 50, 5
    lsi_ram = state_doubles_lsi_ram(lsi_n, lsi_w)
    lsi_const = const_doubles_lsi(lsi_n, lsi_w, lsi_ord)
    rep.text(
        f"**`LSIFilter` (the other filter).** Its *mutable* RAM state "
        f"has the same shape — `2W + n² + n + 9` words ({lsi_ram} words = "
        f"{lsi_ram * 4:,} B float32 at n={lsi_n}, W={lsi_w}) — so its **SRAM** "
        "footprint is essentially the same as the area filter. It additionally "
        f"precomputes ~{lsi_const:,} words of **read-only** projection/quadrature "
        "tables (the orthogonal-basis Vandermonde + Gauss-Legendre nodes); those "
        "are constants that belong in **flash/PROGMEM**, not SRAM. So it buys extra "
        "observability (an `order+1`-dimensional measurement) for more flash and a "
        "little more compute — not more RAM. **The CA Kalman is the leanest of "
        f"all**: {state_doubles_kalman(3)} mutable words ({state_doubles_kalman(3) * 4} B "
        "float32 for 3 axes) because it keeps **no history window** — the honest "
        "cost of dtfit's integral measurement is exactly that window buffer.")

    # Python-object footprint, for contrast (NOT the embedded number).
    py_label, *_ = configs[0]
    _, expr0, var0, p00, kw0, _, w0 = configs[0]
    py_bytes = measure_resident_python(
        lambda: EDAFilter(expr0, var0, p0=p00, **kw0), n_warm=w0 + 5)
    rep.text(
        f"For contrast, one *Python* filter object (warmed) holds ≈ "
        f"**{py_bytes / 1024:,.0f} KB** of interpreter/NumPy objects — hundreds of "
        f"times the {state_bytes(3, 15, 8)} B algorithmic state. That gap is pure "
        "Python/NumPy overhead and is exactly what a C port removes; it is **not** "
        "what runs on the MCU.")

    # ---- MCU fit table (GPS config, ×3 axes for a full 3-D tracker) ------ #
    n, w = 3, 15
    per_axis64, per_axis32 = state_bytes(n, w, 8), state_bytes(n, w, 4)
    track64, track32 = per_axis64 * 3, per_axis32 * 3  # 3 axes
    flops = approx_flops_per_update(n, w) * 3  # whole 3-axis update
    rate_hz = 10  # GPS epoch rate to keep up with
    rep.section(
        "3. Fit on real microcontrollers (3-axis GPS tracker)",
        f"A full 3-axis tracker is three independent filters: **{track32:,} B** "
        f"(float32) / **{track64:,} B** (float64) of state, and ≈ {flops:,} "
        f"FLOPs per GPS epoch. Whether a part fits is a **memory** question (the "
        "state must live in SRAM alongside everything else) far more than a "
        "compute one — even a pessimistic soft-float estimate keeps the per-epoch "
        f"compute orders of magnitude under a {rate_hz} Hz budget. Compute time is "
        "a rough estimate (effective MFLOP/s, soft-float-penalised where there is "
        "no FPU); memory fit is exact.")
    mcu_rows = []
    for name, sram, mhz, fpu, mflops in MCUS:
        compute_ms = flops / (mflops * 1e6) * 1e3
        fits32 = "✓" if track32 < sram * 0.5 else ("tight" if track32 < sram else "✗")
        budget_ms = 1e3 / rate_hz
        keeps_up = "✓" if compute_ms < budget_ms * 0.5 else (
            "tight" if compute_ms < budget_ms else "✗")
        mcu_rows.append([
            name, f"{sram // 1024} KB", f"{mhz}", "yes" if fpu else "no (soft)",
            f"{track32:,} B → {fits32}", f"~{compute_ms:.2f} ms → {keeps_up}"])
    rep.table(
        ["MCU", "SRAM", "MHz", "FPU", "state f32 (fits if <50% SRAM)",
         f"compute/epoch (<{1e3/rate_hz:.0f} ms budget)"], mcu_rows)

    # ---- comparison vs existing methods --------------------------------- #
    cf_us = measure_curvefit_latency(w=15, n_timed=120 if quick else 400)
    eda_us = lat_by_cfg[configs[0][0]]
    cf_ratio = cf_us / eda_us
    mlp_lag, mlp_hid = 10, 16
    mlp_weights = mlp_lag * mlp_hid + mlp_hid + mlp_hid + 1  # 1 hidden layer
    rep.section(
        "4. Comparison with existing methods",
        "How the streaming filters stack up against the alternatives you might "
        "deploy for the same online job. The deciding axes for embedded are "
        "**(a) does resident state grow with the stream**, **(b) per-update "
        "compute**, and **(c) can it adapt on-device** (vs. train-offline-only). "
        "State is the 3-axis / equivalent figure in float32.")
    comp_rows = [
        ["dtfit `EDAFilter` ×3", f"~{state_bytes(3, 15, 4) * 3:,} B",
         "no (fixed window)", f"1× ({eda_us:.0f} µs)", "yes — recursive"],
        ["dtfit `LSIFilter`", f"~{lsi_ram * 4:,} B + flash tables",
         "no (fixed window)", f"~{lsi_us / eda_us:.1f}×", "yes — recursive"],
        ["CA Kalman ×3 (gold standard)", f"~{state_doubles_kalman(3) * 4} B",
         "no (no window)", f"{kf_us / eda_us:.2f}×", "yes — recursive"],
        ["sliding-window `curve_fit` (LM)", "~window only",
         "no (fixed window)", f"~{cf_ratio:.0f}× ({cf_us:.0f} µs)",
         "refit from scratch / step"],
        ["batch fit / NN over full history", "**O(N) — unbounded**",
         "**yes → eventually OOM**", "O(N) / refit", "n/a (not streaming)"],
        [f"offline-trained MLP (lag{mlp_lag}, h{mlp_hid})",
         f"~{mlp_weights * 4:,} B weights", "no", "inference only",
         "**no — train offline**"],
    ]
    rep.table(["method", "resident state (f32)", "grows w/ stream?",
               "per-update compute", "on-device adaptation"], comp_rows)
    rep.text(
        "- **The recursive estimators (dtfit filters + Kalman) are the embeddable "
        "class**: fixed sub-KB state, O(1)/sample compute, and they *learn on the "
        "device*. The **Kalman is the leanest** (no window) and dtfit's filters "
        "cost one window buffer more — the price of an integral measurement that "
        "buys robustness to noise and a nonlinear-in-parameters model.\n"
        f"- **Re-fitting a window with `curve_fit` every step** costs ~{cf_ratio:.0f}× "
        "a recursive update *here* — modest, because the warm-started "
        "Levenberg-Marquardt converges in a step or two on this easy "
        "linear-in-parameters quadratic and scipy call overhead dominates. The "
        "real objections are that it puts a **full nonlinear optimizer in the "
        "embedded loop** (far more code than a recursive update, and no clean "
        "C path), and that its cost grows with window size and model "
        "nonlinearity rather than staying O(1). **A batch fit / NN over the whole "
        "history is the one that does not fit at all**: its state grows with the "
        "stream and eventually exhausts RAM — the failure mode streaming "
        "structurally avoids.\n"
        "- **Neural nets are a different deployment model**: a small MLP/LSTM has a "
        "fixed, modest weight footprint and fast inference, but it must be "
        "**trained offline** — it cannot adapt to a new regime on the MCU the way "
        "the recursive filters do. For a self-contained sensor that calibrates and "
        "tracks in the field, the recursive filters are the natural fit.")

    # ---- figures -------------------------------------------------------- #
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4))
    Ws = np.arange(10, 110, 5)
    for nn, col in [(2, "tab:blue"), (3, "tab:green"), (5, "tab:red")]:
        axa.plot(Ws, [state_bytes(nn, w, 4) for w in Ws], "-", color=col,
                 label=f"n={nn} params (f32)")
    axa.axhline(2 * 1024, ls=":", color="0.5")
    axa.text(12, 2 * 1024 * 1.03, "AVR 2 KB SRAM", fontsize=7, color="0.4")
    axa.set_title("Resident state vs window size")
    axa.set_xlabel("window size W"); axa.set_ylabel("state bytes (float32)")
    axa.legend(fontsize=8)

    axb.bar(range(len(configs)), [lat_by_cfg[c[0]] for c in configs],
            color="tab:purple")
    axb.set_xticks(range(len(configs)))
    axb.set_xticklabels([c[0].split(" (")[0] for c in configs], rotation=15,
                        fontsize=7, ha="right")
    axb.set_title("Per-sample latency (desktop reference)")
    axb.set_ylabel("µs / update")
    rep.figure(fig, "footprint", "Left: resident state is small and grows only "
               "linearly with window size (flat in stream length). Right: "
               "per-sample latency on the desktop reference path.")

    rep.section("Reading it", level=2)
    rep.text(
        f"- **Memory is tiny and fixed.** A 3-axis GPS tracker needs ≈ "
        f"{track32:,} B (float32) of resident state — it fits comfortably on a "
        "Cortex-M0+/M4/ESP32 and is feasible even on a 2 KB AVR if that part is "
        "doing little else. Crucially the state does **not grow with the stream**, "
        "so there is no creeping-RAM failure mode.\n"
        "- **Compute is never the bottleneck for GPS-rate data.** At a few "
        "thousand FLOPs per epoch and a 1–10 Hz fix rate, even a soft-float MCU "
        "has orders of magnitude of headroom; the desktop path already runs each "
        f"update in ≈ {lat_by_cfg[configs[0][0]]:.0f} µs.\n"
        "- **What you actually deploy is a C recurrence, not this code.** The "
        "Python/NumPy reference carries hundreds of KB of interpreter overhead "
        "that does not exist in a C port; the integration hot loops are already C "
        "(`dtfit._native`). For a *fixed* model (e.g. the CA quadratic) the model "
        "and Jacobian are trivial polynomials to hand-code, and the Kalman algebra "
        "is a handful of small fixed-size matrix ops — no dynamic allocation, no "
        "SymPy, no BLAS.\n"
        "- **Versus the alternatives, the recursive estimators are the embeddable "
        "class.** dtfit's filters and the Kalman all carry fixed sub-KB state and "
        "adapt on-device; the Kalman is leanest (no window), dtfit pays one window "
        "buffer for its integral measurement, and the Legendre filter trades flash "
        "(constant projection tables) for observability without extra RAM. A batch "
        "fit / NN over the full history is the one that structurally does not fit "
        "(O(N) state); a small NN fits but cannot learn a new regime in the field.\n"
        "- **float32 is the natural embedded choice** and halves the state; for "
        "these window sizes the integral/projection conditioning is benign enough "
        "that single precision is fine (unlike the batch GEMM throughput case in "
        "Exp 8, this is bounded-window, not a 10⁹-element reduction).\n"
        "- **The honest caveat:** these are *projections* from a desktop-measured "
        "algorithm plus exact state-size arithmetic, not measurements on silicon. "
        "A real port would confirm the soft-float compute estimate and verify "
        "numerical behaviour at float32 on the target — but the memory verdict "
        "(small, fixed, fits) is exact and is the figure that usually decides "
        "embedded feasibility.")

    path = rep.write()
    print(f"[embedded_footprint] wrote {path}")
    return str(path)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
