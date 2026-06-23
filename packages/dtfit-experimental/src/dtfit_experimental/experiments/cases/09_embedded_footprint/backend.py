"""Backend infrastructure for the embedded / low-resource footprint experiment.

This module is the **single source of truth for the measurement and accounting
code** behind ``09_embedded_footprint.ipynb``; the notebook imports it and does
all the presentation (tables, figures, narrative). Keeping the infra here means
the latency timers, state-size arithmetic and MCU-fit model are defined once and
the notebook stays a thin, rerunnable layer over them.

It answers one question -- **can the streaming filters run on the kind of
microcontroller you would bolt to a GPS module (Arduino / STM32 / ESP32
class)?** -- via the three things that decide it:

* the **deployable state size** -- :func:`state_doubles_eac`, :func:`state_bytes`,
  :func:`state_doubles_lsi_ram`, :func:`const_doubles_lsi`,
  :func:`state_doubles_kalman`, and the rough :func:`approx_flops_per_update`:
  the exact, no-malloc C-struct word/byte counts (the real embedded number);
* the **measured per-sample latency** -- :func:`measure_latency`,
  :func:`measure_kalman_latency`, :func:`measure_curvefit_latency`, plus the
  Python-object footprint :func:`measure_resident_python` (the interpreter
  overhead a C port removes, reported only for contrast);
* the **MCU fit model** -- the :data:`MCUS` datasheet table and the
  :data:`CONFIGS` latency/memory configurations.

The honest framing: NumPy does not run on an AVR, so the latency figures are the
desktop reference for the *algorithm shape*, while the memory figures are the
exact C-struct size that does deploy. No matplotlib, no narrative, no report.md
-- those live in the notebook.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
from scipy.optimize import curve_fit


from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "CONFIGS", "MCUS",
    "state_doubles_eac", "state_bytes", "state_doubles_lsi_ram",
    "const_doubles_lsi", "state_doubles_kalman", "approx_flops_per_update",
    "measure_latency", "measure_kalman_latency", "measure_curvefit_latency",
    "measure_resident_python",
]


# --------------------------------------------------------------------------- #
# configurations: (label, model expr, var, p0, kwargs, n params, window W)
# the streaming-filter setups whose latency and memory are measured/sized.
# --------------------------------------------------------------------------- #
CONFIGS = [
    ("CA quadratic (GPS axis)", "c0 + c1*t + c2*t**2", "t", [0.0, 0.0, 0.0],
     dict(window_size=15, q_diag=[1e-2] * 3, r=0.5, n_sub=2, adapt_r=True), 3, 15),
    ("damped sine (control ID)", "A*exp(-d*t)*sin(w*t)", "t", [1.0, 0.1, 1.0],
     dict(window_size=50, q_diag=[1e-2] * 3, r=1.0, n_sub=2), 3, 50),
    ("linear (range smoother)", "a + b*t", "t", [0.0, 0.0],
     dict(window_size=20, q_diag=[1e-1, 1e-2], r=0.6, n_sub=2, adapt_r=True), 2, 20),
]


# --------------------------------------------------------------------------- #
# MCU classes (datasheet SRAM / clock); compute estimate is deliberately rough
# name, SRAM bytes, clock MHz, has FPU, eff. MFLOP/s (rough, soft-float
# penalised where no FPU)
# --------------------------------------------------------------------------- #
MCUS = [
    ("AVR ATmega328 (Uno/Nano)", 2 * 1024, 16, False, 0.05),
    ("ARM Cortex-M0+ (SAMD21/Zero)", 32 * 1024, 48, False, 0.3),
    ("ARM Cortex-M4F (STM32F4/Teensy)", 192 * 1024, 168, True, 30.0),
    ("ESP32 (Xtensa LX6 FPU)", 520 * 1024, 240, True, 40.0),
]


# --------------------------------------------------------------------------- #
# algorithmic (deployable) state size -- the C struct, the real embedded number
# --------------------------------------------------------------------------- #
def state_doubles_eac(n: int, w: int, n_sub: int = 2) -> int:
    """Floating-point words a minimal C port of EACFilter must keep alive
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
    return state_doubles_eac(n, w) * dtype_bytes


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
    """Rough FLOP count for one EACFilter update with a degree-(n-1)
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
    """Mean wall-clock us per `partial_fit` on a warmed-up filter."""
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
    """Mean us per 3-axis CA Kalman update (recursive, no window)."""
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
    """Mean us per step for the batch alternative: refit a CA quadratic on the
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
