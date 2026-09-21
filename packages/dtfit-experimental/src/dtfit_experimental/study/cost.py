"""Cost measures of the study tier: peak memory of a callable and the state
size of the embedded streaming filter."""

from __future__ import annotations

import tracemalloc
from typing import Callable

import numpy as np


# big data: peak RAM of a callable. Streaming exists to hold that peak flat
# where the whole-array path grows O(N), so measuring it is the whole claim.
def peak_memory(fn: Callable[[], object]) -> tuple[object, float]:
    """Run ``fn``, returning ``(result, peak_MiB)`` over its allocations."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / (1024 * 1024)


# embedded: the streaming filter's deployable, no-malloc state size
def embedded_footprint(n_params: int, window: int, kind: str = "block") -> dict:
    """Words and bytes of the fixed streaming-filter C struct.

    Block-basis filter state is the window buffer (t, y) plus the covariance
    P (n^2), the estimate (n), scratch (n) and about 8 words of bookkeeping.
    A Legendre-spectrum filter adds a read-only projection table, which lives
    in flash rather than SRAM.

    Returns word and byte counts for float32, the deployable size, and float64.
    """
    n = int(n_params)
    sram_words = 2 * window + n * n + 2 * n + 8
    flash_words = 0
    if kind == "legendre":
        order = max(1, n)  # rough: order ~ params; projection (order+1)*window
        flash_words = (order + 1) * window
    return {
        "sram_words": sram_words,
        "sram_bytes_f32": sram_words * 4,
        "sram_bytes_f64": sram_words * 8,
        "flash_words": flash_words,
        "flash_bytes_f32": flash_words * 4,
    }


# MCU datasheet table: name, SRAM bytes, clock MHz, FPU present, MFLOP/s
# (order-of-magnitude estimate, not a measurement).
MCUS = [
    ("AVR ATmega328 (Uno)", 2 * 1024, 16, False, 0.05),
    ("ARM Cortex-M0+ (SAMD21)", 32 * 1024, 48, False, 0.3),
    ("ARM Cortex-M4F (STM32F4)", 192 * 1024, 168, True, 30.0),
    ("ESP32 (LX6 FPU)", 520 * 1024, 240, True, 40.0),
    ("nRF52840", 262144, 64, True, 10.0),
]


def footprint_rows(lat: dict, *, n: int = 3, W: int = 60) -> dict:
    """Live no-malloc state per estimator, which does not grow with the
    stream: the deployable word and byte counts of a hand-coded C struct.

    Args:
        lat: ``{estimator_name: latency_us}``, the median per-step latency
            from an accuracy sweep, keyed by exactly
            "dtfit block filter", "dtfit Legendre filter",
            "EKF (params-as-state)" and "RLS (AR predictor)".
        n: Physical-parameter count of the tracked model.
        W: Window size (samples).

    Returns:
        A dict holding ``state`` (the per-estimator state table; the
        "Kalman-CA (3-axis)" row's ``latency_us`` is always ``None``, since
        it is not one of the swept estimators), ``track32`` (bytes of an
        ``n``-axis tracker's float32 block-filter state), ``mcu`` (the
        MCU-fit table), ``sweep_W`` / ``sweep`` (the resident-state sweep in
        float32 bytes over window size for a few parameter counts) and
        ``lat`` (the argument, passed through).

    Raises:
        KeyError: If ``lat`` is missing any of the four estimator names
            above.
    """
    ea = embedded_footprint(n, W, kind="block")
    leg = embedded_footprint(n, W, kind="legendre")
    ekf_words = n * n + 2 * n + 8
    rls_words = 4 * 4 + 4 + 4
    kf_words = 3 * (3 * 3 + 3) + 8
    state = [
        dict(estimator="dtfit block filter", state_words=ea["sram_words"],
             float32_B=str(ea["sram_bytes_f32"]), window_buffer=f"yes (W={W})",
             params="yes", latency_us=lat["dtfit block filter"]),
        dict(estimator="dtfit Legendre filter", state_words=leg["sram_words"],
             float32_B=f"{leg['sram_bytes_f32']} +{leg['flash_bytes_f32']}B flash",
             window_buffer=f"yes (W={W})", params="yes",
             latency_us=lat["dtfit Legendre filter"]),
        dict(estimator="EKF (params-as-state)", state_words=ekf_words,
             float32_B=str(ekf_words * 4), window_buffer="no", params="yes",
             latency_us=lat["EKF (params-as-state)"]),
        dict(estimator="RLS (AR predictor)", state_words=rls_words,
             float32_B=str(rls_words * 4), window_buffer="no", params="no",
             latency_us=lat["RLS (AR predictor)"]),
        dict(estimator="Kalman-CA (3-axis)", state_words=kf_words,
             float32_B=str(kf_words * 4), window_buffer="no", params="no",
             latency_us=None),
    ]
    track32 = ea["sram_bytes_f32"] * 3        # a 3-axis tracker
    mcu = [dict(MCU=name, SRAM_KB=sram // 1024, clock_MHz=clock,
                FPU=("yes" if fpu else "no (soft)"), MFLOPs=mflops,
                fits=("yes" if track32 < sram * 0.5
                      else ("tight" if track32 < sram else "no")))
           for name, sram, clock, fpu, mflops in MCUS]
    Ws = np.arange(10, 110, 5)
    sweep = {nn: [embedded_footprint(nn, int(w))["sram_bytes_f32"] for w in Ws]
             for nn in (2, 3, 5)}
    return dict(state=state, track32=track32, mcu=mcu,
                sweep_W=Ws, sweep=sweep, lat=lat)
