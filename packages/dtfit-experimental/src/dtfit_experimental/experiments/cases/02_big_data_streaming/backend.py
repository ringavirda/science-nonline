"""Backend infrastructure for the big-data / streaming scaling-law experiment.

This module is the **single source of truth for the simulation and estimation
code** behind ``02_big_data_streaming.ipynb``; the notebook imports it and does
all the presentation (tables, figures, narrative). Keeping the infra here means
the chunk generator, the map-reduce volume sweep and the online-cost benchmark
are defined once and the notebook stays a thin, rerunnable layer over them.

"Big-data applicability" for a streaming method is not a single byte count; it
is the *scaling behaviour*: does memory stay flat (O(1) per sample) and time
grow only linearly as the volume increases? If so the method runs at any scale,
10 GB or 10 TB, in bounded memory -- exactly where a batch fit (or a batch NN /
ARIMA that must hold the whole array) runs out of RAM.

It provides two measurement tracks:

* **Volume scaling** -- :func:`volume_track` streams a drifting signal generated
  and consumed in fixed chunks (:func:`_gen_chunk`) through the map-reduce
  adaptation (:class:`PartitionedLSI`), which keeps only O(order) accumulator
  state, and records throughput + peak memory across increasing volumes.
* **Online cost** -- :func:`online_track` measures the per-sample budget of
  :class:`EACFilter` against a sliding-window ``fit_eac`` batch refit, plus the
  tracked frequency / drift-flag history for the presentation figure.

The functions return numbers / arrays / dicts only; no plotting, no report
writing -- the notebook owns all of that.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

import dtfit as dt
from dtfit import PartitionedLSI
from dtfit.streaming import EACFilter

__all__ = [
    "CHUNK", "DOMAIN",
    "volume_track", "online_track",
]

CHUNK = 2_000_000          # samples per chunk (~32 MB of float64 x+y)
DOMAIN = (0.0, 10.0)       # fixed global domain for the additive reduce


def _gen_chunk(t0, t1, n, rng):
    """A noisy exponential-ish signal segment on [t0, t1)."""
    t = np.linspace(t0, t1, n, endpoint=False)
    y = 1.0 * np.exp(0.20 * t) + rng.normal(0.0, 0.05, n)
    return t, y


def volume_track(volumes_gb, *, seed=0, chunk=CHUNK):
    """Stream each volume through PartitionedLSI; record throughput + peak MB.

    Each volume is folded chunk-by-chunk into the map-reduce LSI accumulator,
    which keeps only O(order) state and never stores the data. ``chunk`` sets
    the samples per chunk (lower it to cap transient RAM on small machines).
    Returns a list of dict rows with volume / samples / time / throughput /
    GB-per-second / peak memory."""
    rows = []
    for v in volumes_gb:
        n_total = int(v * 1e9 / 8)           # float64 y bytes
        n_chunks = max(1, n_total // chunk)
        n_total = n_chunks * chunk
        rng = np.random.default_rng(seed)
        acc = PartitionedLSI("a*exp(b*t)", "t", domain=DOMAIN, order=6)
        tracemalloc.start()
        t0 = time.perf_counter()
        span = (DOMAIN[1] - DOMAIN[0]) / n_chunks
        for c in range(n_chunks):
            x, y = _gen_chunk(DOMAIN[0] + c * span, DOMAIN[0] + (c + 1) * span,
                              chunk, rng)
            acc.update(x, y)
        elapsed = time.perf_counter() - t0
        peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        thru = n_total / elapsed / 1e6        # Msamples/s
        gbps = v / elapsed
        rows.append({"gb": v, "n": n_total, "s": elapsed, "msps": thru,
                     "gbps": gbps, "peak_mb": peak_mb})
    return rows


def online_track(n, *, seed=0, batch_sizes=(10_000, 50_000, 250_000)):
    """Per-sample cost + memory of the streaming filter vs heavier updaters.

    Tracks a high-rate sine with a mid-stream frequency jump online with an
    :class:`EACFilter`, timing each ``partial_fit`` and recording the tracked
    frequency and any drift flags. Then contrasts the constant-cost streaming
    update against a *batch* ``fit_eac`` refit at increasing sizes (whose cost
    grows with the history). Returns a dict with the per-step cost, peak memory,
    the batch-refit costs, and the tracking history for the figure."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40, n)
    half = n // 2
    w = np.where(np.arange(n) < half, 1.0, 1.6)
    dt_ = np.diff(t, prepend=t[0])
    phase = np.cumsum(w * dt_)
    y = 3.0 * np.sin(phase) + rng.normal(0, 0.3, n)

    flt = EACFilter("A*sin(w*t)", "t", p0=[2.0, 1.0], window_size=50,
                    q_diag=[1e-3, 5e-4], r=5.0, n_sub=2, adapt_r=True)
    tracemalloc.start()
    costs, track, w_hist, drift = [], [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        flt.partial_fit(t[i], y[i])
        costs.append((time.perf_counter() - t0) * 1e6)
        if flt.drift_flag_:
            drift.append(i)
        track.append(float(flt.predict(np.array([t[i]]))[0]) if len(flt._t) else np.nan)
        w_hist.append(flt.params_["w"])
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    us_step = float(np.mean(costs[100:]))

    # Contrast: the cost of a *batch* re-fit grows with the history length, so
    # tracking a stream by re-fitting all data seen so far is O(N) per step =
    # O(N^2) overall. Measure fit_eac at increasing sizes to show the growth.
    # warm up the SymPy lambdify / solver caches so the first timed size is
    # not inflated by one-off compilation overhead.
    dt.fit_eac(np.linspace(0, 40, 500), np.sin(np.linspace(0, 40, 500)),
               "A*sin(w*t)", "t", p0=[2.0, 1.0])
    batch_costs = []
    for m in batch_sizes:
        tm = np.linspace(0, 40, m)
        ym = 3.0 * np.sin(1.0 * tm) + rng.normal(0, 0.3, m)
        t0 = time.perf_counter()
        dt.fit_eac(tm, ym, "A*sin(w*t)", "t", p0=[2.0, 1.0])
        batch_costs.append((m, (time.perf_counter() - t0) * 1e3))

    return {"us_step": us_step, "peak_mb": peak_mb, "batch_costs": batch_costs,
            "n": n, "t": t, "y": y, "track": np.array(track),
            "w_hist": np.array(w_hist), "drift": drift, "half": half}
