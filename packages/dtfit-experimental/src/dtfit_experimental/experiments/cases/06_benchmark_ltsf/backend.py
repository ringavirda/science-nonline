"""Backend infrastructure for the LTSF-benchmark experiment.

This module is the **single source of truth for the data-loading and forecasting
code** behind ``06_benchmark_ltsf.ipynb``; the notebook imports it and does all
the presentation (tables, figures, narrative). Keeping the infra here means the
forecasters / benchmark protocol are defined once and the notebook stays a thin,
rerunnable layer over them.

The experiment places dtfit on the **exact** long-term-forecasting (LTSF)
benchmark used by DLinear (arXiv:2205.13504), TimesNet (arXiv:2210.02186) and
Time-LLM (arXiv:2310.01728): standard splits, train-fit z-score normalization,
lookback->horizon windows, MSE/MAE on the normalized values. The papers'
**published** MSE numbers are transcribed (cited) rather than re-run, so dtfit's
measured numbers sit comparably alongside them.

It provides:

* the **published-results table** -- :data:`PUBLISHED_MSE` (transcribed from the
  papers' ar5iv editions) and the benchmark constants :data:`HORIZONS`,
  :data:`LOOKBACK`, :data:`DAMP`, :data:`SEASONAL_FRAC`;
* the **forecasters** -- :func:`series_extrapolate`, three fit-then-extrapolate
  variants (``"legendre"`` trend, ``"fourier"`` seasonal, ``"trend_seasonal"``
  DLinear-style decomposition), all NLinear-anchored to the last observation;
* the **benchmark evaluation** -- :func:`evaluate` (MSE/MAE over the test
  windows) and :func:`run_dataset` / :func:`seasonal_helped` batch helpers.

The data itself is loaded through the shared
:mod:`dtfit_experimental.experiments.common.datasets` LTSF loader, which
reproduces the Informer/Autoformer pipeline the papers all use.
"""

from __future__ import annotations

import numpy as np

from dtfit_experimental.experiments.common import datasets as ds

__all__ = [
    "PUBLISHED_MSE", "HORIZONS", "LOOKBACK", "DAMP", "SEASONAL_FRAC",
    "available", "series_extrapolate", "evaluate", "run_dataset",
    "seasonal_helped", "sample_window",
]

# Published MULTIVARIATE MSE on the LTSF benchmark (lookback 96), transcribed
# from the papers' ar5iv editions. Horizons 96/192/336/720.
#   DLinear  : arXiv:2205.13504 (table 2)   -- note: paper uses a longer lookback
#   TimesNet : arXiv:2210.02186             -- lookback 96
#   Time-LLM : arXiv:2310.01728             -- lookback 96
PUBLISHED_MSE = {
    "ETTh1": {
        "DLinear": [0.375, 0.405, 0.439, 0.472],
        "TimesNet": [0.384, 0.436, 0.491, 0.521],
        "Time-LLM": [0.362, 0.398, 0.430, 0.442],
    },
    "ETTm1": {
        "DLinear": [0.299, 0.335, 0.369, 0.425],
        "TimesNet": [0.338, 0.374, 0.410, 0.478],
        "Time-LLM": [0.272, 0.310, 0.352, 0.383],
    },
    "weather": {
        "DLinear": [0.176, 0.220, 0.265, 0.323],
        "TimesNet": [0.172, 0.219, 0.280, 0.365],
        "Time-LLM": [0.147, 0.189, 0.262, 0.304],
    },
}
HORIZONS = [96, 192, 336, 720]
LOOKBACK = 96
DAMP = 0.95  # damped-trend factor: saturates long-horizon trend growth
SEASONAL_FRAC = 0.15  # min residual-energy share for a harmonic to be continued


def available() -> list[str]:
    """LTSF dataset keys whose CSV is present locally (re-export of the loader)."""
    return ds.available()


def series_extrapolate(look, H, order, basis="trend_seasonal", n_harm=3):
    """Forecast H steps for each channel of the ``(L, C)`` lookback.

    Three forecasters share the same fit-then-extrapolate spirit; all are
    NLinear-anchored to the last observation for continuity:

    * ``"legendre"`` -- a low-order Legendre (LSI empirical-spectrum) **trend**
      only. The trend-restorer baseline.
    * ``"fourier"``  -- low-frequency harmonics over the lookback continued
      forward (adaptation #2).
    * ``"trend_seasonal"`` -- a DLinear-style decomposition: the low-order
      Legendre trend **plus** a data-driven Fourier seasonal term fitted on the
      detrended residual and extrapolated periodically. This models the
      *restorable* structure (trend + seasonality), not the trend alone.
    """
    L, C = look.shape
    anchor = look[-1:]            # (1, C) last observed value per channel
    u = np.linspace(-1.0, 1.0, L)
    step = (u[-1] - u[0]) / (L - 1)
    u_fut = u[-1] + step * np.arange(1, H + 1)

    if basis == "fourier":
        d = look - anchor
        K = order
        idx = np.arange(L)
        cols = [np.ones(L)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * idx / L), np.sin(2 * np.pi * k * idx / L)]
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), d, rcond=None)
        fut = np.arange(L, L + H)
        colsf = [np.ones(H)]
        for k in range(1, K + 1):
            colsf += [np.cos(2 * np.pi * k * fut / L), np.sin(2 * np.pi * k * fut / L)]
        return np.column_stack(colsf) @ coef + anchor

    # --- low-order Legendre trend (shared by 'legendre' and 'trend_seasonal') -- #
    from numpy.polynomial.legendre import legvander
    d = look - anchor
    V = legvander(u, order)                          # (L, order+1)
    tcoef, *_ = np.linalg.lstsq(V, d, rcond=None)
    trend_in = V @ tcoef + anchor                    # (L, C) in-sample trend
    # Damped-trend extrapolation: a slope fit on a short, noisy lookback diverges
    # if continued linearly over long horizons. We keep the LSI linear trend but
    # saturate its growth geometrically (damped-trend / NLinear-style), so the
    # forecast stays bounded near the last observation instead of exploding.
    last_trend = V[-1] @ tcoef                       # (C,) trend deviation at u[-1]
    raw_dev = legvander(u_fut, order) @ tcoef - last_trend  # (H, C) linear growth
    h = np.arange(1, H + 1)
    sat = (1.0 - DAMP ** h) / (1.0 - DAMP)           # saturating cumulative profile
    scale = (sat / h)[:, None]                       # (H,1) damping ratio (1 at h=1)
    trend_fut = anchor + raw_dev * scale             # (H, C) bounded trend
    if basis == "legendre":
        return trend_fut

    # --- trend_seasonal: Fourier seasonal on the detrended residual ---------- #
    resid = look - trend_in                          # (L, C)
    F = np.fft.rfft(resid, axis=0)                    # (L//2+1, C)
    freqs = np.fft.rfftfreq(L)                        # cycles / sample
    mag = np.abs(F); mag[0] = 0.0                     # drop DC (already in trend)
    fut = np.arange(L, L + H)
    seasonal_fut = np.zeros((H, C))
    seasonal_last = np.zeros(C)
    power = (mag ** 2)                                # spectral energy per bin
    total = power[1:].sum(axis=0) + 1e-12             # residual energy per channel
    for c in range(C):
        # energy-fraction gate: continue a harmonic only if it is a *dominant*
        # spectral peak, holding a large share of the residual energy. A period
        # pinned from a 96-point lookback drifts out of phase when extrapolated
        # over long horizons, so a weak/uncertain peak hurts; this keeps only the
        # clean, strong cycles (so the seasonal term is help-or-neutral and zero
        # on aperiodic channels, where the forecast falls back to the trend).
        frac = power[1:, c] / total[c]                # energy share per non-DC bin
        cand = 1 + np.where(frac > SEASONAL_FRAC)[0]
        if cand.size == 0:
            continue
        keep = cand[np.argsort(mag[cand, c])[-n_harm:]]
        for k in keep:
            amp = 2.0 * np.abs(F[k, c]) / L
            ph = np.angle(F[k, c])
            seasonal_fut[:, c] += amp * np.cos(2 * np.pi * freqs[k] * fut + ph)
            seasonal_last[c] += amp * np.cos(2 * np.pi * freqs[k] * (L - 1) + ph)
    # anchor the composite to the last observation (continuity); the double
    # anchor cancels algebraically, leaving forecast = last obs + dtrend + dseasonal
    return trend_fut + seasonal_fut + (anchor[0] - trend_in[-1] - seasonal_last)


def evaluate(name, H, order, basis, max_windows, n_harm=3):
    """Mean (MSE, MAE) of a forecaster over the test windows of one dataset.

    Reproduces the LTSF protocol: z-score (train-fit) normalized windows, MSE/MAE
    on the normalized values, averaged across ``max_windows`` test windows.
    """
    se = ae = cnt = 0
    for look, target in ds.test_windows(name, LOOKBACK, H, max_windows=max_windows):
        pred = series_extrapolate(look, H, order, basis, n_harm)
        se += float(np.mean((pred - target) ** 2))
        ae += float(np.mean(np.abs(pred - target)))
        cnt += 1
    return (se / cnt, ae / cnt) if cnt else (np.nan, np.nan)


def run_dataset(name, horizons, order, max_windows, n_harm=3):
    """Measure both dtfit forecasters on one dataset across ``horizons``.

    Returns ``(trend_mse, ts_mse, ts_mae)`` -- three ``{horizon: value}`` dicts
    for the trend-only MSE, the trend+seasonal MSE, and the trend+seasonal MAE.
    """
    trend_mse, ts_mse, ts_mae = {}, {}, {}
    for H in horizons:
        mse_tr, _ = evaluate(name, H, order, "legendre", max_windows)
        mse_ts, mae_ts = evaluate(name, H, order, "trend_seasonal", max_windows, n_harm)
        trend_mse[H] = mse_tr
        ts_mse[H] = mse_ts
        ts_mae[H] = mae_ts
    return trend_mse, ts_mse, ts_mae


def seasonal_helped(trend_mse, ts_mse, h0):
    """Split datasets by whether the seasonal term helped at horizon ``h0``.

    ``trend_mse`` / ``ts_mse`` are ``{name: {horizon: mse}}``. Returns
    ``(helped, hurt)`` lists of dataset names where trend+seasonal beat (or did
    not beat) trend-only at the shortest horizon.
    """
    helped, hurt = [], []
    for name in ts_mse:
        tr, ts = trend_mse[name].get(h0), ts_mse[name].get(h0)
        if tr and ts and np.isfinite(tr) and np.isfinite(ts) and tr > 0:
            (helped if ts < tr else hurt).append(name)
    return helped, hurt


def sample_window(name, H, order, n_harm=3):
    """One sample lookback/target window plus both forecasts, for plotting.

    Returns ``(look, target, pred_trend, pred_trend_seasonal)`` for the first
    test window of ``name`` at horizon ``H`` (all ``(*, C)`` arrays).
    """
    wins = list(ds.test_windows(name, LOOKBACK, H, max_windows=1))
    look, target = wins[0]
    pred_tr = series_extrapolate(look, H, order, "legendre")
    pred_ts = series_extrapolate(look, H, order, "trend_seasonal", n_harm)
    return look, target, pred_tr, pred_ts
