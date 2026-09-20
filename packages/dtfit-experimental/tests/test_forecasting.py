"""The forecasting model specs, their seeding, and the baseline toolkit.

Each test names the mutation of forecasting.py it fails under.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks, hilbert

from dtfit_experimental.study import forecasting as F


# Fails when the frequency seed is dropped (forcing _w0_from to a constant
# raises "expected non-empty vector for x": no oscillation survives to pick
# peaks from).
def test_fit_kind_damped_recovers_decay_rate():
    def decay_from_peaks(sig, t):
        idx, _ = find_peaks(np.abs(sig))
        idx = idx[2:-2]
        return -np.polyfit(t[idx], np.log(np.abs(sig[idx])), 1)[0]

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 600)
    amp, w, z = 2.0, 20.0, 0.15
    y = amp * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z**2) * t)
    y_noisy = y + rng.normal(0, 0.005, t.size)

    pred = F.fit_kind("damped", t, y_noisy, t)
    decay_true = decay_from_peaks(y, t)
    decay_pred = decay_from_peaks(pred, t)
    assert decay_pred == pytest.approx(decay_true, rel=0.02)


# Fails when _detect_chirp is bypassed (a fixed w0=1, k=0 instead of the
# data): measured, w0 and k then land 54 and 124 percent off.
def test_fit_kind_chirp_recovers_sweep():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 400)
    w0, k = 5.0, 8.0
    y = np.sin(w0 * t + k * t**2)
    y_noisy = y + rng.normal(0, 0.01, t.size)

    pred = F.fit_kind("chirp", t, y_noisy, t)
    phase_true = np.unwrap(np.angle(hilbert(y - y.mean())))
    phase_pred = np.unwrap(np.angle(hilbert(pred - pred.mean())))
    k_true, w0_true, _ = np.polyfit(t, phase_true, 2)
    k_pred, w0_pred, _ = np.polyfit(t, phase_pred, 2)
    assert w0_pred == pytest.approx(w0_true, rel=0.02)
    assert k_pred == pytest.approx(k_true, rel=0.02)


# Fails when the torch guard is removed: baseline_preds then raises
# whatever bl.lstm_forecast raises on a stand-in module instead of omitting
# the key.
def test_baseline_preds_omits_lstm_without_torch(monkeypatch):
    monkeypatch.setattr(F.notebook, "optional_import", lambda name: None)
    y = np.sin(np.linspace(0, 10, 60))
    out = F.baseline_preds(y, 5, dict(seasonal=False, period=None), quick=False)
    for key in ("random walk", "drift", "poly extrap", "ETS (Holt-Winters)",
                "Theta", "ARIMA", "MLP"):
        assert key in out
    assert "LSTM" not in out


# Fails when the torch guard is removed: with torch present, LSTM would be
# skipped even though optional_import returns a module.
def test_baseline_preds_includes_lstm_with_torch(monkeypatch):
    monkeypatch.setattr(F.notebook, "optional_import", lambda name: object())
    monkeypatch.setattr(
        F.bl, "lstm_forecast", lambda y_tr, h, lookback, epochs: np.zeros(h))
    y = np.sin(np.linspace(0, 10, 60))
    out = F.baseline_preds(y, 5, dict(seasonal=False, period=None), quick=False)
    assert "LSTM" in out


# _w0_from finds the daily angular frequency of a 24-sample cycle within 1
# percent. Fails when the FFT bin is taken off by one, which is a real trap
# for a fixed-integer-sample cycle: period 23 or 25 both give a visibly
# different w0.
def test_w0_from_finds_daily_cycle():
    t = np.arange(240, dtype=float)
    y = np.sin(2 * np.pi * t / 24.0)
    w0 = F._w0_from(y, t)
    assert w0 == pytest.approx(2 * np.pi / 24.0, rel=0.01)
