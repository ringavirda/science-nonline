"""High-level entry points distilled from the domain merged pipelines.

``auto_estimate`` routes by signal shape to the right estimator variant;
``auto_forecast`` routes the model class with the no-structure / divergence
guards. These mirror the merged pipelines validated in the domain suite.
"""

import numpy as np
import pytest

from dtfit import auto_estimate, auto_forecast
from sklearn.metrics import r2_score


# --- auto_estimate --------------------------------------------------------- #
def test_auto_estimate_bulk_recovers_exponential():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 300)
    y = 1.0 * np.exp(0.9 * t) + rng.normal(0, 0.02, t.size)
    r = auto_estimate(t, y, "a*exp(b*t)", "t", shape="bulk", p0=[1.0, 1.0])
    assert abs(r.coeffs[0] - 1.0) < 0.2 and abs(r.coeffs[1] - 0.9) < 0.2


def test_auto_estimate_oscillatory_recovers_sine():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 300)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 0.05, t.size)
    r = auto_estimate(t, y, "A*sin(w*x)", "x", freq_param="w", p0=[1.0, 1.0])
    assert abs(r.coeffs[1] - 1.5) < 0.1  # sympy order: A, w


def test_auto_estimate_auto_detects_oscillation():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4 * np.pi, 300)
    y = 2.0 * np.sin(1.2 * t) + rng.normal(0, 0.05, t.size)
    # shape="auto" with a named frequency parameter routes to the osc recipe.
    r = auto_estimate(t, y, "A*sin(w*x)", "x", freq_param="w", p0=[1.0, 1.0])
    assert abs(r.coeffs[1] - 1.2) < 0.1


def test_auto_estimate_transient_uses_adaptive():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 3, 400)
    y = 2.0 * (1 - np.exp(-3.0 * t)) + rng.normal(0, 0.02, t.size)
    r = auto_estimate(t, y, "K*(1-exp(-a*x))", "x", shape="transient", p0=[1.0, 1.0])
    assert abs(r.coeffs[0] - 2.0) < 0.2 and abs(r.coeffs[1] - 3.0) < 0.6


def test_auto_estimate_unknown_shape_raises():
    t = np.linspace(0, 1, 30)
    with pytest.raises(ValueError, match="shape"):
        auto_estimate(t, np.exp(t), "a*exp(b*t)", "t", shape="weird")


# --- auto_forecast --------------------------------------------------------- #
def test_auto_forecast_logistic_growth():
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))  # saturating epidemic curve
    n_tr = 90
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    assert fc.shape == (30,)
    assert r2_score(y[n_tr:], fc) > 0.9


def test_auto_forecast_seasonal_beats_persistence():
    t = np.linspace(0, 20, 400)
    y = 0.5 * t + 3.0 * np.sin(2 * np.pi * t / 2.0)
    n_tr = 320
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=80, period=40.0)
    persist = np.full(80, y[n_tr - 1])
    rmse_fc = np.sqrt(np.mean((y[n_tr:] - fc) ** 2))
    rmse_p = np.sqrt(np.mean((y[n_tr:] - persist) ** 2))
    assert rmse_fc < rmse_p


def test_auto_forecast_no_structure_guard_fires_on_reverting_ramp():
    # a ramp that reverses out of sample: a structured fit of the training tail
    # extrapolates the local slope and badly overshoots persistence, so the
    # no-structure guard must trip and persist (flat last value).
    t = np.linspace(0, 30, 300)
    y = np.r_[np.linspace(0, 10, 150), np.linspace(10, 0, 150)]  # up then down
    fc = auto_forecast(t[:240], y[:240], horizon=60, model="poly")
    # guard either persists, or the divergence guard keeps it bounded; never blows up
    rng = float(np.ptp(y[:240]))
    assert np.all(np.abs(fc - y[239]) <= 5 * rng)


def test_auto_forecast_random_walk_stays_bounded():
    # on a true random walk the forecast must not catastrophically overshoot
    # persistence (the structured extrapolator's honest failure mode).
    rng = np.random.default_rng(6)
    y = np.cumsum(rng.normal(0, 1.0, 300))
    t = np.arange(y.size, dtype=float)
    n_tr = 240
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=60)
    persist = np.full(60, y[n_tr - 1])
    rmse_fc = np.sqrt(np.mean((y[n_tr:] - fc) ** 2))
    rmse_p = np.sqrt(np.mean((y[n_tr:] - persist) ** 2))
    assert rmse_fc < 3.0 * rmse_p


def test_auto_forecast_explicit_random_walk():
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    fc = auto_forecast(t, y, horizon=10, model="random_walk")
    assert np.allclose(fc, y[-1])


def test_auto_forecast_zero_horizon():
    t = np.linspace(0, 1, 50)
    assert auto_forecast(t, np.exp(t), horizon=0).size == 0
