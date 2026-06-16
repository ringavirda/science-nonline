"""Streaming EqualAreasFilter (online EDA with NIS drift detection)."""

import numpy as np

from dtfit import EqualAreasFilter


def test_filter_tracks_stable_sine():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    flt = EqualAreasFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001], r=20.0,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)

    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_params_and_predict_shapes():
    flt = EqualAreasFilter("A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=10)
    assert set(flt.params_) == {"A", "w"}
    out = flt.predict(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)


def test_partial_fit_returns_self():
    flt = EqualAreasFilter("A*sin(w*t)", "t")
    assert flt.partial_fit(0.0, 0.0) is flt


def _feed_step(low: float, high: float, n: int = 120):
    """Run the filter over a clean level step and return (filter, direction)."""
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(n, low), np.full(n, high)] + rng.normal(0, 0.02, 2 * n)
    x = np.linspace(0, 1.0, levels.size)
    flt = EqualAreasFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20, q_diag=[1e-3, 1e-3], r=0.2
    )
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    return flt, direction


def test_drift_detected_upward():
    flt, direction = _feed_step(1.0, 3.0)
    assert flt.n_drifts_ >= 1
    assert direction == 1  # upward shift


def test_drift_detected_downward():
    flt, direction = _feed_step(3.0, 1.0)
    assert flt.n_drifts_ >= 1
    assert direction == -1  # downward shift


def test_no_false_drift_on_stable_signal():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
    flt = EqualAreasFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50, q_diag=[0.05, 0.001], r=20.0
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert flt.n_drifts_ == 0  # a stationary signal must not trip the detector


def test_vector_measurement_tracks_with_adaptive_r():
    # n_sub>1 (vector measurement) paired with adapt_r still recovers the model.
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
    flt = EqualAreasFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001], r=20.0, n_sub=4, adapt_r=True,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_inflate_drift_reset_detects_step_and_keeps_window():
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(120, 1.0), np.full(120, 3.0)] + rng.normal(0, 0.02, 240)
    x = np.linspace(0, 1.0, levels.size)
    flt = EqualAreasFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20,
        q_diag=[1e-3, 1e-3], r=0.2, drift_reset="inflate",
    )
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    assert flt.n_drifts_ >= 1
    assert direction == 1
    # "inflate" keeps the sliding window populated (unlike "full" which clears).
    assert len(flt._t) > 0
