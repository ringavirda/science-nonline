"""Streaming EDAFilter (online EDA with NIS drift detection)."""

import numpy as np

from dtfit import EDAFilter, LSIFilter


def test_filter_tracks_stable_sine():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    flt = EDAFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001], r=20.0,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)

    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_params_and_predict_shapes():
    flt = EDAFilter("A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=10)
    assert set(flt.params_) == {"A", "w"}
    out = flt.predict(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)


def test_partial_fit_returns_self():
    flt = EDAFilter("A*sin(w*t)", "t")
    assert flt.partial_fit(0.0, 0.0) is flt


def _feed_step(low: float, high: float, n: int = 120):
    """Run the filter over a clean level step and return (filter, direction)."""
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(n, low), np.full(n, high)] + rng.normal(0, 0.02, 2 * n)
    x = np.linspace(0, 1.0, levels.size)
    flt = EDAFilter(
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
    flt = EDAFilter(
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
    flt = EDAFilter(
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
    flt = EDAFilter(
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


# --------------------------------------------------------------------------- #
# LSIFilter -- streaming LSI (online integral least-squares)
# --------------------------------------------------------------------------- #
def test_lsi_filter_tracks_stable_sine():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    flt = LSIFilter(
        "A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=50, order=5,
        q_diag=[1e-3, 5e-4], r=5.0,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)

    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_lsi_filter_recovers_exponential():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6, 600)
    y = 2.5 * np.exp(-0.6 * t) + rng.normal(0, 0.05, t.size)
    flt = LSIFilter(
        "a*exp(b*t)", "t", p0=[1.0, -0.2], window_size=50, order=5,
        q_diag=[1e-4, 1e-4], r=0.5,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    p = flt.params_
    assert abs(p["a"] - 2.5) < 0.3
    assert abs(p["b"] + 0.6) < 0.15


def test_lsi_filter_params_and_predict_shapes():
    flt = LSIFilter("A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=10)
    assert set(flt.params_) == {"A", "w"}
    out = flt.predict(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)


def test_lsi_filter_partial_fit_returns_self():
    flt = LSIFilter("A*sin(w*t)", "t")
    assert flt.partial_fit(0.0, 0.0) is flt


def test_lsi_filter_no_false_drift_on_stable_signal():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
    flt = LSIFilter(
        "A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=50, order=5,
        q_diag=[1e-3, 5e-4], r=5.0,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert flt.n_drifts_ == 0  # a stationary signal must not trip the detector


def test_lsi_filter_detects_level_step():
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(150, 1.0), np.full(150, 3.0)] + rng.normal(0, 0.02, 300)
    x = np.linspace(0, 1.5, levels.size)
    flt = LSIFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20, order=5,
        q_diag=[1e-3, 1e-3], r=0.2,
    )
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    assert flt.n_drifts_ >= 1
    assert direction == 1  # upward level shift


def test_filter_survives_exp_overflow_without_nan_poisoning():
    """An unbounded model whose rate wanders cannot permanently NaN-poison the
    filter: predict() must stay finite even if a transient overflow occurs.

    Regression for the GPS experiment, where a climb model `z0+c*(1-exp(-k*t))`
    drove the time-constant toward its singular value and turned every later
    prediction into NaN.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 12, 400)
    y = 5.0 + 4.0 * (1.0 - np.exp(-t / 3.0)) + rng.normal(0, 0.3, t.size)
    flt = EDAFilter(
        "z0 + c*(1-exp(-k*t))", "t", p0=[3.0, 0.3, 4.0], window_size=40,
        q_diag=[1e-3, 1e-3, 1e-3], r=0.5, n_sub=2, adapt_r=True,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
        assert np.all(np.isfinite(flt.p)), "filter parameters went non-finite"
    pred = flt.predict(t)
    assert np.all(np.isfinite(pred)), "predict() returned NaN/inf"


def test_lsi_filter_survives_nonfinite_update():
    """The Legendre filter shares the same non-finite-update guard."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 12, 300)
    y = 5.0 + 4.0 * (1.0 - np.exp(-t / 3.0)) + rng.normal(0, 0.3, t.size)
    flt = LSIFilter(
        "z0 + c*(1-exp(-k*t))", "t", p0=[3.0, 0.3, 4.0], window_size=30,
        order=4, q_diag=[1e-3, 1e-3, 1e-3], r=0.5,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert np.all(np.isfinite(flt.p))
    assert np.all(np.isfinite(flt.predict(t)))


def test_last_residual_is_the_forecast_innovation():
    """``last_residual_`` is NaN before the window fills, then equals the
    one-step forecast error y_new - f(t_new; p) at the newest sample."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, 6, 200)
    y = 2.0 + 0.5 * t + 0.1 * t**2 + rng.normal(0, 0.2, t.size)
    flt = EDAFilter(
        "c0 + c1*t + c2*t**2", "t", p0=[0.0, 0.0, 0.0], window_size=15,
        q_diag=[1e-2, 1e-2, 1e-2], r=0.5, n_sub=2, adapt_r=True,
    )
    assert np.isnan(flt.last_residual_)  # nothing ingested yet
    seen_finite = False
    for ti, yi in zip(t, y):
        yhat = float(flt.predict(np.array([ti]))[0])  # pre-update prediction
        flt.partial_fit(ti, yi)
        if np.isfinite(flt.last_residual_):
            assert abs(flt.last_residual_ - (yi - yhat)) < 1e-9
            seen_finite = True
    assert seen_finite


def test_lsi_filter_exposes_last_residual():
    flt = LSIFilter(
        "c0 + c1*t", "t", p0=[0.0, 0.0], window_size=20, order=3,
        q_diag=[1e-2, 1e-2], r=0.5,
    )
    assert np.isnan(flt.last_residual_)
    t = np.linspace(0, 5, 120)
    for ti in t:
        flt.partial_fit(ti, 1.0 + 0.5 * ti)
    assert np.isfinite(flt.last_residual_)


def test_inflate_scales_covariance_for_both_filters():
    """``inflate`` multiplies the parameter covariance (the external-detector
    re-arming hook) -- both with an explicit factor and the configured default."""
    for cls, kw in [
        (EDAFilter, dict(window_size=15)),
        (LSIFilter, dict(window_size=20, order=3)),
    ]:
        flt = cls("c0 + c1*t", "t", p0=[0.0, 0.0],
                  q_diag=[1e-2, 1e-2], r=0.5, drift_inflation=50.0, **kw)
        p0 = flt.P.copy()
        flt.inflate(7.0)
        assert np.allclose(flt.P, p0 * 7.0)
        flt.inflate()  # defaults to drift_inflation
        assert np.allclose(flt.P, p0 * 7.0 * 50.0)
