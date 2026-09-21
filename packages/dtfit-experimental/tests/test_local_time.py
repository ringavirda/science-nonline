"""LocalTimeFilter: the exact origin shift, invariance to the clock's zero,
agreement with the absolute-time filter where that one is well conditioned,
process noise per unit of time, and the guards."""

import warnings

import numpy as np
import pytest

from dtfit import ImageFilter
from dtfit_experimental import LocalTimeFilter, shift_matrix

OPTIONS = dict(order=3, window_size=12, drift_reset="inflate")


def _track(n=120, dt=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = dt * np.arange(n)
    y = 5.0 + 3.0 * np.sin(0.4 * t) + 0.3 * t + rng.normal(0.0, 0.2, n)
    return t, y


def _run(flt, t, y):
    out = np.empty(t.size)
    for i in range(t.size):
        flt.partial_fit(t[i], y[i])
        out[i] = float(flt.predict(np.array([t[i] + 1.0]))[0])
    return out


def test_shift_matrix_moves_the_origin_exactly():
    """Fails when the binomial factor is dropped from T[i, j]."""
    c = np.array([1.5, -2.0, 0.7, 0.25])
    d = 1.7
    s = np.linspace(-2.0, 2.0, 9)
    shifted = shift_matrix(3, d) @ c
    assert np.allclose(np.polyval(shifted[::-1], s),
                       np.polyval(c[::-1], s + d), rtol=0, atol=1e-12)
    assert np.allclose(shift_matrix(3, -d) @ shifted, c, rtol=0, atol=1e-12)


def test_clock_origin_does_not_move_the_forecast():
    """Fails when the buffered times stay absolute instead of moving with
    the origin."""
    t, y = _track()
    near = _run(LocalTimeFilter(2, p0=[y[0], 0, 0], **OPTIONS), t, y)
    far = _run(LocalTimeFilter(2, p0=[y[0], 0, 0], **OPTIONS), t + 1.0e4, y)
    assert np.allclose(near, far, rtol=0, atol=1e-6)


def test_agrees_with_the_absolute_filter_without_process_noise():
    """Fails when P is left untransported (only p moved to the new origin):
    with no process noise and a clock starting at zero the two filters are
    one estimator in two coordinate systems."""
    t, y = _track(n=60)
    local = _run(LocalTimeFilter(2, q_rate=0.0, p0=[y[0], 0, 0], **OPTIONS),
                 t, y)
    absolute = _run(ImageFilter("c0 + c1*tt + c2*tt**2", "tt",
                                q_diag=[0.0] * 3, p0=[y[0], 0, 0], **OPTIONS),
                    t, y)
    assert np.allclose(local, absolute, rtol=0, atol=1e-6)


@pytest.mark.parametrize("gap", [0.5, 2.0])
def test_process_noise_grows_with_the_gap(gap):
    """Fails when q_rate is added once per update instead of per unit of
    time between samples."""
    flt = LocalTimeFilter(2, q_rate=[0.1, 0.2, 0.3], **OPTIONS)
    flt.partial_fit(100.0, 1.0)
    P0 = flt.P.copy()
    flt.partial_fit(100.0 + gap, 1.0)
    T = shift_matrix(2, gap)
    expected = T @ P0 @ T.T + np.diag([0.1, 0.2, 0.3]) * gap
    assert np.allclose(flt.P, expected, rtol=0, atol=1e-12)


def test_noiseless_quadratic_is_followed_far_from_the_clock_zero():
    """Fails when the buffered times are shifted forward instead of back."""
    t = 5.0e3 + 0.5 * np.arange(40)
    s = t - t[0]
    y = 3.0 + 2.0 * s - 0.05 * s ** 2
    flt = LocalTimeFilter(2, p0=[y[0], 0, 0], **OPTIONS)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    ahead = t[-1] + 1.0 - t[0]
    assert float(flt.predict(np.array([t[-1] + 1.0]))[0]) == pytest.approx(
        3.0 + 2.0 * ahead - 0.05 * ahead ** 2, abs=1e-5)
    assert flt.t_ref_ == t[-1]


def test_non_finite_sample_moves_nothing():
    """Fails when the state is transported before the sample is checked."""
    t, y = _track(n=30)
    flt = LocalTimeFilter(2, p0=[y[0], 0, 0], **OPTIONS)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    p, P, ref = flt.p.copy(), flt.P.copy(), flt.t_ref_
    with pytest.warns(RuntimeWarning):
        flt.partial_fit(t[-1] + 3.0, np.nan)
    with pytest.warns(RuntimeWarning):
        flt.partial_fit(np.nan, 1.0)
    assert np.array_equal(flt.p, p) and np.array_equal(flt.P, P)
    assert flt.t_ref_ == ref


@pytest.mark.parametrize("degree", [0, 10])
def test_degree_outside_the_named_coefficients_is_refused(degree):
    with pytest.raises(ValueError, match="degree"):
        LocalTimeFilter(degree)


@pytest.mark.parametrize("option", ["q_diag", "regressors", "stream",
                                    "param_names"])
def test_options_the_local_clock_cannot_honour_are_refused(option):
    with pytest.raises(ValueError, match=option):
        LocalTimeFilter(2, **{option: None})


@pytest.mark.parametrize("q_rate", [-0.1, [0.1, 0.1]])
def test_q_rate_must_be_non_negative_and_sized(q_rate):
    with pytest.raises(ValueError, match="q_rate"):
        LocalTimeFilter(2, q_rate=q_rate)


def test_samples_out_of_time_order_are_refused():
    flt = LocalTimeFilter(2)
    flt.partial_fit(10.0, 1.0)
    with pytest.raises(ValueError, match="time order"):
        flt.partial_fit(9.5, 1.0)


def test_regressors_are_refused_at_ingest():
    with pytest.raises(ValueError, match="regressors"):
        LocalTimeFilter(2).partial_fit(0.0, 1.0, regressors={"S": 0.0})


def test_forecast_before_the_first_sample_is_refused():
    with pytest.raises(RuntimeError, match="no sample"):
        LocalTimeFilter(2).predict(np.array([1.0]))


def test_update_is_the_local_ingest():
    """Fails when ``update`` is left bound to the absolute-time ingest."""
    flt = LocalTimeFilter(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flt.update(50.0, 1.0)
    assert flt.t_ref_ == 50.0
