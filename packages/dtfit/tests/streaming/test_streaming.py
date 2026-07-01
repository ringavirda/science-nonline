"""Streaming EACFilter (online EAC with NIS drift detection)."""

import numpy as np
import pytest

from dtfit import EACFilter, LSIFilter


def test_filter_tracks_stable_sine():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    flt = EACFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001], r=20.0,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)

    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_params_and_predict_shapes():
    flt = EACFilter("A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=10)
    assert set(flt.params_) == {"A", "w"}
    out = flt.predict(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)


def test_partial_fit_returns_self():
    flt = EACFilter("A*sin(w*t)", "t")
    assert flt.partial_fit(0.0, 0.0) is flt


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_running_param_uncertainty_contracts(cls):
    """Both filters expose a running covariance/std that shrinks as the
    parameters become identified (the streaming twin of FittingResult.stderr)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 1500)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
    flt = cls("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50)

    flt.partial_fit(t[0], y[0])
    early = float(np.trace(flt.param_cov_))
    for ti, yi in zip(t[1:], y[1:]):
        flt.partial_fit(ti, yi)

    assert flt.param_cov_.shape == (2, 2)
    assert set(flt.stderr_) == {"A", "w"}
    assert all(v >= 0 for v in flt.stderr_.values())
    # the posterior covariance has contracted from its large initial value
    assert float(np.trace(flt.param_cov_)) < early


def _feed_step(low: float, high: float, n: int = 120):
    """Run the filter over a clean level step and return (filter, direction)."""
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(n, low), np.full(n, high)] + rng.normal(0, 0.02, 2 * n)
    x = np.linspace(0, 1.0, levels.size)
    flt = EACFilter(
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


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_coast_matches_predict_in_support(cls):
    """coast() reduces to predict() at and before the anchor (in-window)."""
    rng = np.random.default_rng(0)
    t = np.arange(40) * 0.1
    y = 2.0 + 3.0 * t + 0.5 * t**2 + rng.normal(0, 0.05, t.size)
    flt = cls("c0 + c1*t + c2*t**2", "t", p0=[y[0], 0.0, 0.0], window_size=15)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    xin = t[-5:]  # all at or before the anchor
    assert np.allclose(flt.coast(xin), flt.predict(xin))


def test_coast_stays_bounded_where_cubic_diverges():
    """Past the window a fitted cubic's predict() runs away; the order-1 coast
    (constant velocity) stays close to a straight-line extrapolation."""
    rng = np.random.default_rng(1)
    t = np.arange(40) * 0.1
    y = 2.0 + 3.0 * t - 0.1 * t**3 + rng.normal(0, 0.05, t.size)
    flt = LSIFilter("c0 + c1*t + c2*t**2 + c3*t**3", "t",
                    p0=[y[0], 0.0, 0.0, 0.0], window_size=15, order=5)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    a = t[-1]
    gap = a + np.arange(1, 21) * 0.1  # 2 s past support
    c1 = flt.coast(gap, order=1)
    assert np.all(np.isfinite(c1))
    # an order-1 coast is exactly linear in (x - a): its second difference is ~0,
    # whereas the raw cubic predict() curves away
    assert np.allclose(np.diff(c1, 2), 0.0, atol=1e-9)
    assert not np.allclose(np.diff(flt.predict(gap), 2), 0.0, atol=1e-9)


def test_coast_rejects_regressor_models():
    flt = LSIFilter("c0 + c1*t + S", "t", regressors="S")
    with pytest.raises(NotImplementedError):
        flt.coast(np.array([1.0]))


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_predict_cov_is_nonneg_shaped_and_contracts(cls):
    """predict_cov maps the parameter covariance into output space: non-negative,
    shaped like x, and it contracts as the estimate becomes identified."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 800)
    y = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)
    flt = cls("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=40, q_diag=[1e-4, 1e-4])
    flt.partial_fit(t[0], y[0])
    early = float(flt.predict_cov(np.array([10.0]))[0])
    for ti, yi in zip(t[1:], y[1:]):
        flt.partial_fit(ti, yi)
    v = flt.predict_cov(np.array([5.0, 10.0, 15.0]))
    assert v.shape == (3,)
    assert np.all(v >= 0.0)
    assert float(flt.predict_cov(np.array([10.0]))[0]) < early   # uncertainty shrank
    # one-sigma band is finite and usable alongside predict()
    band = np.sqrt(flt.predict_cov(np.array([10.0])))
    assert np.all(np.isfinite(band))


def test_no_false_drift_on_stable_signal():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
    flt = EACFilter(
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
    flt = EACFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001], r=20.0, n_sub=4, adapt_r=True,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_subareas_detect_amplitude_jump_on_oscillation():
    # On an oscillation an amplitude jump nets to little signed area, so a single
    # scalar area (n_sub=1) detects it unreliably. Splitting the window into
    # several sub-areas gives the energy NIS a chi^2(n_sub) statistic with several
    # independent channels, which catches the jump with no false alarms.
    n = 900
    t = np.linspace(0, 40, n)
    half = n // 2
    amp = np.where(np.arange(n) < half, 2.0, 3.5)
    y = amp * np.sin(1.8 * t) + np.random.default_rng(3).normal(0, 0.2, n)

    def detect(n_sub):
        flt = EACFilter(
            "A*sin(w*t)", "t", p0=[2.0, 1.8], window_size=60, n_sub=n_sub,
            adapt_r=True, q_diag=[3e-3, 1e-4], drift_reset="inflate",
        )
        first, false = None, 0
        for i in range(n):
            flt.partial_fit(t[i], y[i])
            if flt.drift_flag_:
                if i < half:
                    false += 1
                elif first is None:
                    first = i
        return first, false

    first, false = detect(n_sub=5)
    assert first is not None        # the vector detector catches the jump
    assert first - half < 60        # within one window stride of the shift
    assert false == 0               # and raises no false alarm before it


def test_inflate_drift_reset_detects_step_and_keeps_window():
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(120, 1.0), np.full(120, 3.0)] + rng.normal(0, 0.02, 240)
    x = np.linspace(0, 1.0, levels.size)
    flt = EACFilter(
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
    flt = EACFilter(
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
    flt = EACFilter(
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


def test_accumulative_window_acquires_before_full():
    """The window is accumulative: both filters produce a usable estimate *before*
    window_size samples have arrived (no full-window dead time), and the estimate
    only improves once the window is full."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 24, 1200)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.05 * 3.0, t.size)
    for cls, kw in [(EACFilter, dict(window_size=60, n_sub=2)),
                    (LSIFilter, dict(window_size=60, order=5))]:
        flt = cls("A*sin(w*t)", "t", p0=[1.0, 1.0], q_diag=[1e-3, 1e-3],
                  r=0.5, adapt_r=True, **kw)
        assert 0 < flt.min_window < flt.W      # starts well before a full window
        mid = None
        for i, (ti, yi) in enumerate(zip(t, y)):
            flt.partial_fit(ti, yi)
            if i == flt.W - 2:                 # one step before the window first fills
                mid = abs(flt.params_["A"] - 3.0) / 3.0
        # an estimate exists and is already acquiring before the window is full ...
        assert mid is not None and mid < 0.5
        # ... and it converges by the end.
        assert abs(flt.params_["A"] - 3.0) < 0.3


def test_adaptive_window_auto_sizes_to_the_model():
    """The adaptive window sizes itself from the data: a polynomial (global
    parameters) grows a far wider window than an oscillation (locally observable),
    and both stay accurate -- with no per-model tuning."""
    def run(expr, p0, true, T, n, seed):
        ts = np.linspace(0, T, n)
        import sympy as sp
        sym = sp.Symbol("t")
        mdl = sp.sympify(expr)
        ps = sorted((s for s in mdl.free_symbols if s != sym), key=str)
        f = sp.lambdify([sym, *ps], mdl, "numpy")
        clean = f(ts, *[true[str(s)] for s in ps])
        y = clean + np.random.default_rng(seed).normal(
            0, 0.05 * (clean.std() + 1e-9), n)
        flt = LSIFilter(expr, "t", p0=p0, window_size=300, adaptive_window=True,
                        order=5, q_diag=[1e-4] * len(ps), r=0.5, adapt_r=True)
        for ti, yi in zip(ts, y):
            flt.partial_fit(ti, yi)
        err = np.mean([abs(flt.params_[str(s)] - true[str(s)]) /
                       abs(true[str(s)]) for s in ps]) * 100
        return flt._W_eff, err

    w_osc, e_osc = run("A*sin(w*t)", [1.0, 1.0], {"A": 3.0, "w": 1.5}, 24, 1200, 0)
    w_poly, e_poly = run("c0+c1*t+c2*t**2", [0.0, 0.0, 0.0],
                         {"c0": 1.0, "c1": 2.0, "c2": 0.5}, 6, 600, 0)
    assert w_poly > 2 * w_osc        # the polynomial needs a much wider window
    assert e_osc < 3.0 and e_poly < 6.0   # both identified well, no tuning


def test_adaptive_window_collapses_and_regrows_on_drift():
    """On a regime change the adaptive window collapses back to min_window (to
    flush stale old-regime data) and then re-grows as the new regime is
    identified."""
    rng = np.random.default_rng(1)
    n = 900
    t = np.linspace(0, 40, n)
    half = n // 2
    amp = np.where(np.arange(n) < half, 2.0, 3.5)
    y = amp * np.sin(1.8 * t) + rng.normal(0, 0.2, n)
    flt = LSIFilter("A*sin(w*t)", "t", p0=[2.0, 1.8], window_size=300,
                    adaptive_window=True, order=6, q_diag=[3e-3, 1e-4],
                    drift_reset="inflate")
    W = np.empty(n)
    for i in range(n):
        flt.partial_fit(t[i], y[i])
        W[i] = flt._W_eff
    assert flt.n_drifts_ >= 1                       # the regime change is detected
    assert W[:half].max() > 3 * flt.min_window      # grew wide before the change
    post_min = W[half:half + 60].min()
    assert post_min <= flt.min_window + 2           # collapsed to ~min_window after
    assert W[-1] > post_min + 5                      # then re-grew past the collapse
    assert abs(flt.params_["A"] - 3.5) < 0.4         # re-acquired the new regime


def test_adaptive_window_shrinks_on_maneuver():
    """The window auto-SHRINKS when the model lags time-varying dynamics (its
    forecast residual becomes systematically same-sign), while a static fit keeps a
    wide window -- so a maneuvering signal is tracked with a shorter, more
    responsive window than the same model gets on a static one, no hand-tuning."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 1500)
    # static: the line model matches the data exactly -> white residuals -> wide window
    y_static = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)
    fs = LSIFilter("c0 + c1*t", "t", p0=[2.0, 0.5], window_size=120,
                   adaptive_window=True, order=3, q_diag=[1e-3, 1e-3], adapt_noise=True)
    for ti, yi in zip(t, y_static):
        fs.partial_fit(ti, yi)
    # maneuvering: the same line model must chase a curving signal -> it lags, the
    # residual autocorrelates (runs of one sign) -> the window shrinks
    y_man = 3.0 * np.sin(0.4 * t) + rng.normal(0, 0.05, t.size)
    fm = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=120,
                   adaptive_window=True, order=3, q_diag=[1e-3, 1e-3], adapt_noise=True)
    for ti, yi in zip(t, y_man):
        fm.partial_fit(ti, yi)
    assert fm._W_eff < fs._W_eff // 2          # maneuver -> much shorter than static
    assert fm._resid_corr > fs._resid_corr      # driven by residual autocorrelation
    assert fs._W_eff > 3 * fs.min_window        # static still grows wide (unchanged)


def test_eac_adaptive_window_is_stable_and_finite():
    """The area filter's adaptive window uses a covariance-reduction criterion: it
    must settle a *finite* window on an oscillation (never run away to the cap) and
    stay accurate -- unlike the estimate-movement test, which is unstable for the
    scalar area measurement."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 12, 700)
    clean = 2.0 * np.sin(2.5 * t)
    y = clean + rng.normal(0, 0.05 * clean.std(), t.size)
    flt = EACFilter("A*sin(w*t)", "t", p0=[1.5, 2.0], window_size=300,
                    adaptive_window=True, n_sub=2, q_diag=[1e-4, 1e-4],
                    r=0.5, adapt_r=True)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert flt.min_window < flt._W_eff < flt.W        # settled, did not run to cap
    assert abs(flt.params_["A"] - 2.0) / 2.0 < 0.1    # and stayed accurate


def test_min_window_is_respected_and_clamped():
    """``min_window`` controls when acquisition starts and is clamped sanely."""
    f = LSIFilter("A*sin(w*t)", "t", window_size=40, order=5, min_window=12)
    assert f.min_window == 12
    # below the floor (order+2) it is clamped up; above window_size, down.
    assert LSIFilter("A*sin(w*t)", "t", window_size=40, order=5,
                     min_window=1).min_window == 7        # order + 2
    assert LSIFilter("A*sin(w*t)", "t", window_size=40, order=5,
                     min_window=999).min_window == 40     # window_size
    # the area filter defaults to half the window (a scalar area needs support).
    assert EACFilter("A*sin(w*t)", "t", window_size=60, n_sub=2).min_window == 30


def test_inflate_scales_covariance_for_both_filters():
    """``inflate`` multiplies the parameter covariance (the external-detector
    re-arming hook) -- both with an explicit factor and the configured default."""
    for cls, kw in [
        (EACFilter, dict(window_size=15)),
        (LSIFilter, dict(window_size=20, order=3)),
    ]:
        flt = cls("c0 + c1*t", "t", p0=[0.0, 0.0],
                  q_diag=[1e-2, 1e-2], r=0.5, drift_inflation=50.0, **kw)
        p0 = flt.P.copy()
        flt.inflate(7.0)
        assert np.allclose(flt.P, p0 * 7.0)
        flt.inflate()  # defaults to drift_inflation
        assert np.allclose(flt.P, p0 * 7.0 * 50.0)


# --------------------------------------------------------------------------- #
# Robust mode -- in-window residual winsorization rejects gross outliers
# --------------------------------------------------------------------------- #
def _outlier_sine(seed, frac):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40, 1600)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.15, t.size)
    mask = rng.random(t.size) < frac
    y[mask] += rng.normal(0, 24.0, int(mask.sum()))  # gross spikes (~8x amplitude)
    return t, y


def test_robust_mode_resists_outliers_both_filters():
    """With 10% gross outliers the robust filter must stay far closer to the truth
    than the non-robust one, for both the area and the spectrum filter."""
    for cls, kw in [(EACFilter, dict(window_size=60, n_sub=2)),
                    (LSIFilter, dict(window_size=60, order=5))]:
        t, y = _outlier_sine(0, 0.10)

        def err(robust):
            flt = cls("A*sin(w*t)", "t", p0=[1.0, 1.0],
                      q_diag=[1e-3, 1e-3], adapt_r=True, robust=robust, **kw)
            for ti, yi in zip(t, y):
                flt.partial_fit(ti, yi)
            p = flt.params_
            return abs(p["A"] - 3.0) / 3.0 + abs(p["w"] - 1.5) / 1.5

        assert err(True) < 0.5 * err(False)   # robust at least halves the error
        assert err(True) < 0.20                # and lands close to the truth


def test_robust_mode_clean_signal_matches_default():
    """On a clean signal the robust gate is inactive, so it must not degrade the
    estimate relative to the non-robust filter."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40, 1600)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.15, t.size)
    out = {}
    for robust in (False, True):
        flt = LSIFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=60, order=5,
                        q_diag=[1e-3, 1e-3], robust=robust)
        for ti, yi in zip(t, y):
            flt.partial_fit(ti, yi)
        out[robust] = abs(flt.params_["A"] - 3.0)
    assert out[True] < 0.3
    assert out[True] <= out[False] + 0.1   # no meaningful degradation


def test_robust_mode_still_detects_drift():
    """Winsorizing the residual around its MEDIAN preserves a sustained shift, so a
    genuine regime change is still detected under robust mode."""
    rng = np.random.default_rng(0)
    levels = np.r_[np.full(120, 1.0), np.full(120, 3.0)] + rng.normal(0, 0.02, 240)
    x = np.linspace(0, 1.0, levels.size)
    flt = EACFilter("a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20,
                    q_diag=[1e-3, 1e-3], r=0.2, robust=True)
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    assert flt.n_drifts_ >= 1
    assert direction == 1


# --------------------------------------------------------------------------- #
# External-regressor support on LSI/EAC -- the model may depend on measured
# side-channels, not just t, while keeping the integral/spectral measurement.
# --------------------------------------------------------------------------- #
def test_external_regressor_recovers_and_improves_both_filters():
    """A model ``c0 + c1*t + Sx`` carries a measured basis ``Sx`` as an external
    regressor. Both filters recover (c0, c1) and the regressor-aided fit beats the
    raw measurement -- the richer model fuses through the integral update."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 500)
    Sx = 0.5 * t**2 * np.sin(0.3 * t)              # an arbitrary measured side-channel
    truth = 3.0 - 0.8 * t + Sx
    y = truth + rng.normal(0, 0.3, t.size)
    raw = float(np.sqrt(np.mean((y - truth) ** 2)))
    for cls, kw in [(LSIFilter, dict(order=4)),
                    (EACFilter, dict(n_sub=2, adapt_r=True))]:
        flt = cls("c0 + c1*t + Sx", "t", regressors="Sx", p0=[0.0, 0.0],
                  window_size=20, q_diag=[1e-3, 1e-3], r=0.5, **kw)
        sm = np.zeros_like(t)
        for i in range(t.size):
            flt.partial_fit(t[i], y[i], regressors={"Sx": Sx[i]})
            sm[i] = float(flt.predict(np.array([t[i]]), regressors={"Sx": Sx[i]})[0])
        p = flt.params_
        assert abs(p["c0"] - 3.0) < 0.4
        assert abs(p["c1"] + 0.8) < 0.1
        smoothing = float(np.sqrt(np.mean((sm[40:] - truth[40:]) ** 2)))
        assert smoothing < 0.5 * raw                  # the basis sharply cuts error


def test_external_regressor_accepts_sequence_and_missing_raises():
    flt = LSIFilter("a*u + b*v", "t", regressors=["u", "v"], p0=[1.0, 1.0],
                    window_size=10, order=3)
    flt.partial_fit(0.0, 1.0, regressors=[2.0, 3.0])  # positional sequence form
    with pytest.raises(ValueError):
        flt.partial_fit(0.1, 1.0)                     # regressors required but omitted


def test_external_regressor_name_clashing_with_sympy_singleton():
    """A regressor named ``S`` (a SymPy singleton) is still usable -- the filter
    binds regressor names to plain Symbols when parsing."""
    for cls in (LSIFilter, EACFilter):
        flt = cls("c0 + S", "t", regressors="S", p0=[0.0], window_size=10)
        assert "S" not in flt.params_           # S is the regressor, not a parameter
        for ti in np.linspace(0, 1, 25):
            flt.partial_fit(ti, 2.0 + ti, regressors={"S": ti})
        assert abs(flt.params_["c0"] - 2.0) < 0.2   # recovers the offset


def test_external_regressor_predict_needs_regressors():
    flt = EACFilter("a*u + b", "t", regressors="u", p0=[1.0, 0.0], window_size=8)
    for ti in np.linspace(0, 1, 20):
        flt.partial_fit(ti, 2.0 * ti, regressors={"u": ti})
    out = flt.predict(np.array([0.0, 1.0]), regressors={"u": np.array([0.0, 1.0])})
    assert out.shape == (2,)
    with pytest.raises(ValueError):
        flt.predict(np.array([0.0]))                  # no regressors supplied


def test_filter_presets_configure_and_track():
    # Presets are thin constructors over the full knob set; overrides win.
    t = np.linspace(0, 6, 300)
    y = 2.0 * np.sin(1.5 * t)
    f = LSIFilter.tracking("A*sin(w*x)", "x")
    assert f.adaptive_window is True
    for ti, yi in zip(t, y):
        f.partial_fit(ti, yi)
    assert abs(f.params_["A"] - 2.0) < 0.2

    g = EACFilter.robust("A*sin(w*x)", "x")
    assert g._robust is True and g.drift_reset == "inflate"

    # an explicit override beats the preset default
    h = LSIFilter.robust("A*sin(w*x)", "x", adapt_noise=False)
    assert h.adapt_noise is False
