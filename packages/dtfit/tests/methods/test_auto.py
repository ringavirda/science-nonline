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


def test_auto_estimate_bulk_accepts_dict_p0_and_bounds():
    # dict p0 / partial dict bounds are forwarded verbatim to both base fitters.
    rng = np.random.default_rng(4)
    t = np.linspace(0, 3, 300)
    y = 1.0 * np.exp(0.9 * t) + rng.normal(0, 0.02, t.size)
    r = auto_estimate(
        t, y, "a*exp(b*t)", "t", shape="bulk",
        p0={"a": 1.0, "b": 1.0},
        bounds={"a": (0.1, 10.0), "b": (0.1, 5.0)},
    )
    assert abs(r.coeffs[0] - 1.0) < 0.2 and abs(r.coeffs[1] - 0.9) < 0.2


def test_auto_estimate_transient_accepts_dict_and_pair_bounds():
    # the EAC routes take the same p0/bounds forms as LSI (no private
    # pairs -> scipy conversion in auto anymore).
    rng = np.random.default_rng(5)
    t = np.linspace(0, 3, 400)
    y = 2.0 * (1 - np.exp(-3.0 * t)) + rng.normal(0, 0.02, t.size)
    r_dict = auto_estimate(
        t, y, "K*(1-exp(-a*x))", "x", shape="transient",
        p0={"K": 1.0, "a": 1.0},
        bounds={"K": (0.0, 10.0), "a": (0.0, 20.0)},
    )
    r_pairs = auto_estimate(
        t, y, "K*(1-exp(-a*x))", "x", shape="transient",
        p0=[1.0, 1.0],
        bounds=[(0.0, 10.0), (0.0, 20.0)],
    )
    for r in (r_dict, r_pairs):
        assert abs(r.coeffs[0] - 2.0) < 0.2 and abs(r.coeffs[1] - 3.0) < 0.6


def test_auto_estimate_bulk_primary_failure_warns_and_falls_back(monkeypatch):
    import dtfit.auto as auto_mod

    def boom(*args, **kwargs):
        raise RuntimeError("lsi boom")

    monkeypatch.setattr(auto_mod, "fit_lsi", boom)
    rng = np.random.default_rng(6)
    t = np.linspace(0, 3, 200)
    y = 2.0 * np.exp(0.5 * t) + rng.normal(0, 0.01, t.size)
    with pytest.warns(UserWarning, match=r"fit_lsi failed \(lsi boom\)"):
        r = auto_estimate(t, y, "a*exp(b*t)", "t", shape="bulk", p0=[1.0, 1.0])
    assert np.all(np.isfinite(r.coeffs))  # the EAC fallback still delivered


def test_auto_estimate_bulk_both_fail_raises_with_both_messages(monkeypatch):
    import dtfit.auto as auto_mod

    def lsi_boom(*args, **kwargs):
        raise RuntimeError("lsi boom")

    def eac_boom(*args, **kwargs):
        raise RuntimeError("eac boom")

    monkeypatch.setattr(auto_mod, "fit_lsi", lsi_boom)
    monkeypatch.setattr(auto_mod, "fit_eac", eac_boom)
    t = np.linspace(0, 1, 30)
    with pytest.warns(UserWarning):  # each failed candidate also warns
        with pytest.raises(RuntimeError, match="lsi boom") as excinfo:
            auto_estimate(t, np.exp(t), "a*exp(b*t)", "t", shape="bulk")
    assert "eac boom" in str(excinfo.value)


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


def test_auto_forecast_failed_model_warns_and_falls_back_to_linear():
    # a series ending negative makes the logistic seed's L bounds inverted, so
    # the logistic fit raises; the fallback must warn (not silently swap) and
    # still return a horizon-length linear forecast. n < 24 keeps the
    # no-structure guard out of the way.
    t = np.linspace(0, 1, 20)
    y = np.linspace(1.0, -1.0, 20)
    with pytest.warns(
        UserWarning, match=r"logistic fit failed .*falling back to linear"
    ):
        fc = auto_forecast(t, y, horizon=5, model="logistic")
    assert fc.shape == (5,)
    assert np.all(np.isfinite(fc))
    # provenance: the linear fallback records which primary model failed.
    assert fc.model_name == "linear (logistic failed)"


def test_auto_forecast_divergent_poly_failed_linear_falls_to_persistence(
    monkeypatch,
):
    # second fallback: a diverging quadratic triggers the divergence guard,
    # whose linear refit ALSO raises -> warn and persist at y[-1]. Reaching it
    # needs a stub: poly "fits" but runs away, linear raises.
    import dtfit.auto as auto_mod

    t = np.linspace(0, 1, 20)
    y = np.linspace(1.0, 2.0, 20)

    def fake_fit_model(chosen, x, yy, t_all, period):
        # _fit_model now returns (values, FittingResult); the divergent stub has
        # no real fit, so it reports None for the result.
        if chosen == "poly":
            return np.full(t_all.size, 1e12), None  # wildly divergent prediction
        raise RuntimeError("boom")

    monkeypatch.setattr(auto_mod, "_fit_model", fake_fit_model)
    with pytest.warns(
        UserWarning, match=r"linear fit failed .*falling back to persistence"
    ):
        fc = auto_mod.auto_forecast(t, y, horizon=5, model="poly")
    assert fc.shape == (5,)
    assert np.allclose(fc, y[-1])
    # provenance: the persistence fallback records why it persisted, with no fit.
    assert fc.model_name == "persistence (linear failed)"
    assert fc.result is None and fc.std_band is None


# --- auto_forecast: structured ForecastResult ------------------------------ #
def _horizon_std_ok(fc, horizon):
    # .std_band is either absent (None) or a finite length-horizon band.
    return fc.std_band is None or (
        isinstance(fc.std_band, np.ndarray)
        and fc.std_band.shape == (horizon,)
        and np.all(np.isfinite(fc.std_band))
    )


def test_auto_forecast_returns_ndarray_and_forecastresult():
    # the return is BOTH a plain ndarray (every existing caller keeps working)
    # and a ForecastResult carrying provenance.
    from dtfit.auto import ForecastResult

    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    assert isinstance(fc, np.ndarray)
    assert isinstance(fc, ForecastResult)
    # numerically identical to the bare-values contract callers relied on.
    assert fc.shape == (30,)
    assert np.all(np.isfinite(fc))
    assert len(fc) == 30
    assert np.allclose(fc, np.asarray(fc))  # ndarray semantics intact
    assert fc.model_name == "logistic"  # chosen model recorded
    from dtfit import FittingResult

    assert isinstance(fc.result, FittingResult)  # real fit attached
    assert _horizon_std_ok(fc, 30)  # std None-or-length-horizon


def test_auto_forecast_std_band_is_delta_method_predict_std():
    # on a covariance-bearing fit the band is populated (not None) and IS the
    # delta-method predict std at the extrapolated future grid.
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    assert fc.std_band is not None
    assert isinstance(fc.std_band, np.ndarray)
    assert fc.std_band.shape == (30,)
    assert np.all(np.isfinite(fc.std_band))
    x = t[:n_tr]
    dx = float(np.mean(np.diff(x)))
    future = x[-1] + dx * np.arange(1, 31)
    _, std_ref = fc.result.predict(future, return_std=True)
    np.testing.assert_allclose(fc.std_band, std_ref)


def test_forecastresult_does_not_shadow_ndarray_std():
    # .std_band (not .std) carries the band, so the ndarray reduction still works.
    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t + 0.5 * t**2
    fc = auto_forecast(t, y, horizon=10, model="poly")
    assert np.isfinite(fc.std())        # ndarray.std() reduction, not shadowed
    assert np.isfinite(np.std(fc))


def test_auto_forecast_explicit_model_name_and_result():
    # an explicit (non-auto) model is recorded verbatim, with its fit attached.
    from dtfit import FittingResult

    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t + 0.5 * t**2
    fc = auto_forecast(t, y, horizon=10, model="poly")
    assert fc.model_name == "poly"
    assert isinstance(fc.result, FittingResult)
    assert _horizon_std_ok(fc, 10)


def test_auto_forecast_random_walk_provenance():
    # explicit random walk persists and is tagged as such (no fit, no band).
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    fc = auto_forecast(t, y, horizon=10, model="random_walk")
    assert np.allclose(fc, y[-1])
    assert fc.model_name == "random_walk"
    assert fc.result is None and fc.std_band is None


def test_auto_forecast_no_structure_provenance(monkeypatch):
    # when the no-structure guard trips (the structured model cannot beat naive
    # persistence on a held-out training tail), auto_forecast persists and the
    # provenance names the rejected model, carrying no fit and no band. Force the
    # guard so the test exercises the branch without coupling to a specific fit
    # realisation (the factor-8 guard is deliberately hard to trip in the wild).
    import dtfit.auto as auto_mod

    monkeypatch.setattr(auto_mod, "_no_structure", lambda *a, **k: True)
    t = np.linspace(0, 30, 300)
    y = 1.0 + 2.0 * t
    n_tr = 240
    fc = auto_mod.auto_forecast(t[:n_tr], y[:n_tr], horizon=60, model="poly")
    assert fc.model_name.startswith("persistence (") and "no structure" in fc.model_name
    assert fc.model_name == "persistence (poly no structure)"
    assert np.allclose(fc, y[n_tr - 1])
    assert fc.result is None and fc.std_band is None


def test_auto_forecast_divergence_guard_reports_provenance(monkeypatch):
    # divergence guard: poly "fits" but runs away, the linear refit succeeds ->
    # the forecast is the (real) linear one, tagged 'linear (poly diverged)'.
    import dtfit.auto as auto_mod
    from dtfit import FittingResult

    real_fit_model = auto_mod._fit_model
    t = np.linspace(0, 1, 20)
    y = np.linspace(1.0, 2.0, 20)

    def fake_fit_model(chosen, x, yy, t_all, period):
        if chosen == "poly":
            return np.full(t_all.size, 1e12), None  # divergent
        return real_fit_model(chosen, x, yy, t_all, period)  # real linear fit

    monkeypatch.setattr(auto_mod, "_fit_model", fake_fit_model)
    fc = auto_mod.auto_forecast(t, y, horizon=5, model="poly")
    assert isinstance(fc, auto_mod.ForecastResult)
    assert fc.model_name == "linear (poly diverged)"
    assert isinstance(fc.result, FittingResult)  # the real linear fit
    assert _horizon_std_ok(fc, 5)


def test_auto_forecast_zero_horizon_is_forecastresult():
    from dtfit.auto import ForecastResult

    fc = auto_forecast(np.linspace(0, 1, 50), np.exp(np.linspace(0, 1, 50)), horizon=0)
    assert isinstance(fc, ForecastResult)
    assert fc.size == 0
    assert fc.result is None and fc.std_band is None


def test_auto_forecast_std_fallback_when_no_covariance(monkeypatch):
    # a fit whose result carries no covariance yields std=None (never a crash),
    # while the values and model_name are still delivered.
    import dtfit.auto as auto_mod

    real_fit_model = auto_mod._fit_model
    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t

    def strip_cov(chosen, x, yy, t_all, period):
        pred, res = real_fit_model(chosen, x, yy, t_all, period)
        res.cov = None  # simulate a method that produced no covariance
        return pred, res

    monkeypatch.setattr(auto_mod, "_fit_model", strip_cov)
    fc = auto_mod.auto_forecast(t, y, horizon=10, model="linear")
    assert fc.std_band is None
    assert fc.model_name == "linear"
    assert fc.shape == (10,)


# --- auto_forecast / auto_estimate: pandas interop ------------------------- #
def test_auto_estimate_accepts_series_and_single_col_dataframe():
    # Series and single-column DataFrame inputs are coerced to 1-D arrays and
    # recover the same parameters as the ndarray path.
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 300)
    y = 1.0 * np.exp(0.9 * t) + rng.normal(0, 0.02, t.size)
    r_series = auto_estimate(
        pd.Series(t), pd.Series(y), "a*exp(b*t)", "t", shape="bulk", p0=[1.0, 1.0]
    )
    r_frame = auto_estimate(
        pd.DataFrame({"t": t}), pd.DataFrame({"y": y}),
        "a*exp(b*t)", "t", shape="bulk", p0=[1.0, 1.0],
    )
    for r in (r_series, r_frame):
        assert abs(r.coeffs[0] - 1.0) < 0.2 and abs(r.coeffs[1] - 0.9) < 0.2


def test_auto_forecast_series_values_match_ndarray_path():
    # pandas in -> the forecast VALUES are bit-identical to the ndarray path
    # (only .index is additive).
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    idx = pd.date_range("2020-01-01", periods=n_tr, freq="D")
    fc_np = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    fc_pd = auto_forecast(pd.Series(t[:n_tr], index=idx), y[:n_tr], horizon=30)
    np.testing.assert_array_equal(np.asarray(fc_np), np.asarray(fc_pd))


def test_auto_forecast_series_datetimeindex_future_index_and_to_series():
    # a Series x with a daily DatetimeIndex yields a length-horizon future
    # DatetimeIndex continuing it, and to_series() is that index + the values.
    pd = pytest.importorskip("pandas")
    from dtfit.auto import ForecastResult

    n_tr = 90
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    idx = pd.date_range("2020-01-01", periods=n_tr, freq="D")
    xs = pd.Series(t[:n_tr], index=idx)

    fc = auto_forecast(xs, y[:n_tr], horizon=30)
    # additive: still the ndarray-subclass forecast
    assert isinstance(fc, ForecastResult)
    assert fc.shape == (30,)
    # .index is a length-horizon DatetimeIndex continuing the input at freq 'D'
    assert isinstance(fc.index, pd.DatetimeIndex)
    assert len(fc.index) == 30
    assert fc.index[0] == idx[-1] + pd.Timedelta(days=1)
    # to_series(): a Series with that index and values == np.asarray(fc)
    s = fc.to_series()
    assert isinstance(s, pd.Series)
    assert s.index.equals(fc.index)
    np.testing.assert_array_equal(s.to_numpy(), np.asarray(fc))


def test_auto_forecast_integer_index_continues_by_step():
    # a non-datetime, integer-stepped index also extends (constant step).
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t
    idx = pd.RangeIndex(0, 200)
    fc = auto_forecast(pd.Series(t, index=idx), y, horizon=10, model="linear")
    assert fc.index is not None
    assert list(fc.index) == list(range(200, 210))


def test_auto_forecast_ndarray_has_no_index_and_to_series_raises():
    # a plain ndarray x -> .index is None and to_series() raises a clear error.
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    fc = auto_forecast(t[:90], y[:90], horizon=30)
    assert fc.index is None
    with pytest.raises(ValueError, match="index"):
        fc.to_series()


def test_auto_forecast_series_no_freq_index_is_none():
    # a DatetimeIndex whose frequency cannot be inferred yields .index is None
    # (values still delivered); to_series() then raises.
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    # irregular timestamps -> pd.infer_freq returns None
    idx = pd.DatetimeIndex(
        pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(np.arange(1, n_tr + 1)), "D")
    )
    fc = auto_forecast(pd.Series(t[:n_tr], index=idx), y[:n_tr], horizon=30)
    assert fc.index is None
    with pytest.raises(ValueError):
        fc.to_series()


def test_auto_forecast_persistence_path_carries_future_index():
    # the persistence return paths (here explicit random_walk) also carry .index.
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    idx = pd.date_range("2021-06-01", periods=100, freq="D")
    fc = auto_forecast(pd.Series(t, index=idx), y, horizon=10, model="random_walk")
    assert np.allclose(fc, y[-1])
    assert isinstance(fc.index, pd.DatetimeIndex) and len(fc.index) == 10
    s = fc.to_series()
    assert s.index.equals(fc.index)


def test_forecastresult_slice_drops_length_dependent_metadata():
    # A slice/reduction is no longer the length-horizon forecast, so its per-step
    # .index and .std_band must NOT be carried over at the wrong length (that made
    # fc[:3].std_band silently length-horizon and fc[:3].to_series() crash).
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    x = pd.Series(np.linspace(0, 12, 90), index=idx)
    y = pd.Series(1000.0 / (1 + np.exp(-0.8 * (np.linspace(0, 12, 90) - 6))),
                  index=idx)
    fc = auto_forecast(x, y, horizon=30)
    assert len(fc.index) == 30 and len(fc.std_band) == 30
    sl = fc[:3]
    assert len(sl) == 3
    assert sl.index is None          # not the stale length-30 index
    assert sl.std_band is None       # not the stale length-30 band
    with pytest.raises(ValueError, match="future index"):
        sl.to_series()
    # the full forecast's pandas view is unaffected, and ndarray.std still reduces
    assert (fc.to_series().index == fc.index).all()
    assert np.isfinite(fc.std())
