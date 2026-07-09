"""scikit-learn compatibility of NonlineRegressor."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from dtfit import NonlineRegressor


def _reg():
    return NonlineRegressor("a*atan(w*x)", "x", method="eac", p0=[1.0, 1.0])


def test_fit_predict_score(arctan_data):
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.predict(x).shape == x.shape
    assert reg.n_features_in_ == 1
    assert reg.score(x, y) > 0.8


def test_clone_and_get_params():
    reg = _reg()
    assert reg.get_params()["method"] == "eac"
    cloned = clone(reg)
    assert not hasattr(cloned, "coef_")


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        _reg().predict(np.array([1.0, 2.0]))


def test_pipeline(arctan_data):
    x, y, _ = arctan_data
    pipe = Pipeline([("reg", _reg())]).fit(x.reshape(-1, 1), y)
    assert pipe.score(x.reshape(-1, 1), y) > 0.8


def test_cross_val_score(arctan_data):
    x, y, _ = arctan_data
    scores = cross_val_score(
        _reg(), x.reshape(-1, 1), y, cv=KFold(3, shuffle=True, random_state=0)
    )
    assert len(scores) == 3
    assert scores.mean() > 0.8


def test_multifeature_rejected():
    reg = NonlineRegressor("a*x", "x")
    with pytest.raises(ValueError):
        reg.fit(np.zeros((10, 2)), np.zeros(10))


def test_predict_multifeature_rejected(arctan_data):
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    with pytest.raises(ValueError):
        reg.predict(np.zeros((5, 2)))


def test_sklearn_tags():
    tags = _reg().__sklearn_tags__()
    assert tags.estimator_type == "regressor"
    assert tags.target_tags.required is True


def test_feature_names_in_(arctan_data):
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    df = pd.DataFrame({"x": x})
    reg = _reg().fit(df, y)
    assert list(reg.feature_names_in_) == ["x"]
    assert reg.n_features_in_ == 1


def test_zero_arg_constructible_and_clonable():
    """scikit-learn contract: an estimator must be constructible with no args and
    every ``__init__`` parameter must have a default (so ``clone`` and
    meta-estimator introspection work)."""
    import inspect

    sig = inspect.signature(NonlineRegressor.__init__)
    required = [
        name for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect._empty
    ]
    assert not required, f"__init__ params without defaults: {required}"

    reg = NonlineRegressor()          # zero-arg construction
    cloned = clone(reg)               # relies on get_params/set_params round-trip
    assert isinstance(cloned, NonlineRegressor)
    assert cloned.get_params() == reg.get_params()
    assert not hasattr(cloned, "coef_")


def test_default_estimator_fits_affine():
    """The default model is a runnable affine fit (so ``NonlineRegressor()`` is a
    usable estimator, not merely constructible)."""
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 80)
    y = 1.5 + 0.7 * x + 0.01 * rng.standard_normal(x.size)
    reg = NonlineRegressor().fit(x, y)
    assert reg.score(x, y) > 0.99


def test_result_exposes_full_fitting_result(arctan_data):
    """``fit`` stores the full FittingResult as ``result_``, so uncertainty is
    reachable from the sklearn route."""
    from dtfit import FittingResult

    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    assert isinstance(reg.result_, FittingResult)
    assert np.allclose(reg.result_.coeffs, reg.coef_)
    assert reg.result_.converged is not None
    assert reg.result_.cov is not None
    se = reg.result_.stderr()
    assert set(se) == {"a", "w"}
    assert all(s > 0 for s in se.values())
    ci = reg.result_.confidence_intervals()
    for name, value in reg.result_.params.items():
        lo, hi = ci[name]
        assert lo <= value <= hi


def test_dict_p0_and_bounds_pass_through(arctan_data):
    """Dict-keyed p0/bounds go through the estimator to the fitter untouched."""
    x, y, truth = arctan_data
    reg = NonlineRegressor(
        "a*atan(w*x)",
        "x",
        method="eac",
        p0={"a": 4.0, "w": 1.0},
        bounds={"a": (0.0, 10.0)},
    ).fit(x, y)
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_nan_policy_forwarded(arctan_data):
    """``nan_policy`` reaches the fitter: 'raise' rejects NaN, 'omit' drops the
    bad pairs and still fits."""
    x, y, truth = arctan_data
    y_bad = y.copy()
    y_bad[3] = np.nan
    with pytest.raises(ValueError):
        _reg().fit(x, y_bad)
    reg = NonlineRegressor(
        "a*atan(w*x)", "x", method="eac", p0=[1.0, 1.0], nan_policy="omit"
    )
    assert reg.__sklearn_tags__().input_tags.allow_nan is True
    reg.fit(x, y_bad)
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_eac_kwargs_reach_fitter(monkeypatch, arctan_data):
    """Constructor kwargs are forwarded to ``fit_eac`` verbatim. A behavioral
    outcome alone cannot distinguish a dropped kwarg (``robust=True`` or
    ``loss="soft_l1"`` each rescue the outlier case on their own), so spy on
    the call."""
    x, y, _ = arctan_data
    captured = {}

    class _Stub:
        coeffs = np.array([1.0, 1.0])
        model = staticmethod(np.asarray)

    def fake_eac(xx, yy, expr, var, **kwargs):
        captured.update(kwargs)
        return _Stub()

    monkeypatch.setattr("dtfit.estimators._regressor.fit_eac", fake_eac)
    NonlineRegressor(
        "a*atan(w*x)", "x", method="eac", p0=[1.0, 1.0],
        robust=True, huber_c=2.5, loss="soft_l1", window_mode="curvature",
        nan_policy="omit", active_ratio=0.9,
    ).fit(x, y)
    assert captured["loss"] == "soft_l1"
    assert captured["window_mode"] == "curvature"
    assert captured["huber_c"] == 2.5
    assert captured["robust"] is True
    assert captured["nan_policy"] == "omit"
    assert captured["active_ratio"] == 0.9


def test_robust_loss_window_mode_forwarded(arctan_data):
    """The robust levers rescue an outlier-contaminated fit end-to-end (EAC and
    LSI routes); the per-kwarg forwarding guard is the spy test above."""
    x, y, truth = arctan_data
    y_out = y.copy()
    y_out[50] += 60.0  # gross outlier
    expected = [truth["a"], truth["w"]]
    robust = NonlineRegressor(
        "a*atan(w*x)",
        "x",
        method="eac",
        p0=[1.0, 1.0],
        robust=True,
        huber_c=2.5,
        loss="soft_l1",
        window_mode="curvature",
    ).fit(x, y_out)
    assert np.allclose(robust.coef_, expected, rtol=0.15)
    plain = _reg().fit(x, y_out)
    assert np.linalg.norm(robust.coef_ - expected) <= np.linalg.norm(
        plain.coef_ - expected
    )
    robust_lsi = NonlineRegressor(
        "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0], robust=True
    ).fit(x, y_out)
    assert np.allclose(robust_lsi.coef_, expected, rtol=0.15)


def test_sample_order_invariance(arctan_data):
    """Shuffled samples give the same fit as ordered ones (the estimator sorts
    by x before the integral fitters)."""
    x, y, _ = arctan_data
    rng = np.random.default_rng(1)
    perm = rng.permutation(x.size)
    reg_sorted = _reg().fit(x, y)
    reg_shuffled = _reg().fit(x[perm], y[perm])
    assert np.allclose(reg_sorted.coef_, reg_shuffled.coef_)


def test_pickle_round_trip(arctan_data):
    """A fitted estimator pickles (the lambdified model is rebuilt on load) and
    the original stays usable after being pickled."""
    import pickle

    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    expected = reg.predict(x[:10])
    clone_ = pickle.loads(pickle.dumps(reg))
    assert np.allclose(clone_.predict(x[:10]), expected)
    assert np.allclose(clone_.result_.coeffs, reg.result_.coeffs)
    # __getstate__ must not strip ``model_`` off the live estimator.
    assert np.allclose(reg.predict(x[:10]), expected)


def test_plain_list_input(arctan_data):
    """A plain 1-D Python list is promoted to a single column like an array."""
    x, y, _ = arctan_data
    reg = _reg().fit(list(x), list(y))
    assert reg.n_features_in_ == 1
    assert reg.predict(list(x[:5])).shape == (5,)


def test_sparse_input_rejected(arctan_data):
    """Sparse X gets the standard scikit-learn 'dense data required' error."""
    sparse = pytest.importorskip("scipy.sparse")
    x, y, _ = arctan_data
    with pytest.raises(TypeError, match="[Ss]parse"):
        _reg().fit(sparse.csr_matrix(x.reshape(-1, 1)), y)


def test_dict_p0_works_on_dsb_route(lint_exp_data):
    """Dict p0 is documented for ALL method routes; DSB's positional-only
    fit_dsb gets a normalized array from the estimator."""
    x, y = lint_exp_data
    pos = NonlineRegressor(
        "a + b*x + c*exp(d*x)", "x", method="dsb",
        p0=[0.5, 0.2, 0.3, 0.4],
    ).fit(x, y)
    named = NonlineRegressor(
        "a + b*x + c*exp(d*x)", "x", method="dsb",
        p0={"a": 0.5, "b": 0.2, "c": 0.3, "d": 0.4},
    ).fit(x, y)
    assert np.allclose(pos.coef_, named.coef_)
    with pytest.raises(ValueError, match=r"p0"):
        NonlineRegressor(
            "a + b*x + c*exp(d*x)", "x", method="dsb", p0={"a": 0.5}
        ).fit(x, y)


# --- v0.3: callable models -------------------------------------------------


def test_callable_model_fits_and_scores(arctan_data):
    """A plain Python callable ``f(x, *params)`` fits and scores on the LSI
    route; its parameter names come from the signature and the result carries a
    numeric evaluator (no expression) so ``predict`` still works."""
    x, y, truth = arctan_data

    def model(x, a, w):
        return a * np.arctan(w * x)

    reg = NonlineRegressor(model, "x", method="lsi", p0=[1.0, 1.0]).fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.score(x, y) > 0.8
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)
    # A callable carries no expression: the result predicts via param_model.
    assert reg.result_.expr is None
    assert reg.result_.param_model is not None
    assert reg.result_.names == ("a", "w")
    assert reg.predict(x[:5]).shape == (5,)


def test_callable_param_names_when_signature_opaque(arctan_data):
    """A callable whose signature cannot be introspected (``*args``) is fit by
    passing ``param_names``, which the estimator forwards to the fitter."""
    x, y, truth = arctan_data

    def model(x, *p):  # opaque signature -> names must be supplied
        return p[0] * np.arctan(p[1] * x)

    reg = NonlineRegressor(
        model, "x", method="lsi", param_names=["a", "w"], p0=[1.0, 1.0]
    ).fit(x, y)
    assert reg.result_.names == ("a", "w")
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_callable_and_param_names_forwarded_to_fitter(monkeypatch, arctan_data):
    """The callable model and ``param_names`` reach the fitter untouched (a
    behavioral outcome cannot prove ``param_names`` was forwarded, so spy)."""
    x, y, _ = arctan_data
    captured = {}

    class _Stub:
        coeffs = np.array([1.0, 1.0])
        model = staticmethod(np.asarray)

    def fake_lsi(xx, yy, expr, var, **kwargs):
        captured["expr"] = expr
        captured["param_names"] = kwargs.get("param_names")
        return _Stub()

    monkeypatch.setattr("dtfit.estimators._regressor.fit_lsi", fake_lsi)

    def model(x, *p):
        return p[0] * x

    NonlineRegressor(model, "x", method="lsi", param_names=["a", "b"]).fit(x, y)
    assert captured["expr"] is model
    assert list(captured["param_names"]) == ["a", "b"]


def test_callable_model_on_dsb_raises(lint_exp_data):
    """DSB is symbolic-only: a callable model raises a clear error at fit time."""
    x, y = lint_exp_data

    def model(x, a, b):
        return a + b * x

    with pytest.raises(ValueError, match=r"dsb"):
        NonlineRegressor(model, "x", method="dsb").fit(x, y)


# --- v0.3: sample_weight ---------------------------------------------------


def test_sample_weight_downweights_outliers(arctan_data):
    """Down-weighting a contaminated region beats an unweighted fit: the weighted
    coefficients land closer to the ground truth."""
    x, y, truth = arctan_data
    expected = np.array([truth["a"], truth["w"]])
    y_bad = y.copy()
    corrupt = (x > 4.0) & (x < 6.0)
    y_bad[corrupt] += 25.0  # a whole contaminated band
    weight = np.ones_like(x)
    weight[corrupt] = 1e-3  # trust the corrupted samples far less
    weighted = NonlineRegressor(
        "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
    ).fit(x, y_bad, sample_weight=weight)
    plain = NonlineRegressor(
        "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
    ).fit(x, y_bad)
    assert np.linalg.norm(weighted.coef_ - expected) < np.linalg.norm(
        plain.coef_ - expected
    )
    assert np.allclose(weighted.coef_, expected, rtol=0.15)


def test_sample_weight_forwarded_as_sigma(monkeypatch, arctan_data):
    """``sample_weight`` becomes the fitter's ``sigma = 1/sqrt(weight)`` and is
    forwarded as *relative* weights (``absolute_sigma`` left at its False
    default)."""
    x, y, _ = arctan_data
    captured = {}

    class _Stub:
        coeffs = np.array([1.0, 1.0])
        model = staticmethod(np.asarray)

    def fake_lsi(xx, yy, expr, var, **kwargs):
        captured.update(kwargs)
        return _Stub()

    monkeypatch.setattr("dtfit.estimators._regressor.fit_lsi", fake_lsi)
    weight = np.linspace(0.5, 2.0, x.size)
    NonlineRegressor("a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]).fit(
        x, y, sample_weight=weight
    )
    # arctan_data x is already ascending, so the estimator's x-sort is a no-op
    # and sigma aligns 1:1 with the input weights.
    assert np.allclose(captured["sigma"], 1.0 / np.sqrt(weight))
    assert captured.get("absolute_sigma", False) is False


def test_zero_sample_weight_ignores_sample(arctan_data):
    """A zero weight effectively drops its sample (mapped to a huge sigma) rather
    than crashing on 1/sqrt(0); the fit still succeeds and is sensible."""
    x, y, truth = arctan_data
    weight = np.ones_like(x)
    weight[::7] = 0.0  # scatter some zero-weight samples
    reg = NonlineRegressor(
        "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
    ).fit(x, y, sample_weight=weight)
    assert np.all(np.isfinite(reg.coef_))
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_all_zero_sample_weight_raises(arctan_data):
    """All-zero weights are an error (nothing to fit)."""
    x, y, _ = arctan_data
    with pytest.raises(ValueError, match=r"(?i)zero"):
        NonlineRegressor(
            "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
        ).fit(x, y, sample_weight=np.zeros_like(x))


def test_negative_sample_weight_raises(arctan_data):
    """A negative weight is meaningless for a curve fit and raises."""
    x, y, _ = arctan_data
    weight = np.ones_like(x)
    weight[0] = -1.0
    with pytest.raises(ValueError, match=r"(?i)non-negative|negative"):
        NonlineRegressor(
            "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
        ).fit(x, y, sample_weight=weight)


def test_sample_weight_on_dsb_raises(lint_exp_data):
    """DSB has no per-sample weighting: a ``sample_weight`` raises."""
    x, y = lint_exp_data
    with pytest.raises(ValueError, match=r"sample_weight is not supported"):
        NonlineRegressor(
            "a + b*x + c*exp(d*x)", "x", method="dsb"
        ).fit(x, y, sample_weight=np.ones_like(x))


# --- v0.3: fit-quality stats surfaced via result_ --------------------------


def test_result_has_rsquared(arctan_data):
    """The LSI fitter records the v0.3 fit-quality stats, reachable through
    ``result_``."""
    x, y, _ = arctan_data
    reg = NonlineRegressor("a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]).fit(
        x, y
    )
    r2 = reg.result_.rsquared
    assert r2 is not None
    assert 0.8 < r2 <= 1.0
    assert reg.result_.aic is not None
    assert reg.result_.bic is not None
    assert reg.result_.n_obs == x.size


def test_sample_weight_with_nan_omit_both_routes(arctan_data):
    """sample_weight + nan_policy='omit' + a NaN row must fit on BOTH the lsi and
    eac routes (regression: the two fitters once disagreed on sigma length under
    omit, so the lsi route raised while eac fit)."""
    x, y, truth = arctan_data
    y = y.copy()
    y[7] = np.nan
    w = np.full(x.size, 2.0)  # full-length weights, one per raw sample
    for method in ("lsi", "eac"):
        reg = NonlineRegressor(
            "a*atan(w*x)", "x", method=method, p0=[1.0, 1.0],
            nan_policy="omit",
        ).fit(x, y, sample_weight=w)
        assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.2)


# --- v0.4: pandas in -> pandas out -----------------------------------------


def test_predict_series_returns_aligned_series(arctan_data):
    """fit on a Series x/y, then predict(Series) -> a Series carrying the input's
    index whose values equal the plain-ndarray prediction (values unchanged)."""
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    idx = pd.RangeIndex(100, 100 + x.size)
    sx = pd.Series(x, index=idx)
    sy = pd.Series(y, index=idx)
    reg = _reg().fit(sx, sy)

    xq = x[:5]
    idxq = pd.Index([7, 8, 9, 10, 11])
    pred = reg.predict(pd.Series(xq, index=idxq))
    assert isinstance(pred, pd.Series)
    assert list(pred.index) == list(idxq)
    # values are byte-for-byte the ndarray prediction (no numeric change)
    assert np.array_equal(pred.to_numpy(), reg.predict(xq))


def test_predict_ndarray_still_ndarray(arctan_data):
    """A plain ndarray (or list) input keeps returning an ndarray -- the pandas
    branch must not leak into the non-pandas path."""
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    out = reg.predict(x[:5])
    assert isinstance(out, np.ndarray)
    assert not hasattr(out, "index")
    out_list = reg.predict(list(x[:5]))
    assert isinstance(out_list, np.ndarray)


def test_predict_single_col_dataframe_returns_series(arctan_data):
    """A single-column DataFrame X predicts to a Series aligned to the frame's
    row index."""
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    # fit with a named single-column frame so feature_names_in_ matches the
    # query frame (no sklearn feature-name warning at predict time).
    reg = _reg().fit(pd.DataFrame({"x": x}), y)
    idxq = pd.Index([3, 4, 5, 6, 7])
    df = pd.DataFrame({"x": x[:5]}, index=idxq)
    pred = reg.predict(df)
    assert isinstance(pred, pd.Series)
    assert list(pred.index) == list(idxq)
    # values match an all-ndarray fit+predict (identical data -> identical fit),
    # keeping the comparison free of sklearn's feature-name mismatch warning.
    expected = _reg().fit(x, y).predict(x[:5])
    assert np.array_equal(pred.to_numpy(), expected)


def test_sample_weight_as_series_works(arctan_data):
    """A pandas Series sample_weight is coerced to ndarray and reaches the fitter
    as sigma exactly like an ndarray weight."""
    pd = pytest.importorskip("pandas")
    x, y, truth = arctan_data
    weight = np.ones_like(x)
    corrupt = (x > 4.0) & (x < 6.0)
    y_bad = y.copy()
    y_bad[corrupt] += 25.0
    weight[corrupt] = 1e-3
    sw = pd.Series(weight, index=pd.RangeIndex(x.size))
    weighted = NonlineRegressor(
        "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
    ).fit(x, y_bad, sample_weight=sw)
    # same result as the equivalent ndarray weight
    weighted_np = NonlineRegressor(
        "a*atan(w*x)", "x", method="lsi", p0=[1.0, 1.0]
    ).fit(x, y_bad, sample_weight=weight)
    assert np.allclose(weighted.coef_, weighted_np.coef_)
    assert np.allclose(weighted.coef_, [truth["a"], truth["w"]], rtol=0.15)
