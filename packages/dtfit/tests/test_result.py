"""FittingResult: named params, uncertainty, prediction bands, serialization."""

import numpy as np
import pytest

from dtfit import fit_lsi, FittingResult


@pytest.fixture
def fit():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 300)
    y = 2.0 * np.exp(0.8 * t) + rng.normal(0, 0.05, t.size)
    return fit_lsi(t, y, "a*exp(b*t)", "t", p0=[1.0, 1.0]), t, y


def test_named_params_and_back_compat(fit):
    r, t, y = fit
    assert set(r.params) == {"a", "b"}
    assert r.params["a"] == pytest.approx(2.0, abs=0.1)
    # back-compat: bare coeffs array + callable model
    assert r.coeffs.shape == (2,)
    assert np.asarray(r.model(t)).shape == t.shape


def test_stderr_and_confidence_intervals(fit):
    r, t, y = fit
    se = r.stderr()
    assert set(se) == {"a", "b"} and all(v >= 0 for v in se.values())
    ci = r.confidence_intervals(level=0.95)
    lo, hi = ci["b"]
    assert lo < r.params["b"] < hi


def test_predict_with_std(fit):
    r, t, y = fit
    yhat, std = r.predict(t, return_std=True)
    assert yhat.shape == t.shape and std.shape == t.shape
    assert np.all(std >= 0)


def test_serialization_roundtrip(fit):
    r, t, y = fit
    d = r.to_dict()
    assert set(d) >= {"expr", "var", "names", "coeffs", "cov"}
    r2 = FittingResult.from_dict(d)
    assert np.allclose(r2.coeffs, r.coeffs)
    assert np.allclose(r2.predict(t), r.predict(t))  # rebuilt model matches


def test_summary_is_str(fit):
    r, _, _ = fit
    s = r.summary()
    assert "a =" in s and "b =" in s


def test_no_expr_result_degrades_gracefully():
    # a result built from only a callable (no expr) -> model() works but
    # UQ/serialization are unavailable
    x = np.linspace(0, 1, 20)
    r = FittingResult(coeffs=np.array([0.0, 0.0, 1.0]), model=np.poly1d([1.0, 0.0, 0.0]))
    assert np.asarray(r.model(x)).shape == x.shape
    with pytest.raises(ValueError):
        r.to_dict()
