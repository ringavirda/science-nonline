"""EAC (equal-areas criterion) batch method."""

import numpy as np

from dtfit import NonlineRegressor, fit_eac


def test_fit_eac_recovers_arctan(arctan_data):
    x, y, true = arctan_data
    result = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    a, w = result.coeffs  # params sorted by name: a, w
    assert abs(a - true["a"]) < 0.5
    assert abs(w - true["w"]) < 0.5


def test_eac_overdetermined_returns_covariance(arctan_data):
    x, y, _ = arctan_data
    # Default n_windows = 2 * n_params > n_params -> overdetermined.
    result = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    assert result.cov is not None
    assert result.cov.shape == (2, 2)
    assert np.all(np.isfinite(np.sqrt(np.diag(result.cov))))


def test_eac_exactly_determined_has_no_covariance(arctan_data):
    x, y, _ = arctan_data
    result = fit_eac(x, y, "a*atan(w*x)", "x", n_windows=2, p0=[1.0, 1.0])
    assert result.cov is None


def test_eac_bounds_and_robust_loss(arctan_data):
    x, y, true = arctan_data
    # Per-parameter (lo, hi) pairs: the canonical bounds form. (The historical
    # scipy-tuple ([0, 0], [10, 10]) spelling is ambiguous for 2 parameters and
    # is now read as per-parameter pairs -- see normalize_bounds.)
    result = fit_eac(
        x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
        bounds=[(0, 10), (0, 10)], loss="soft_l1",
    )
    a, w = result.coeffs
    assert 0 <= a <= 10 and 0 <= w <= 10
    assert abs(a - true["a"]) < 0.5


def test_eac_curvature_window_mode(arctan_data):
    x, y, true = arctan_data
    result = fit_eac(x, y, "a*atan(w*x)", "x", window_mode="curvature",
                     p0=[1.0, 1.0])
    a, _ = result.coeffs
    assert abs(a - true["a"]) < 0.5
    assert result.cov is not None and result.cov.shape == (2, 2)


def test_eac_regressor_needs_no_polyfit(arctan_data):
    # EAC fits raw data, so no polynomial pre-fit stage is required.
    x, y, _ = arctan_data
    reg = NonlineRegressor("a*atan(w*x)", "x", method="eac").fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.predict(x).shape == x.shape


def test_validation_rejects_malformed_input():
    import pytest
    from dtfit import fit_lsi
    x = np.linspace(0, 1, 40)
    y = np.exp(0.7 * x)
    # length mismatch
    with pytest.raises(ValueError, match="same length"):
        fit_eac(x, y[:-1], "a*exp(b*x)", "x", p0=[1.0, 1.0])
    # non-finite
    yb = y.copy()
    yb[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_lsi(x, yb, "a*exp(b*x)", "x")
    # 2-D input
    with pytest.raises(ValueError, match="1-D"):
        fit_lsi(x.reshape(-1, 1), y, "a*exp(b*x)", "x")


def test_fit_lsi_random_state_is_reproducible():
    from dtfit import fit_lsi
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 3, 120)
    y = 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.05, x.size)
    bounds = [(0.1, 5.0), (0.1, 2.0)]
    a = fit_lsi(x, y, "a*exp(b*x)", "x", bounds=bounds, random_state=7).coeffs
    b = fit_lsi(x, y, "a*exp(b*x)", "x", bounds=bounds, random_state=7).coeffs
    np.testing.assert_allclose(a, b)  # same seed -> identical global search


# --- p0 / bounds normalization (dict, positional, scipy-tuple forms) -------- #
def test_eac_dict_p0_matches_positional(arctan_data):
    x, y, _ = arctan_data
    pos = fit_eac(x, y, "a*atan(w*x)", "x", p0=[2.0, 1.0]).coeffs
    named = fit_eac(x, y, "a*atan(w*x)", "x", p0={"a": 2.0, "w": 1.0}).coeffs
    np.testing.assert_allclose(named, pos)


def test_eac_partial_dict_bounds(arctan_data):
    x, y, true = arctan_data
    # Only 'a' is bounded; 'w' stays unbounded -- the dict may be partial.
    result = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                     bounds={"a": (0.0, 10.0)})
    a, _ = result.coeffs
    assert 0.0 <= a <= 10.0
    assert abs(a - true["a"]) < 0.5


def test_eac_scipy_tuple_bounds_three_params():
    # n_params >= 3: the scipy-style (lo, hi) arrays form is unambiguous and
    # must keep working.
    rng = np.random.default_rng(2)
    x = np.linspace(0, 4, 200)
    y = 1.0 + 2.0 * np.exp(-1.5 * x) + rng.normal(0, 0.02, x.size)
    result = fit_eac(x, y, "c + a*exp(-b*x)", "x", p0=[1.0, 1.0, 1.0],
                     bounds=([0, 0, 0], [10, 10, 10]))
    a, b, c = result.coeffs  # sorted names: a, b, c
    assert np.all((result.coeffs >= 0) & (result.coeffs <= 10))
    assert abs(a - 2.0) < 0.3 and abs(b - 1.5) < 0.3 and abs(c - 1.0) < 0.3


def test_eac_p0_dict_missing_and_unknown_names_raise(arctan_data):
    import pytest
    x, y, _ = arctan_data
    with pytest.raises(ValueError, match=r"missing \['w'\]"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0={"a": 1.0})
    with pytest.raises(ValueError, match=r"unknown \['z'\]"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0={"a": 1.0, "w": 1.0, "z": 3.0})


def test_eac_bounds_lo_above_hi_raises(arctan_data):
    import pytest
    x, y, _ = arctan_data
    with pytest.raises(ValueError, match="'w'"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                bounds=[(0.0, 10.0), (5.0, 1.0)])


def test_eac_active_ratio_defaults_to_all_samples(arctan_data):
    """v0.2: the fitter no longer silently discards the trailing 20% -- the
    default active region is the whole record; 0.8 is the opt-in transient
    recipe and gives a (slightly) different window layout."""
    x, y, true = arctan_data
    full = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    lead = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], active_ratio=0.8)
    assert abs(full.coeffs[0] - true["a"]) < 0.5
    assert not np.allclose(full.coeffs, lead.coeffs, atol=1e-12)


def test_eac_robust_message_reflects_inner_solver(arctan_data):
    x, y, _ = arctan_data
    result = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], robust=True)
    # The robust path must carry the last inner solver's status, not the old
    # hard-coded 'robust IRLS' success constant.
    assert result.message != "robust IRLS"
    assert result.message.startswith("robust IRLS (")


def test_eac_converged_flag_propagates_solver_failure(monkeypatch, arctan_data):
    """The ``converged`` FLAG must reflect the last solver on BOTH paths (the
    robust path previously stamped an unconditional success)."""
    import types

    import dtfit.methods._eac as _eac_mod

    x, y, _ = arctan_data

    real_ls = _eac_mod.least_squares

    def failing_ls(*args, **kwargs):
        sol = real_ls(*args, **kwargs)
        return types.SimpleNamespace(x=sol.x, jac=sol.jac, fun=sol.fun,
                                     success=False, message="inner failed")

    monkeypatch.setattr(_eac_mod, "least_squares", failing_ls)
    plain = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    assert plain.converged is False
    robust = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], robust=True)
    assert robust.converged is False
    assert "inner failed" in robust.message


def test_eac_default_guess_clipped_into_bounds(exp_data):
    """A named bracket that excludes the default all-ones seed must not crash
    ('Initial guess is outside of provided bounds'): the seed is clipped into
    the box, matching fit_lsi's solver behavior."""
    x, y, true = exp_data
    res = fit_eac(x, y, "a*exp(-b*x)", "x", bounds={"a": (2.0, 5.0)})
    assert res.converged
    assert 2.0 <= res.params["a"] <= 5.0
    assert abs(res.params["a"] - true["a"]) < 0.5
    # legacy scipy-tuple form (3 params -- unambiguous) with 1.0 outside
    res3 = fit_eac(x, y, "a*exp(-b*x) + c", "x",
                   bounds=([2.0, 0.0, -1.0], [5.0, 10.0, 1.0]))
    assert 2.0 <= res3.params["a"] <= 5.0
