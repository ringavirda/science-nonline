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
    result = fit_eac(
        x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
        bounds=([0, 0], [10, 10]), loss="soft_l1",
    )
    a, w = result.coeffs
    assert 0 <= a <= 10 and 0 <= w <= 10
    assert abs(a - true["a"]) < 0.5


def test_eac_regressor_needs_no_polyfit(arctan_data):
    # EAC fits raw data, so no polynomial pre-fit stage is required.
    x, y, _ = arctan_data
    reg = NonlineRegressor("a*atan(w*x)", "x", method="eac").fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.predict(x).shape == x.shape
