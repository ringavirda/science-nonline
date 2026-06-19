"""LSI (least-squares integral) batch method."""

import numpy as np

from dtfit import fit_lsi


def test_fit_lsi_recovers_exponential(exp_data):
    x, y, true = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    a, b = result.coeffs  # params sorted by name: a, b
    assert abs(a - true["a"]) < 0.3
    assert abs(b - true["b"]) < 0.3
    assert callable(result.model)


def test_lsi_returns_covariance(exp_data):
    x, y, _ = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    assert result.cov is not None
    assert result.cov.shape == (2, 2)
    # Standard errors are small and finite for this clean signal.
    se = np.sqrt(np.diag(result.cov))
    assert np.all(np.isfinite(se)) and np.all(se < 0.5)


def test_lsi_auto_order(exp_data):
    x, y, true = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", k_star="auto", p0=[1.0, -0.5])
    a, b = result.coeffs
    assert abs(a - true["a"]) < 0.3
    assert abs(b - true["b"]) < 0.3
