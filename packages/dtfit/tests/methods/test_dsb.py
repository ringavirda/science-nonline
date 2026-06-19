"""DSB (symbolic differential spectra balance) reference method."""

from typing import cast

import numpy as np
import pytest

from dtfit import NonlineRegressor, fit_dsb
from dtfit.methods import taylor_coeffs
import sympy as sp


def test_dsb_regressor_fits(lint_exp_data):
    # The regressor runs the required polynomial pre-fit internally.
    x, y = lint_exp_data
    reg = NonlineRegressor(
        "a0 + a1*x + a2*exp(a3*x)", "x", method="dsb"
    ).fit(x, y)
    assert len(reg.coef_) == 4
    assert reg.score(x, y) > 0.8


def test_taylor_coeffs_match_known_series():
    t = sp.Symbol("x")
    # exp(x): a_k = 1/k!
    coeffs = taylor_coeffs(cast(sp.Expr, sp.sympify("exp(x)")), t, 4)
    assert [sp.nsimplify(c) for c in coeffs] == [
        sp.Rational(1, sp.factorial(k)) for k in range(5)
    ]
    # sin(x): 0, 1, 0, -1/6, 0
    s = [float(c) for c in taylor_coeffs(cast(sp.Expr, sp.sympify("sin(x)")), t, 4)]
    assert np.allclose(s, [0, 1, 0, -1 / 6, 0])


def test_dsb_fits_models_without_handwritten_discretes():
    # log and rational forms had NO closed-form discrete rule in the old
    # Spectrum.__reflect; the generic Taylor balance handles them now.
    x = np.linspace(0.0, 1.2, 300)
    y = 0.2 + 1.5 * np.log(1 + 0.9 * x)
    reg = NonlineRegressor(
        "a0 + a1*log(1 + a2*x)", "x", method="dsb", poly_degree=6
    ).fit(x, y)
    assert reg.score(x, y) > 0.95


def test_dsb_underdetermined_balance_raises():
    # atan's even Maclaurin orders vanish, so degree-2 poly cannot identify the
    # three parameters -- DSB must say so rather than return garbage.
    coeffs_poly = np.array([1.0, 2.0, 0.0])  # only orders 0,1 constrain params
    with pytest.raises(RuntimeError):
        fit_dsb(coeffs_poly, "a0 + a1*atan(a2*x)", "x")
