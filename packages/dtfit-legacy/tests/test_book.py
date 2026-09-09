import numpy as np
import pytest

from dtfit_legacy.book import (
    dsb_balance,
    eac_areas,
    intermediate_polynomial,
    lsi_integral_monomial,
    monomial_weight_matrix,
)


def _exp(n=200, h=2.0):
    t = np.linspace(0.0, h, n)
    return t, 3.0 * np.exp(-1.2 * t)


def test_intermediate_polynomial_is_the_maclaurin_spectrum():
    t, y = _exp()
    d, h = intermediate_polynomial(t, y, 8)
    assert h == pytest.approx(2.0)
    # the data's Maclaurin coefficients of 3 exp(-1.2 t): 3, -3.6, 2.16
    assert d[:3] == pytest.approx([3.0, -3.6, 2.16], rel=0.05)


def test_dsb_balance_recovers_the_exponential():
    # the square balance is exact only inside the Taylor regime, so the
    # window is short; on the wide window of _exp() it is biased by design
    t, y = _exp(h=0.02)
    r = dsb_balance(t, y, "a*exp(b*t)", "t", p0=[1.0, -1.0])
    assert r.coeffs == pytest.approx([3.0, -1.2], rel=0.03)


def test_lsi_integral_monomial_recovers_at_low_order():
    t, y = _exp()
    r = lsi_integral_monomial(t, y, "a*exp(b*t)", "t", order=6,
                              p0=[1.0, -1.0])
    assert r.coeffs == pytest.approx([3.0, -1.2], rel=0.02)


def test_monomial_weight_matrix_conditioning_explodes_by_order_12():
    # dissertation eq 2.14: cond(M) ~ exp(3.5 K), above 1e16 near K = 11
    c6 = np.linalg.cond(monomial_weight_matrix(2.0, 6))
    c12 = np.linalg.cond(monomial_weight_matrix(2.0, 12))
    assert c12 > 1e16 > c6


def test_lsi_integral_monomial_breaks_down_at_high_order():
    t, y = _exp()
    try:
        r = lsi_integral_monomial(t, y, "a*exp(b*t)", "t", order=14,
                                  p0=[1.0, -1.0])
    except ValueError:
        return  # the weight matrix is no longer positive definite
    err = np.max(np.abs(r.coeffs / np.array([3.0, -1.2]) - 1.0))
    assert err > 0.02  # it survives, but no longer at low-order accuracy


def test_eac_areas_recovers_inside_the_convergence_radius():
    t = np.linspace(0.0, 0.5, 200)
    y = 1.0 * np.arctan(1.0 * t)
    r = eac_areas(t, y, "a*atan(w*t)", "t", order=6, p0=[0.8, 0.8])
    assert r.coeffs == pytest.approx([1.0, 1.0], rel=0.05)


def test_eac_areas_truncation_bias_grows_outside_the_radius():
    # dissertation figure 2.4: the truncated spectrum of atan(w t) has
    # radius 1/w; a window past it biases the source form
    def err(h):
        t = np.linspace(0.0, h, 400)
        y = np.arctan(t)
        r = eac_areas(t, y, "a*atan(w*t)", "t", order=6, p0=[0.8, 0.8])
        return float(np.max(np.abs(r.coeffs - 1.0)))
    wide, narrow = err(3.0), err(0.5)
    assert not np.isfinite(wide) or wide > 10.0 * narrow
