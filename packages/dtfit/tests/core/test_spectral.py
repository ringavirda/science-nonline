"""Shared spectral-match solver (``dtfit._core._spectral``) -- bounds gating.

The global (differential-evolution) stage needs a finite search box, but the
local trf solve handles ``+/-inf`` bounds natively. ``solve_weighted_nlls``
must therefore run DE only when every bound is finite while *always* passing
the bounds to the local solve -- a partially-bounded problem stays constrained
instead of being silently solved unbounded (the historical upstream behavior).
"""

import numpy as np
import pytest

from dtfit import fit_lsi
from dtfit._core._spectral import solve_weighted_nlls


def _weighted_problem(target):
    """A tiny diagonal-weighted NLLS: residual(c) = sqrt_w * (target - c)."""
    target = np.asarray(target, dtype=float)
    sqrt_w = np.ones_like(target)

    def residual(c):
        return sqrt_w * (target - c)

    return residual, sqrt_w, target


def test_mixed_bounds_run_local_solve_and_respect_bound():
    # One parameter bounded (0, inf), one unbounded: DE cannot run (infinite
    # box) but the trf solve must still receive and honour the bounds.
    residual, sqrt_w, beta = _weighted_problem([-3.0, 2.0])
    guess = np.array([1.0, 1.0])
    coeffs, jac, converged, message, nfev = solve_weighted_nlls(
        residual, sqrt_w, beta, guess, p0=guess,
        bounds=[(0.0, np.inf), (-np.inf, np.inf)],
    )
    assert converged
    assert coeffs[0] >= 0.0, "the (0, inf) bound was dropped"
    assert abs(coeffs[0] - 0.0) < 1e-6  # clamped at the active bound
    assert abs(coeffs[1] - 2.0) < 1e-6  # free parameter reaches the target
    assert jac.shape == (2, 2)
    assert isinstance(message, str) and message
    assert isinstance(nfev, int) and nfev > 0


def test_mixed_bounds_without_p0_still_constrained():
    # No seed supplied: the infinite box forbids DE, so the driver must fall
    # back to a bounded local solve from the default guess -- not crash, and
    # not drop the finite bound.
    residual, sqrt_w, beta = _weighted_problem([-1.0, 4.0])
    coeffs, _, converged, _, _ = solve_weighted_nlls(
        residual, sqrt_w, beta, np.ones(2), p0=None,
        bounds=[(0.5, np.inf), (-np.inf, np.inf)],
    )
    assert converged
    assert coeffs[0] >= 0.5
    assert abs(coeffs[1] - 4.0) < 1e-6


def test_all_finite_bounds_still_use_global_stage_reproducibly():
    # Finite box + no seed -> the DE global stage runs (seeded, reproducible).
    residual, sqrt_w, beta = _weighted_problem([0.3, -0.7])
    a = solve_weighted_nlls(residual, sqrt_w, beta, np.ones(2), p0=None,
                            bounds=[(-1.0, 1.0), (-1.0, 1.0)], seed=3)[0]
    b = solve_weighted_nlls(residual, sqrt_w, beta, np.ones(2), p0=None,
                            bounds=[(-1.0, 1.0), (-1.0, 1.0)], seed=3)[0]
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, [0.3, -0.7], atol=1e-4)


def test_fit_lsi_with_mixed_bounds_converges_and_respects_bound():
    # End-to-end through fit_lsi: one param bounded (0, inf), one unbounded.
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 2.0, 120)
    y = 2.5 * np.exp(-1.2 * x) + rng.normal(0, 0.02, x.size)
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                     bounds=[(0.0, np.inf), (-np.inf, np.inf)])
    a, b = result.coeffs
    assert result.converged
    assert a >= 0.0
    assert abs(a - 2.5) < 0.3 and abs(b + 1.2) < 0.3


def test_infinite_bounds_reject_nothing_finite_seeded():
    # Sanity: an all-(-inf, inf) box behaves like the unbounded fit.
    residual, sqrt_w, beta = _weighted_problem([1.5])
    coeffs, _, converged, _, _ = solve_weighted_nlls(
        residual, sqrt_w, beta, np.ones(1), p0=None,
        bounds=[(-np.inf, np.inf)],
    )
    assert converged and abs(coeffs[0] - 1.5) < 1e-6


def test_lo_above_hi_raises_from_fit_lsi():
    x = np.linspace(0.0, 2.0, 60)
    y = np.exp(-x)
    with pytest.raises(ValueError, match="'b'"):
        fit_lsi(x, y, "a*exp(b*x)", "x", bounds={"b": (2.0, -2.0)})
