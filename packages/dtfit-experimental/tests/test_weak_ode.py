"""Weak-form ODE identification: the operators read derivatives from the data,
and each linearized law recovers its constants without an ODE solve or a p0."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dtfit_experimental.weak_ode import (
    fit_logistic,
    fit_lotka_volterra_prey,
    fit_michaelis_menten,
    seed_nlls,
    weak_operators,
)


def test_weak_operators_read_the_first_derivative():
    # I1(g) projects g' against the test functions; check it against a
    # function with a known derivative (g = sin, g' = cos) to a tight
    # tolerance on a fine grid.
    t = np.linspace(0.0, 2.0 * np.pi, 2000)
    i0, i1, _ = weak_operators(t, n_test=8)
    got = i1(np.sin(t))
    want = i0(np.cos(t))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_weak_operators_read_the_second_derivative():
    t = np.linspace(0.0, 2.0 * np.pi, 2000)
    i0, _, i2 = weak_operators(t, n_test=8)
    got = i2(np.sin(t))          # projects g''
    want = i0(-np.sin(t))        # g'' = -sin
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_logistic_recovers_r_and_k():
    r, k, y0 = 1.4, 5.0, 0.4
    t = np.linspace(0.0, 8.0, 500)
    y = solve_ivp(lambda _t, yy: r * yy * (1 - yy / k), (0, 8), [y0],
                  t_eval=t, rtol=1e-9).y[0]
    y = y + 0.01 * y0 * np.random.default_rng(0).standard_normal(t.size)
    p = fit_logistic(t, y)
    assert p["r"] == pytest.approx(r, rel=0.06)
    assert p["K"] == pytest.approx(k, rel=0.06)


def test_michaelis_menten_recovers_vm_and_km():
    vm, km, y0 = 1.2, 0.8, 4.0
    t = np.linspace(0.0, 8.0, 400)
    y = solve_ivp(lambda _t, yy: -vm * yy / (km + yy), (0, 8), [y0],
                  t_eval=t, rtol=1e-9).y[0]
    y = y + 0.01 * y0 * np.random.default_rng(1).standard_normal(t.size)
    p = fit_michaelis_menten(t, y)
    assert p["Vm"] == pytest.approx(vm, rel=0.08)
    assert p["Km"] == pytest.approx(km, rel=0.15)   # Km is the softer constant


def test_lotka_volterra_recovers_rates_from_prey_alone():
    al, be, de, ga = 1.1, 0.4, 0.1, 0.4
    t = np.linspace(0.0, 30.0, 1500)
    s = solve_ivp(
        lambda _t, z: [al * z[0] - be * z[0] * z[1],
                       de * z[0] * z[1] - ga * z[1]],
        (0, 30), [10.0, 5.0], t_eval=t, rtol=1e-10, atol=1e-12,
    )
    x = s.y[0] * (1 + 0.005 * np.random.default_rng(2).standard_normal(t.size))
    p = fit_lotka_volterra_prey(t, x)
    assert p["alpha"] == pytest.approx(al, rel=0.10)
    assert p["gamma"] == pytest.approx(ga, rel=0.10)
    assert p["delta"] == pytest.approx(de, rel=0.15)
    assert "beta" not in p   # structurally unidentifiable from the prey alone


def _damped(w, z, t):
    return solve_ivp(lambda _t, s: [s[1], -2 * z * w * s[1] - w * w * s[0]],
                     (t[0], t[-1]), [1.0, 0.0], t_eval=t, rtol=1e-10).y[0]


def test_damped_oscillator_recovers_omega_and_zeta():
    from dtfit_experimental.weak_ode import fit_damped_oscillator
    w, z = 3.0, 0.15
    t = np.linspace(0.0, 8.0, 600)
    y = _damped(w, z, t) + 0.01 * np.random.default_rng(3).standard_normal(t.size)
    p = fit_damped_oscillator(t, y)
    assert p["omega"] == pytest.approx(w, rel=0.05)
    assert p["zeta"] == pytest.approx(z, rel=0.20)


def test_recovery_is_stable_across_seeds():
    # median over seeds at 5 percent noise stays within a few percent, and no
    # seed blows up (the linear solve cannot land in a wrong basin).
    r, k, y0 = 1.4, 5.0, 0.4
    t = np.linspace(0.0, 8.0, 500)
    yc = solve_ivp(lambda _t, y: r * y * (1 - y / k), (0, 8), [y0],
                   t_eval=t, rtol=1e-9).y[0]
    errs = []
    for s in range(20):
        y = yc + 0.05 * y0 * np.random.default_rng(s).standard_normal(t.size)
        p = fit_logistic(t, y)
        errs.append(max(abs(p["r"] / r - 1), abs(p["K"] / k - 1)))
    assert np.median(errs) < 0.05
    assert np.max(errs) < 0.25


def test_fits_a_non_uniform_grid():
    # the trapezoid weights from np.gradient carry a clustered grid
    r, k, y0 = 1.2, 4.0, 0.5
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 8.0, 500))
    yc = solve_ivp(lambda _t, y: r * y * (1 - y / k), (t[0], t[-1]), [y0],
                   t_eval=t, rtol=1e-9).y[0]
    y = yc + 0.02 * y0 * rng.standard_normal(t.size)
    p = fit_logistic(t, y)
    assert p["r"] == pytest.approx(r, rel=0.08)
    assert p["K"] == pytest.approx(k, rel=0.08)


def test_weak_form_beats_finite_difference_under_noise():
    # differentiating the noisy y pointwise (finite difference) then fitting the
    # same linear law is far noisier than the weak projection; the weak form's
    # median error must be smaller.
    vm, km, y0 = 1.2, 0.8, 4.0
    t = np.linspace(0.0, 8.0, 400)
    yc = solve_ivp(lambda _t, yy: -vm * yy / (km + yy), (0, 8), [y0],
                   t_eval=t, rtol=1e-9).y[0]
    ew, ef = [], []
    for s in range(20):
        y = yc + 0.05 * y0 * np.random.default_rng(s).standard_normal(t.size)
        p = fit_michaelis_menten(t, y)
        ew.append(abs(p["Vm"] / vm - 1))
        dy = np.gradient(y, t)
        c = np.linalg.lstsq(np.column_stack([dy, y]),
                            -0.5 * np.gradient(y * y, t), rcond=None)[0]
        ef.append(abs(c[1] / vm - 1))
    assert np.median(ew) < np.median(ef)


def test_seed_nlls_recovers_logistic_from_a_poor_start():
    # a noiseless logistic from a 3x start: fails if the solver is skipped
    # and p0 is echoed back unchanged.
    r, k, y0 = 1.4, 5.0, 0.4
    t = np.linspace(0.0, 8.0, 500)
    y = solve_ivp(lambda _t, yy: r * yy * (1 - yy / k), (0, 8), [y0],
                  t_eval=t, rtol=1e-9, atol=1e-11).y[0]
    p0 = np.array([r, k]) * 3.0

    def rhs(_t, state, rr, kk):
        return [rr * state[0] * (1 - state[0] / kk)]

    p = seed_nlls(rhs, [y0], t, y, p0, names=["r", "K"])
    assert p != {"r": p0[0], "K": p0[1]}
    assert p["r"] == pytest.approx(r, abs=1e-6)
    assert p["K"] == pytest.approx(k, abs=1e-6)


def test_seed_nlls_rejects_mismatched_names():
    # fails if the len(p0) != len(names) guard is dropped.
    def rhs(_t, state, rr, kk):
        return [rr * state[0] * (1 - state[0] / kk)]

    t = np.linspace(0.0, 8.0, 50)
    y = np.ones_like(t)
    with pytest.raises(ValueError):
        seed_nlls(rhs, [0.4], t, y, [1.4, 5.0], names=["r"])


def test_seed_nlls_raises_on_divergent_integration():
    # y' = y**2 from y(0) = 1 blows up before t = 2; fails if a bare
    # except turns the integration failure into a nan-filled dict.
    def rhs(_t, state, _a):
        return [state[0] ** 2]

    t = np.linspace(0.0, 2.0, 50)
    y = np.ones_like(t)
    with pytest.raises(RuntimeError):
        seed_nlls(rhs, [1.0], t, y, [1.0], names=["a"])


def test_seed_nlls_rejects_empty_t():
    # fails if the t.size < 2 guard is dropped or moved below the length
    # check: t and y agree in length here, so only that guard rejects it.
    def rhs(_t, state, rr, kk):
        return [rr * state[0] * (1 - state[0] / kk)]

    with pytest.raises(ValueError):
        seed_nlls(rhs, [0.4], [], [], [1.4, 5.0], names=["r", "K"])


def test_seed_nlls_rejects_a_single_sample():
    # a lone (t, y) pair gives solve_ivp a zero-length span, which raises a
    # bare IndexError from inside the solver; fails if the guard is
    # narrowed to t.size == 0.
    def rhs(_t, state, rr, kk):
        return [rr * state[0] * (1 - state[0] / kk)]

    with pytest.raises(ValueError):
        seed_nlls(rhs, [0.4], [0.0], [0.4], [1.4, 5.0], names=["r", "K"])


def test_seed_nlls_rejects_mismatched_t_and_y_length():
    # y has size 1, which numpy broadcasts against the size-50 residual
    # without error; only the t.size != y.size guard rejects this input.
    def rhs(_t, state, rr, kk):
        return [rr * state[0] * (1 - state[0] / kk)]

    t = np.linspace(0.0, 8.0, 50)
    y = np.ones(1)
    with pytest.raises(ValueError):
        seed_nlls(rhs, [0.4], t, y, [1.4, 5.0], names=["r", "K"])


def test_seed_nlls_raises_when_the_fit_does_not_converge():
    # a noiseless logistic recovered exactly with an unbounded solver;
    # fails if the `if not result.success: raise RuntimeError` guard is
    # dropped, since max_nfev=1 stops least_squares before convergence.
    r, k, y0 = 1.4, 5.0, 0.4
    t = np.linspace(0.0, 8.0, 50)
    y = solve_ivp(lambda _t, yy: r * yy * (1 - yy / k), (0, 8), [y0],
                  t_eval=t, rtol=1e-9, atol=1e-11).y[0]

    def rhs(_t, state, rr, kk):
        return [rr * state[0] * (1 - state[0] / kk)]

    with pytest.raises(RuntimeError):
        seed_nlls(rhs, [y0], t, y, [r, k], names=["r", "K"], max_nfev=1)
