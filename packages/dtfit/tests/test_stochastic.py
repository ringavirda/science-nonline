"""Tests for the promoted ``dtfit.stochastic`` package -- the stochastic-series
solution (fit deterministic functionals of a random process to characterize /
forecast / generate it, plus the streaming filter).

Self-contained: ground-truth generators are defined here so stable ``dtfit`` does
not depend on the experimental experiment harness.
"""

from typing import cast

import numpy as np
import pytest

from dtfit import fit_stochastic, StochasticModel, StochasticFilter, Stochastic
from dtfit.stochastic import (
    sample_acf, hurst_spectral, hurst_aggvar, ar1_reversion, garch_persistence,
    cycle_period, decompose_trend_cycle, FORECASTERS,
    ar_order, fit_ar, fractional_difference,
)


# --- ground-truth generators ------------------------------------------------ #
def gen_ar1(n, phi, rng, sigma=1.0, burn=200):
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def gen_arfima(n, d, rng, ntrunc=1200):
    psi = np.empty(ntrunc)
    psi[0] = 1.0
    for j in range(1, ntrunc):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    e = rng.standard_normal(n + ntrunc)
    return np.convolve(e, psi)[ntrunc:ntrunc + n]


def gen_garch(n, omega, alpha, beta, rng, burn=500):
    N = n + burn
    z = rng.standard_normal(N)
    s2 = np.empty(N)
    r = np.empty(N)
    s2[0] = omega / max(1e-9, 1.0 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, N):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:]


def gen_ar2_cycle(n, period, damping, rng, burn=300):
    phi1 = 2.0 * damping * np.cos(2.0 * np.pi / period)
    phi2 = -(damping ** 2)
    N = n + burn
    e = rng.standard_normal(N)
    x = np.zeros(N)
    for t in range(2, N):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x[burn:]


def gen_trend_cycle(n, slope, period, amp, noise_sd, rng):
    t = np.arange(n, dtype=float)
    y = (slope * t + amp * np.sin(2.0 * np.pi * t / period)
         + rng.normal(0.0, noise_sd, n))
    return t, y


# --- the functional estimators (each feeds fit_lsi/fit_eac) ----------------- #
def test_sample_acf_white_noise_is_a_spike():
    acf = sample_acf(np.random.default_rng(0).standard_normal(4000), 12)
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:]) < 0.12)


def test_hurst_spectral_recovers_long_memory():
    H = 0.8
    err = np.mean([abs(hurst_spectral(gen_arfima(4096, 0.3,
                  np.random.default_rng(10 + s)))["H"] - H) for s in range(3)])
    assert err < 0.15


def _gen_ar2(n, p1, p2, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = p1 * x[t - 1] + p2 * x[t - 2] + rng.standard_normal()
    return x


def test_fit_ar_recovers_ar2_order_and_coeffs():
    orders = [ar_order(_gen_ar2(2000, 0.5, 0.3, s)) for s in range(7)]
    assert max(set(orders), key=orders.count) == 2      # AR(2) selected
    fit = fit_ar(_gen_ar2(4000, 0.5, 0.3, 1))
    assert fit["order"] == 2
    assert np.allclose(np.asarray(fit["phi"], dtype=float), [0.5, 0.3], atol=0.08)


def test_ar_order_white_noise_is_zero():
    assert ar_order(np.random.default_rng(0).standard_normal(2000)) == 0


def test_ar_p_not_mislabeled_long_memory():
    """The router classifies a genuine finite-order AR as mean-reverting, not
    long memory (the AR(p) veto), at any order including a near-unit-root AR(1)."""
    for series in (_gen_ar2(3000, 0.5, 0.3, 0),        # AR(2)
                   gen_ar1(3000, 0.95, np.random.default_rng(0))):  # near-unit AR(1)
        m = fit_stochastic(series)
        assert not m.has_long_memory
        assert m.has_mean_reversion


def test_strong_long_memory_still_detected():
    """A strong ARFIMA (d=0.4, H~0.9) is detected as long memory -- reading the
    raw-residual Hurst no longer under-detects it (the AR(1)-whitened innovations
    used to)."""
    got = sum(fit_stochastic(gen_arfima(4096, 0.4, np.random.default_rng(s)))
              .has_long_memory for s in range(5))
    assert got >= 4


def test_fractional_difference_special_cases():
    x = np.cumsum(np.random.default_rng(0).standard_normal(64))
    assert np.allclose(fractional_difference(x, 0.0), x)          # identity
    assert np.allclose(fractional_difference(x, 1.0)[1:], np.diff(x))  # first diff


def test_fractional_difference_whitens_long_memory():
    # differencing an ARFIMA(0,d,0) by its own d whitens it (Hurst -> ~0.5)
    y = gen_arfima(4096, 0.3, np.random.default_rng(2))
    d = hurst_spectral(y)["d"]
    w = fractional_difference(y, d)
    assert abs(hurst_spectral(w)["H"] - 0.5) < abs(hurst_spectral(y)["H"] - 0.5)


def test_simulate_student_t_has_fat_tails_and_unit_scale():
    from scipy.stats import kurtosis
    x = gen_ar1(3000, 0.7, np.random.default_rng(1))
    m = fit_stochastic(x)
    sn = m.simulate(5000, seed=0, dist="normal")
    st = m.simulate(5000, seed=0, dist="t", df=4)
    assert float(kurtosis(st)) > float(kurtosis(sn)) + 1.5     # fatter tails
    assert abs(np.std(st) - np.std(sn)) < 0.25 * np.std(sn)    # same scale


def test_long_memory_simulate_variance_matches_sigma():
    from dtfit.stochastic._simulate import _sim_long_memory
    for H in (0.7, 0.9):
        stds = [np.std(_sim_long_memory(2000, H, 1.0, np.random.default_rng(s)))
                for s in range(6)]
        assert abs(float(np.mean(stds)) - 1.0) < 0.1          # realized std ~ sigma


def test_hurst_aggvar_recovers_long_memory():
    # aggregated-variance Hurst is a stable-package public export; cover it here
    # (previously only exercised via the experimental suite). Noisier than the
    # spectral estimator, so a looser band + more seeds.
    H = 0.8
    ests = [hurst_aggvar(gen_arfima(4096, 0.3,
            np.random.default_rng(60 + s)))["H"] for s in range(5)]
    assert abs(float(np.mean(ests)) - H) < 0.2
    assert all(0.0 <= e <= 1.0 for e in ests)


def test_hurst_aggvar_eac_matches_loglog():
    # the equal-areas (linear-space power-law) branch must recover the same
    # long-memory slope as the log-log fit -- a regression guard for the p0/bounds
    # ordering (it previously pinned H at 1.0 from swapped seed/bound order).
    x = gen_arfima(4096, 0.3, np.random.default_rng(77))
    h_eac = hurst_aggvar(x, method="eac")["H"]
    h_lsi = hurst_aggvar(x, method="lsi")["H"]
    assert 0.0 < h_eac < 1.0 and abs(h_eac - h_lsi) < 0.15


@pytest.mark.parametrize("phi", [0.6, 0.9])
def test_ar1_reversion_recovers_phi(phi):
    est = np.mean([ar1_reversion(gen_ar1(1500, phi,
                  np.random.default_rng(30 + s)))["phi"] for s in range(3)])
    assert abs(est - phi) / phi < 0.12


def test_garch_persistence_recovers_alpha_plus_beta():
    truth = 0.08 + 0.90
    est = np.mean([garch_persistence(gen_garch(4000, 0.05, 0.08, 0.90,
                  np.random.default_rng(40 + s)), use="abs")["persistence"]
                  for s in range(3)])
    assert abs(est - truth) / truth < 0.20


def test_cycle_period_recovers_period():
    P = 16.0
    est = np.mean([cycle_period(gen_ar2_cycle(1500, P, 0.97,
                  np.random.default_rng(50 + s)))["period"] for s in range(3)])
    assert abs(est - P) / P < 0.15


def test_decompose_recovers_trend_and_cycle():
    t, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(0))
    dec = decompose_trend_cycle(t, y, trend_deg=1)
    period, slope = cast(float, dec["period"]), cast(float, dec["slope"])
    assert abs(period - 50.0) / 50.0 < 0.10
    assert abs(slope - 0.02) / 0.02 < 0.30


# --- the merged batch solution: fit_stochastic + Stochastic().fit() --------- #
def test_merged_white_noise_reports_no_structure():
    m = fit_stochastic(np.random.default_rng(0).standard_normal(1500))
    assert isinstance(m, StochasticModel)
    assert m.regime.startswith("white noise") and m.components == ("none",)


def test_merged_random_walk_detected_as_unit_root():
    m = fit_stochastic(np.cumsum(np.random.default_rng(1).standard_normal(1500)))
    assert m.regime.startswith("random walk") and not m.has_trend


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_merged_ar1_is_mean_reverting(phi):
    m = fit_stochastic(gen_ar1(1500, phi, np.random.default_rng(2)))
    assert m.has_mean_reversion and abs(m.ar1_phi - phi) / phi < 0.2


def test_merged_arfima_is_long_memory():
    m = fit_stochastic(gen_arfima(4096, 0.3, np.random.default_rng(3)))
    assert m.has_long_memory and m.hurst > 0.65


def test_merged_trend_cycle_detected_and_forecasts():
    _, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    m = fit_stochastic(y)
    assert m.has_trend and m.has_cycle
    pt, lo, hi = m.forecast(30, return_conf_int=True)
    assert pt.shape == (30,) and np.all(hi >= lo) and np.all(np.isfinite(pt))


def test_stochastic_model_convention_fit():
    """`dtfit.Stochastic().fit(series)` returns the same StochasticModel."""
    y = gen_ar1(1200, 0.7, np.random.default_rng(1))
    s = Stochastic()
    m = s.fit(y)
    assert isinstance(m, StochasticModel) and m is s.model_
    assert m.has_mean_reversion
    # explicit time index form fit(t, series)
    t = np.arange(y.size, dtype=float)
    assert Stochastic().fit(t, y).regime == m.regime
    # a forced forecaster / detection gate flows through
    assert Stochastic(forecaster="drift").fit(y).forecaster_name == "drift"


def test_forecaster_control():
    y = gen_ar1(800, 0.6, np.random.default_rng(1))
    assert fit_stochastic(y).forecaster_name in set(FORECASTERS) | {"custom"}
    assert fit_stochastic(y, forecaster="drift").forecaster_name == "drift"
    with pytest.raises(ValueError):
        fit_stochastic(y, forecaster="nonsense")


# --- the generative model: simulate round-trip ------------------------------ #
@pytest.mark.parametrize("gen,attr", [
    (lambda r: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, r)[1], "has_cycle"),
    (lambda r: gen_ar1(1500, 0.7, r), "has_mean_reversion"),
])
def test_simulate_round_trip_recovers_regime(gen, attr):
    hits = sum(bool(getattr(fit_stochastic(
        fit_stochastic(gen(np.random.default_rng(10 + s))).simulate(seed=s)),
        attr)) for s in range(5))
    assert hits >= 4


def test_simulate_reproducible_and_finite():
    m = fit_stochastic(gen_ar1(800, 0.6, np.random.default_rng(0)))
    a, b = m.simulate(seed=7), m.simulate(seed=7)
    assert a.shape == (m.n,) and np.all(np.isfinite(a)) and np.allclose(a, b)


# --- the streaming filter --------------------------------------------------- #
def test_filter_tracks_ar1_phi_online():
    for phi in (0.3, 0.6, 0.85):
        errs = [abs(StochasticFilter(halflife=300, warmup=100).partial_fit(
            gen_ar1(3000, phi, np.random.default_rng(s))).params_["ar1_phi"] - phi)
            for s in range(3)]
        assert np.mean(errs) < 0.08


def test_filter_detects_persistence_break_with_low_false_alarm():
    hits = 0
    for s in range(5):
        r = np.random.default_rng(10 + s)
        a = gen_ar1(1500, 0.2, r)
        b = gen_ar1(1500, 0.9, r)
        b = b - b.mean() + a[-1]
        f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
        f.partial_fit(np.concatenate([a, b]))
        if any(1500 <= t <= 1900 for t in f.flag_times_):
            hits += 1
    assert hits >= 4
    # stationary stream -> few false alarms
    fa = [StochasticFilter(warmup=80, settle=500, z_thresh=4.0).partial_fit(
        gen_ar1(3000, 0.6, np.random.default_rng(30 + s))).n_flags_
        for s in range(5)]
    assert np.mean(fa) <= 1.5


# --- the vendored statsmodels-free unit-root gate --------------------------- #
def test_unit_root_gate_verdicts_without_statsmodels():
    from dtfit.stochastic._model import _is_nonstationary
    rng = np.random.default_rng(0)
    assert _is_nonstationary(np.cumsum(rng.standard_normal(400)))
    assert not _is_nonstationary(0.05 * np.arange(400) + rng.standard_normal(400) * 3)
    assert not _is_nonstationary(gen_ar1(800, 0.6, rng))
    assert not _is_nonstationary(rng.standard_normal(400))


def test_vendored_adf_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    from dtfit.stochastic._stats import _adf_tau, _adf_pvalue
    for s in range(3):
        for x in (np.cumsum(np.random.default_rng(100 + s).standard_normal(400)),
                  gen_ar1(400, 0.9, np.random.default_rng(200 + s))):
            n = x.size
            maxlag = int(min(12 * (n / 100.0) ** 0.25, 12, n // 3))
            ref = sm.adfuller(x, regression="ct", maxlag=maxlag, autolag="AIC")
            tau = _adf_tau(x)
            assert abs(tau - ref[0]) < 1e-7
            assert abs(_adf_pvalue(tau) - ref[1]) < 1e-6
