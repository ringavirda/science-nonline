"""Regression guards for the correctness / capability hardening pass.

Each test pins an invariant that a specific fix established, so the fix cannot
silently regress:

* covariance stays symmetric & PD over a long stream (streaming Joseph-ish
  symmetrization);
* the SVD covariance is finite for ill-conditioned Jacobians;
* ``fit_eac(loss="soft_l1")`` actually engages robustness (auto ``f_scale``);
* curvature placement never crashes on short/smooth data;
* a one-member ensemble does not report zero uncertainty;
* wrong-length ``p0`` raises a clear error;
* ``PartitionedEAC.merge`` is exactly additive / order-independent;
* the model recommender shortlists cycles under a trend;
* trend/seasonal forecast bands do not fan out like a random walk;
* ``nan_policy="omit"`` fits gappy data;
* ``coast_cov`` grows with gap length;
* ``Model.fit(method="adaptive")`` forwards seeded bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from dtfit import (
    fit_eac, fit_lsi, ensemble_fit, LSIFilter, EACFilter, fit_stochastic,
    suggest_models,
)
from dtfit.scale import PartitionedEAC


# --------------------------------------------------------------------------- #
# streaming covariance: symmetric & positive-semidefinite over a long stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Filter", [LSIFilter, EACFilter])
def test_streaming_covariance_stays_symmetric_and_psd(Filter):
    rng = np.random.default_rng(0)
    flt = Filter.tracking("a + b*t + c*t**2", "t")
    for t in np.linspace(0, 20, 2500):
        flt.partial_fit(float(t), 1.0 + 0.3 * t - 0.05 * t**2
                        + 0.02 * rng.standard_normal())
    P = flt.P
    assert np.allclose(P, P.T, atol=1e-9), "covariance drifted asymmetric"
    eig = np.linalg.eigvalsh(0.5 * (P + P.T))
    assert eig.min() >= -1e-9, f"covariance lost PD: min eig {eig.min():.2e}"
    # stderr_ is therefore real & finite
    assert all(np.isfinite(v) for v in flt.stderr_.values())


# --------------------------------------------------------------------------- #
# covariance from the SVD is finite even for an ill-conditioned Jacobian
# --------------------------------------------------------------------------- #
def test_svd_covariance_finite_for_illconditioned_model():
    # a, b nearly trade off (a*exp(b*t) with tiny curvature over a short span)
    x = np.linspace(0.0, 0.05, 60)
    y = 2.0 * np.exp(0.1 * x) + 1e-4 * np.random.default_rng(1).standard_normal(x.size)
    r = fit_lsi(x, y, "a*exp(b*t)", "t")
    if r.cov is not None:
        assert np.all(np.isfinite(r.cov)), "ill-conditioned covariance not finite"


# --------------------------------------------------------------------------- #
# robust loss actually engages (auto f_scale) vs the old inert default
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fitter", [fit_eac, fit_lsi])
def test_robust_integral_beats_nonrobust_under_dense_outliers(fitter):
    """`robust=True` (per-sample IRLS winsorization of the integrand) recovers
    parameters under dense per-sample outliers where the plain integral is
    dragged, and does not harm a clean fit."""
    rng = np.random.default_rng(3)
    x = np.linspace(0.1, 4.0, 240)
    true = (2.5, -0.6)
    base = true[0] * np.exp(true[1] * x)
    y = base + 0.02 * rng.standard_normal(x.size)
    idx = rng.choice(x.size, size=30, replace=False)
    y[idx] += rng.uniform(3, 6, size=idx.size)

    def relerr(r):
        return (abs(r.params["a"] - true[0]) / abs(true[0])
                + abs(r.params["b"] - true[1]) / abs(true[1]))

    plain = fitter(x, y, "a*exp(b*t)", "t")
    rob = fitter(x, y, "a*exp(b*t)", "t", robust=True)
    assert relerr(rob) < 0.1
    assert relerr(rob) < 0.3 * relerr(plain)
    # clean data: robust must not materially hurt
    yc = base + 0.02 * rng.standard_normal(x.size)
    assert relerr(fitter(x, yc, "a*exp(b*t)", "t", robust=True)) < 0.05


def test_soft_l1_beats_linear_under_outliers():
    rng = np.random.default_rng(3)
    x = np.linspace(0.1, 4.0, 240)
    true = (2.5, -0.6)
    y = true[0] * np.exp(true[1] * x) + 0.02 * rng.standard_normal(x.size)
    idx = rng.choice(x.size, size=24, replace=False)
    y[idx] += rng.uniform(3, 6, size=idx.size)

    def relerr(r):
        return (abs(r.params["a"] - true[0]) / abs(true[0])
                + abs(r.params["b"] - true[1]) / abs(true[1]))

    lin = fit_eac(x, y, "a*exp(b*t)", "t")
    rob = fit_eac(x, y, "a*exp(b*t)", "t", loss="soft_l1")  # auto f_scale
    assert relerr(rob) < 0.6 * relerr(lin), "robust loss did not engage"


# --------------------------------------------------------------------------- #
# curvature placement degrades gracefully on short / flat data (no crash)
# --------------------------------------------------------------------------- #
def test_curvature_falls_back_on_short_flat_data():
    x = np.linspace(0.0, 1.0, 9)
    y = 1.0 + 0.5 * x  # nearly flat -> curvature edges may collapse
    r = fit_eac(x, y, "a + b*t", "t", window_mode="curvature")
    assert np.all(np.isfinite(r.coeffs))


# --------------------------------------------------------------------------- #
# one-member ensemble must not report zero (overconfident) uncertainty
# --------------------------------------------------------------------------- #
def test_single_member_ensemble_not_overconfident():
    x = np.linspace(0.0, 3.0, 30)  # tiny record -> at most one usable window
    y = 2.0 * np.exp(-0.5 * x) + 0.01 * np.random.default_rng(4).standard_normal(x.size)
    res = ensemble_fit(x, y, "a*exp(b*t)", "t", n_windows=8)
    if res.members.shape[0] < 2:
        se = res.stderr()
        # not all-zero: either the analytic fallback covariance or NaN, never a
        # fabricated 0.0 that reads as perfect certainty.
        assert not all(v == 0.0 for v in se.values())


# --------------------------------------------------------------------------- #
# wrong-length p0 raises a clear, early error
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fitter", [fit_lsi, fit_eac])
def test_wrong_length_p0_raises(fitter):
    x = np.linspace(0.1, 3.0, 60)
    y = 2.0 * np.exp(-0.4 * x)
    with pytest.raises(ValueError, match="p0 must have length"):
        fitter(x, y, "a*exp(b*t)", "t", p0=[1.0, 2.0, 3.0])  # model has 2 params


# --------------------------------------------------------------------------- #
# PartitionedEAC.merge is exactly additive & order-independent
# --------------------------------------------------------------------------- #
def test_partitioned_eac_merge_is_associative():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 503))
    y = 2.0 * np.exp(-0.3 * x) + 0.05 * rng.standard_normal(x.size)

    whole = PartitionedEAC("a*exp(b*t)", "t", domain=(0, 10), n_windows=8)
    whole.update(x, y)

    for nchunks in (2, 3, 5, 10):
        idx = np.array_split(np.arange(x.size), nchunks)
        for reverse in (False, True):
            # rebuild fresh accumulators each pass (merge mutates in place)
            accs = []
            for ii in idx:
                a = PartitionedEAC("a*exp(b*t)", "t", domain=(0, 10), n_windows=8)
                a.update(x[ii], y[ii])
                accs.append(a)
            order = accs[::-1] if reverse else accs
            base = order[0]
            for a in order[1:]:
                base.merge(a)
            assert np.allclose(base._areas, whole._areas, atol=1e-12), (
                f"merge not additive for {nchunks} chunks (reverse={reverse})")


# --------------------------------------------------------------------------- #
# recommender shortlists a cycle riding a trend; a logistic stays non-cyclic
# --------------------------------------------------------------------------- #
def test_suggest_shortlists_cycle_under_trend():
    t = np.arange(240, dtype=float)
    rng = np.random.default_rng(0)
    y = 0.03 * t + 3.0 * np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(t.size)
    names = [s.model.name for s in suggest_models(t, y)]
    assert any("sin" in n or "oscill" in n or "damped" in n for n in names), names


# --------------------------------------------------------------------------- #
# trend/seasonal forecast bands do not grow like a random walk
# --------------------------------------------------------------------------- #
def test_trend_seasonal_forecast_bands_do_not_fan_out():
    t = np.arange(400, dtype=float)
    rng = np.random.default_rng(5)
    y = 0.02 * t + 2.0 * np.sin(2 * np.pi * t / 30) + 0.2 * rng.standard_normal(t.size)
    m = fit_stochastic(y)
    # deterministic-mean forecasters must give a ~flat band (not a RW/LM fan)
    if m.forecaster_name.startswith(("trend", "seasonal")):
        _, lo, hi = m.forecast(40, return_conf_int=True)
        width = hi - lo
        assert width[-1] <= 1.5 * width[0] + 1e-9, "bands fan out like a random walk"


# --------------------------------------------------------------------------- #
# nan_policy="omit" fits gappy data; default still raises
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fitter", [fit_lsi, fit_eac])
def test_nan_policy_omit(fitter):
    x = np.linspace(0.2, 3.0, 200)
    y = 2.3 * np.exp(-0.7 * x)
    y[::17] = np.nan
    with pytest.raises(ValueError):
        fitter(x, y, "a*exp(b*t)", "t")
    r = fitter(x, y, "a*exp(b*t)", "t", nan_policy="omit")
    assert abs(r.params["a"] - 2.3) < 0.1 and abs(r.params["b"] + 0.7) < 0.1


def test_regressor_coast_rolls_model_forward():
    """A fused (drift-poly + regressor) model can coast past the window using the
    supplied future regressor, reducing to predict at the anchor; and still raises
    when no future regressor is given."""
    f = LSIFilter("a + b*t + c*t**2 + k*acc", "t", regressors="acc",
                  order=4, p0=[0.0, 0.5, 0.2, 1.0])
    rng = np.random.default_rng(0)
    for t in np.linspace(0, 5, 200):
        acc = float(np.sin(t))
        y = 0.1 + 0.5 * t + 0.2 * t**2 + 1.0 * acc + 0.01 * rng.standard_normal()
        f.partial_fit(t, y, regressors={"acc": acc})
    a = f._t[-1]
    xs = np.array([a, a + 0.2, a + 0.5, a + 1.0])
    coasted = f.coast(xs, order=1, regressors={"acc": np.sin(xs)})
    assert np.all(np.isfinite(coasted))
    # drop-in at the anchor
    at_anchor = f.predict(np.array([a]), regressors={"acc": np.sin(a)})[0]
    assert abs(coasted[0] - at_anchor) < 1e-9
    # regressor genuinely enters the roll-forward: a different future regressor
    # gives a different coast
    other = f.coast(xs, order=1, regressors={"acc": np.sin(xs) + 5.0})
    assert not np.allclose(coasted[1:], other[1:])
    # without a future regressor it must raise (unknown future exogenous input)
    with pytest.raises(NotImplementedError):
        f.coast(xs)


def test_coast_cov_grows_with_gap():
    rng = np.random.default_rng(0)
    f = LSIFilter.tracking("a + b*t + c*t**2", "t", order=4)
    for t in np.linspace(0, 5, 120):
        f.partial_fit(float(t), 1.0 + 0.5 * t + 0.2 * t**2 + 0.01 * rng.standard_normal())
    a = f._t[-1]
    xs = np.array([a, a + 0.5, a + 1.0, a + 2.0, a + 4.0])
    cov = f.coast_cov(xs, order=1)
    assert np.all(np.diff(cov) >= -1e-9), "coast_cov must not shrink with gap"
    assert cov[-1] > cov[0], "coast_cov must grow across a gap"


# --------------------------------------------------------------------------- #
# Model.fit(method="adaptive") forwards the self-seeded bounds
# --------------------------------------------------------------------------- #
def test_model_adaptive_forwards_bounds(monkeypatch):
    import dtfit.models._model as _mod
    from dtfit.models._catalog import CATALOG

    seen = {}
    orig = _mod.fit_eac

    def spy(*args, **kwargs):
        seen.setdefault("bounds", []).append(kwargs.get("bounds"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(_mod, "fit_eac", spy)
    # a logistic self-seeds bounds; both eac and adaptive must forward them
    m = CATALOG["logistic"]()
    x = np.linspace(0.0, 10.0, 160)
    y = 5.0 / (1.0 + np.exp(-0.8 * (x - 5.0)))
    m.fit(x, y, method="eac")
    m.fit(x, y, method="adaptive")
    assert len(seen["bounds"]) == 2
    assert seen["bounds"][0] is not None, "eac path lost the seeded bounds"
    assert seen["bounds"][1] is not None, "adaptive path dropped the seeded bounds"
