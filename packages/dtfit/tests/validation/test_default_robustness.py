"""Phase-2 default-config robustness: the bare public entry points must work.

The docs and notebooks call the library with *defaults* -- ``models.x().fit``,
``NonlineRegressor(expr).fit``, ``suggest_models(x, y)``, ``fit_lsi(x, y, expr,
var)`` with no hand-tuned ``p0`` -- yet the historical tests always fed a good
seed and a method-favourable dataset. These tests exercise the default paths
across the catalogue and require they either produce a finite, usable fit or
fail loudly; nothing may silently return NaN or a stalled seed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dtfit as dt
from accuracy.scenarios import SCENARIOS
from accuracy.harness import ordered_params, r2, param_err, predict

# fourier_series is intentionally outside the default suggest_models sweep (a
# parametric factory offered separately), so a periodic signal is expected to
# surface its fundamental `sine` instead -- not a miss.
_SUGGEST_CASES = [s for s in SCENARIOS if s.factory_name != "fourier_series"]


@pytest.mark.parametrize("scn", _SUGGEST_CASES, ids=[s.name for s in _SUGGEST_CASES])
def test_suggest_recommends_true_family(scn):
    """The recommender must shortlist + rank the true family in the top 3.

    Guards the shape-detection regression where noisy sigmoids / saturating
    curves were tagged oscillatory and their families dropped, so an epidemic
    (logistic) curve came back recommended as `sine`.
    """
    x, y, _ = scn.make(0.03, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = [s.name for s in dt.suggest_models(x, y, top=3)]
    assert scn.factory_name in names or scn.name in names, (
        f"{scn.name}: true family not in top-3 {names}. {scn.note}")


# A representative spread (one per non-oscillatory category) fit through the
# sklearn estimator with *all* defaults (p0=None -> ones, k_star=5,
# filter_data=True). Oscillatory models are deliberately excluded: the bare
# NonlineRegressor does not auto-apply the oscillatory recipe (it has no
# freq_param), so a cycle must be fit via Model.fit / auto_estimate or by
# passing freq_param -- see test_oscillatory_needs_recipe_path and the LSI docs.
_REGRESSOR_MODELS = [
    "linear", "exponential", "exp_decay", "logistic", "gaussian",
    "michaelis_menten",
]


@pytest.mark.parametrize("name", _REGRESSOR_MODELS)
@pytest.mark.parametrize("method", ["lsi", "eda"])
def test_nonline_regressor_defaults(name, method):
    scn = next(s for s in SCENARIOS if s.name == name)
    x, y, clean = scn.make(0.03, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = dt.NonlineRegressor(scn.model().expr, scn.model().var,
                                  method=method).fit(x, y)
    pred = reg.predict(x)
    assert np.all(np.isfinite(reg.coef_))
    assert np.all(np.isfinite(pred))
    # A bare-default fit on a clean signal should still be a usable curve.
    assert r2(clean, pred) > 0.9, f"{name}/{method}: R2={r2(clean, pred):.3f}"


def test_oscillatory_needs_recipe_path():
    """Documents a real usage limit: a cycle is only recovered through the
    oscillatory recipe (Model.fit / auto_estimate / freq_param), *not* the bare
    NonlineRegressor whose smoothing + low default order erase the cycle."""
    scn = next(s for s in SCENARIOS if s.name == "sine")
    x, y, clean = scn.make(0.03, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # recommended path: routes through the oscillatory recipe -> recovers
        good = scn.model().fit(x, y)
        good_r2 = r2(clean, predict(good, x))
        # bare regressor (no recipe): a documented underperformer on cycles
        bare = dt.NonlineRegressor(scn.model().expr, "x", method="lsi").fit(x, y)
        bare_r2 = r2(clean, bare.predict(x))
    assert good_r2 > 0.98, f"recipe path should recover the cycle, got {good_r2:.3f}"
    assert good_r2 > bare_r2, "oscillatory recipe must beat the bare regressor"


def test_dsb_regressor_additive():
    """DSB through the estimator on its intended additive form, defaults only."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 150)
    clean = 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x)
    y = clean + rng.normal(0, 0.03, x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = dt.NonlineRegressor("a0 + a1*x + a2*exp(a3*x)", "x",
                                  method="dsb").fit(x, y)
    pred = reg.predict(x)
    assert np.all(np.isfinite(pred))
    # DSB is the analytical reference: under noise it matches the data's noisy
    # high-order polynomial spectrum, so it behaves as a curve fit, not an exact
    # point estimator (see dsb.md). A usable curve, not near-perfect recovery.
    assert r2(clean, pred) > 0.90


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_self_seeded_fit_is_accurate(scn):
    """The headline doc path -- ``models.family().fit(x, y)`` self-seeded -- must
    recover the truth (params families) or fit the curve (weak-id families) on a
    lightly-noised signal, with no hand-written p0."""
    x, y, clean = scn.make(0.02, seed=7)
    names = ordered_params(scn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scn.model().fit(x, y)
    pred = predict(res, x)
    assert np.all(np.isfinite(res.coeffs)) and np.all(np.isfinite(pred))
    if scn.metric == "params":
        assert param_err(scn, names, res.coeffs) <= 0.15, scn.note
    else:
        assert r2(clean, pred) >= 0.98, scn.note
