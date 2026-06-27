"""dtfit.models: self-seeding catalog families, composition, and suggest_models."""

import numpy as np
import pytest

from dtfit import models, suggest_models, Model


def test_logistic_self_seeds_and_recovers():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 12, 200)
    y = 1000.0 / (1 + np.exp(-0.7 * (t - 6))) + rng.normal(0, 8, t.size)
    r = models.logistic().fit(t, y)  # no p0/bounds supplied
    assert r.params["L"] == pytest.approx(1000, rel=0.05)
    assert r.params["k"] == pytest.approx(0.7, rel=0.1)
    assert r.params["x0"] == pytest.approx(6.0, abs=0.3)


def test_oscillatory_family_recovers_frequency():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 300)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 0.05, t.size)
    r = models.sine().fit(t, y)
    assert r.params["w"] == pytest.approx(1.5, abs=0.1)


def test_model_carries_params_and_shape():
    m = models.damped_oscillation()
    assert m.shape == "oscillatory" and m.freq_param == "w"
    assert set(m.params) == {"A", "w", "z"}


def test_composition_builds_combined_expr():
    m = models.linear() + models.sine()
    # combined params are the union (linear a0,a1 + sine A,c,p,w), no collision
    assert {"a0", "a1", "A", "c", "p", "w"} <= set(m.params)
    assert m.shape == "composite"
    rng = np.random.default_rng(2)
    x = np.linspace(0, 20, 300)
    y = 0.5 * x + 3 * np.sin(2 * np.pi * x / 4) + rng.normal(0, 0.2, x.size)
    r = m.fit(x, y)
    # the cycle is seeded on the detrended residual -> a genuinely good fit
    from dtfit.diagnostics import fit_report
    assert fit_report(r, x, y)["r2"] > 0.95


def test_composition_renames_colliding_params():
    m = models.gaussian() + models.gaussian()  # both have A, mu, s
    # second component's params are renamed, so all are distinct
    assert len(m.params) == 6


def test_suggest_models_ranks_true_family_first():
    rng = np.random.default_rng(3)
    t = np.linspace(0.5, 6, 200)
    y = 2.0 * np.exp(0.5 * t) + rng.normal(0, 1.0, t.size)
    ranked = suggest_models(t, y, top=3)
    assert ranked, "no suggestions returned"
    names = [s.name for s in ranked]
    # the exponential should be the most parsimonious good fit -> top of the list
    assert "exponential" in names[:2]
    assert ranked[0].r2 > 0.95


def test_suggest_models_custom_candidates():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 10, 150)
    y = 3.0 / (1 + np.exp(-0.8 * (t - 5))) + rng.normal(0, 0.05, t.size)
    ranked = suggest_models(t, y, candidates=[models.logistic(), models.linear()])
    assert ranked[0].name == "logistic"


def test_suggest_models_include_exclude_filter():
    rng = np.random.default_rng(3)
    t = np.linspace(0.5, 6, 200)
    y = 2.0 * np.exp(0.5 * t) + rng.normal(0, 1.0, t.size)
    # exclude by category prunes the whole 'growth' family from the shortlist
    names = [s.name for s in suggest_models(t, y, exclude=["growth"])]
    assert "exponential" not in names and "exp_growth_offset" not in names
    # exclude by name drops just that family
    names = [s.name for s in suggest_models(t, y, exclude=["exponential"])]
    assert "exponential" not in names
    # include restricts the search to the named families only
    names = [s.name for s in suggest_models(t, y, include=["linear", "quadratic"])]
    assert set(names) <= {"linear", "quadratic"}


@pytest.mark.parametrize("name,fn,kw", [
    ("first_order", lambda t: 3.0 * (1 - np.exp(-t / 1.2)), {}),
    ("hill", lambda t: 5.0 * t**2 / (3.0**2 + t**2), {}),
    ("weibull_cdf", lambda t: 2.0 * (1 - np.exp(-(t / 3.0) ** 1.5)), {}),
    ("logarithmic", lambda t: 1.0 + 2.0 * np.log(t + 1), {}),
])
def test_expanded_families_self_seed_and_fit(name, fn, kw):
    rng = np.random.default_rng(7)
    t = np.linspace(0.1, 10, 200)
    y = fn(t) + rng.normal(0, 0.02 * (np.ptp(fn(t)) + 1e-9), t.size)
    r = getattr(models, name)().fit(t, y, **kw)
    from dtfit.diagnostics import fit_report
    assert fit_report(r, t, y)["r2"] > 0.97


def test_fourier_series_fits_harmonics():
    rng = np.random.default_rng(8)
    t = np.linspace(0, 4 * np.pi, 400)
    w = 1.0
    y = (np.sin(w * t) + 0.4 * np.sin(3 * w * t) + 0.2 * np.cos(2 * w * t)
         + rng.normal(0, 0.03, t.size))
    r = models.fourier_series(3).fit(t, y)
    from dtfit.diagnostics import fit_report
    assert fit_report(r, t, y)["r2"] > 0.97


def test_catalog_is_comprehensive():
    # every catalogued family parses and carries a category tag
    cats = {m.category for m in models.all_models()}
    assert {"trend", "growth", "decay", "sigmoid", "saturating",
            "peak", "oscillatory"} <= cats
    assert len(models.CATALOG) >= 20


def test_suggest_shortlist_prunes_by_shape():
    from dtfit.models._suggest import _shortlist
    rng = np.random.default_rng(9)
    t = np.linspace(0, 6 * np.pi, 300)
    y = np.sin(2 * t) + rng.normal(0, 0.05, t.size)
    short = {m.name for m in _shortlist(t, y)}
    # oscillatory data: keep the cycles, drop the peak/sigmoid families
    assert "sine" in short
    assert "gaussian" not in short and "logistic" not in short
    assert len(short) < len(models.CATALOG)


def test_model_from_raw_expr():
    m = Model("a*exp(b*x)", "x", name="exp", shape="bulk")
    rng = np.random.default_rng(5)
    x = np.linspace(0, 3, 200)
    y = 1.5 * np.exp(0.6 * x) + rng.normal(0, 0.02, x.size)
    r = m.fit(x, y, p0=[1.0, 1.0])
    assert r.params["b"] == pytest.approx(0.6, abs=0.1)
