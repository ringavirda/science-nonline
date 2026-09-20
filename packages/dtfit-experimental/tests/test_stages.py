"""The batch-stage adapters, streaming-filter factories and map-reduce
accumulators of the evolution matrix.

Each test names the mutation of stages.py it fails under.
"""

import numpy as np

from dtfit_experimental.study import problems, stages
from dtfit_experimental.study.montecarlo import grid, noisy

_KEYS = (
    "S0_nlls", "S1_balance_book", "S2_balance_adequate", "S3a_lsi_monomial",
    "S3b_lsi_legendre", "S3c_eac_book", "S3d_eac_direct",
    "S4a_image_legendre", "S4b_image_block", "S4r_image_robust",
)
_NEEDS_P0 = {
    "S0_nlls": True, "S1_balance_book": False, "S2_balance_adequate": False,
    "S3a_lsi_monomial": True, "S3b_lsi_legendre": True, "S3c_eac_book": True,
    "S3d_eac_direct": True, "S4a_image_legendre": True,
    "S4b_image_block": True, "S4r_image_robust": True,
}


def _draw(family, *, noise=0.05, n=200, seed=0):
    expr, truth, half_width = problems.PROBLEMS[family]
    names, tv = problems.truth_vector(expr, truth)
    model = problems.model(expr, names)
    rng = np.random.default_rng(seed)
    x = grid("uniform", n, span=half_width)
    clean = model(x, *tv)
    y, _ = noisy(clean, noise, rng)
    return x, y, expr, model, tv


def _by_key(key):
    return next(s for s in stages.BATCH_STAGES if s.key == key)


def test_batch_stages_keys_labels_and_needs_p0():
    assert tuple(s.key for s in stages.BATCH_STAGES) == _KEYS
    for s in stages.BATCH_STAGES:
        assert isinstance(s.label, str) and s.label
        # fails if a stage's needs_p0 flag were swapped
        assert s.needs_p0 == _NEEDS_P0[s.key]


def test_stage_result_carries_converged_false_on_silent_failure():
    # this draw's dsb_balance returns converged=False
    x, y, expr, model, tv = _draw("logistic")
    result = _by_key("S1_balance_book").run(x, y, expr, model, tv * 1.25)
    assert result.status == "ok"
    # fails if the adapter discarded FittingResult.converged
    assert result.converged is False


def test_filters_has_the_six_keys():
    assert sorted(stages.FILTERS) == [
        "image_block", "image_legendre", "image_legendre_adaptive",
        "recursive_area", "recursive_spectrum", "recursive_subareas",
    ]


def test_classify_complex_infinity_is_not_applicable():
    exc = KeyError("ComplexInfinity")
    # classification of ComplexInfinity does not depend on the stage key
    status, message = stages._classify(exc, "S3a_lsi_monomial")
    assert status == stages.NOT_APPLICABLE
    assert message == str(exc)


def test_classify_underdetermined_balance_is_not_applicable_on_s1():
    exc = ValueError(
        "Only 2 of the first 4 Maclaurin orders constrain the 3 "
        "parameters; increase the polynomial degree (rank) so the "
        "balance can identify them.")
    status, _ = stages._classify(exc, "S1_balance_book")
    assert status == stages.NOT_APPLICABLE


def test_classify_underdetermined_balance_is_failed_on_s2():
    # fails if the underdetermined-balance regex were classified the same
    # regardless of stage
    exc = ValueError(
        "Only 2 of the first 4 Maclaurin orders constrain the 3 "
        "parameters; increase the polynomial degree (rank) so the "
        "balance can identify them.")
    status, _ = stages._classify(exc, "S2_balance_adequate")
    assert status == "failed"


def test_classify_rank_below_parameters_is_failed_not_not_applicable():
    # fails if the underdetermined-balance regex matched any message
    # containing "parameters" instead of the specific sentence
    exc = ValueError("rank=3 is below the 5 parameters; balance "
                     "underdefined.")
    status, _ = stages._classify(exc, "S1_balance_book")
    assert status == "failed"


def test_classify_generic_exception_is_failed_with_class_name():
    # fails if the classifier were widened to call every exception
    # "not applicable"
    status, message = stages._classify(RuntimeError("boom"), "S0_nlls")
    assert status == "failed"
    assert message == "RuntimeError"


def test_s2_balance_adequate_underdetermined_draw_is_failed_not_na():
    # arctan, seed 1, noise 0.30: S2's own fit_dsb raises the
    # underdetermined-balance ValueError here
    x, y, expr, model, tv = _draw("arctan", noise=0.30, seed=1)
    result = _by_key("S2_balance_adequate").run(x, y, expr, model, tv * 1.25)
    assert result.status == "failed"
    assert result.message == "ValueError"


def test_runtime_error_from_the_inner_call_wraps_to_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")
    monkeypatch.setattr(stages, "curve_fit", boom)
    x, y, expr, model, tv = _draw("exp_decay")
    result = _by_key("S0_nlls").run(x, y, expr, model, tv * 1.25)
    assert result.status == "failed"
    assert result.message == "RuntimeError"
    assert result.params is None
    assert result.cov is None


def test_power_law_source_stages_not_applicable_image_stages_ok():
    x, y, expr, model, tv = _draw("power_law")
    p0 = tv * 1.25
    for key in ("S1_balance_book", "S2_balance_adequate",
               "S3a_lsi_monomial", "S3c_eac_book"):
        result = _by_key(key).run(x, y, expr, model, p0)
        assert result.status == stages.NOT_APPLICABLE, key
    for key in ("S3d_eac_direct", "S4a_image_legendre", "S4b_image_block",
               "S4r_image_robust"):
        result = _by_key(key).run(x, y, expr, model, p0)
        assert result.status == "ok", key


def test_s1_balance_book_not_applicable_on_arctan_ok_on_exp_decay():
    x, y, expr, model, tv = _draw("arctan")
    result = _by_key("S1_balance_book").run(x, y, expr, model, tv * 1.25)
    assert result.status == stages.NOT_APPLICABLE

    x, y, expr, model, tv = _draw("exp_decay")
    result = _by_key("S1_balance_book").run(x, y, expr, model, tv * 1.25)
    assert result.status == "ok"


def test_every_batch_stage_finite_and_right_length_on_exp_decay():
    x, y, expr, model, tv = _draw("exp_decay")
    p0 = tv * 1.25
    for s in stages.BATCH_STAGES:
        result = s.run(x, y, expr, model, p0)
        # fails if an adapter were wired to (y, x, ...)
        assert result.status == "ok", s.key
        assert result.params.shape == tv.shape, s.key
        assert np.all(np.isfinite(result.params)), s.key


#: The parameter values the streaming and cost matrix compares directly:
#: n_sub for the two recursive-area forms, order for the spectrum and
#: image forms, basis/adaptive_window for the three image forms.
_FILTER_PARAMS = {
    "recursive_area": {"n_sub": 1},
    "recursive_subareas": {"n_sub": 4},
    "recursive_spectrum": {"order": 5},
    "image_legendre": {
        "order": 5, "basis": "legendre", "adaptive_window": False},
    "image_block": {"order": 2, "basis": "block", "adaptive_window": False},
    "image_legendre_adaptive": {
        "order": 5, "basis": "legendre", "adaptive_window": True},
}


def test_filters_build_and_partial_fit():
    for key, make in stages.FILTERS.items():
        flt = make(window_size=30)
        flt.partial_fit(0.0, 0.0)
        assert hasattr(flt, "params_")
        for attr, expected in _FILTER_PARAMS[key].items():
            got = flt.basis.name if attr == "basis" else getattr(flt, attr)
            # fails if a filter's pinned parameter drifted from _FILTER_PARAMS
            assert got == expected, (key, attr)


def test_accumulators_build_and_share_update_merge_fit():
    domain = (0.0, 10.0)
    for key, make in stages.ACCUMULATORS.items():
        kwargs = {"domain": domain, "order": 4}
        if key == "integral_eac":
            kwargs = {"domain": domain, "n_windows": 4}
        elif key == "integral_batch_lsi":
            kwargs["n_channels"] = 2
        acc = make("a*exp(b*t)", "t", **kwargs)
        assert hasattr(acc, "update")
        assert hasattr(acc, "merge")
        assert hasattr(acc, "fit")


def test_image_accumulator_merge_reproduces_whole_fit():
    domain = (0.0, 10.0)
    rng = np.random.default_rng(0)
    t = np.linspace(*domain, 400)
    y = 3.0 * np.exp(-0.4 * t) + 0.01 * rng.standard_normal(t.size)
    whole = stages.ImageAccumulator("a*exp(b*t)", "t", domain=domain)
    whole.update(t, y)
    parts = stages.ImageAccumulator("a*exp(b*t)", "t", domain=domain)
    for i in range(0, t.size, 100):
        parts.update(t[i:i + 100], y[i:i + 100])
    c_whole = whole.fit(p0=[3.0, -0.4]).coeffs
    c_parts = parts.fit(p0=[3.0, -0.4]).coeffs
    # fails if merge() dropped a chunk or reduced order-dependently
    assert np.max(np.abs(c_parts / c_whole - 1.0)) < 1e-9


def test_image_accumulator_merge_reduces_two_accumulators():
    domain = (0.0, 10.0)
    rng = np.random.default_rng(0)
    t = np.linspace(*domain, 400)
    y = 3.0 * np.exp(-0.4 * t) + 0.01 * rng.standard_normal(t.size)
    whole = stages.ImageAccumulator("a*exp(b*t)", "t", domain=domain)
    whole.update(t, y)
    half = t.size // 2
    a = stages.ImageAccumulator("a*exp(b*t)", "t", domain=domain)
    a.update(t[:half], y[:half])
    b = stages.ImageAccumulator("a*exp(b*t)", "t", domain=domain)
    b.update(t[half:], y[half:])
    reduced = a.merge(b)
    assert reduced is a
    c_whole = whole.fit(p0=[3.0, -0.4]).coeffs
    c_reduced = reduced.fit(p0=[3.0, -0.4]).coeffs
    # fails if merge() were replaced with a no-op (`return self`)
    assert np.max(np.abs(c_reduced / c_whole - 1.0)) < 1e-9
