"""The evolution matrix's own problem set: the eight families, the model
and truth builders, and the two streaming scenarios.

Each test names the mutation of problems.py it fails under.
"""

import numpy as np
import pytest
import sympy as sp

from dtfit_experimental.study import problems


def test_problems_has_eight_families():
    assert len(problems.PROBLEMS) == 8
    for expr, truth, half_width in problems.PROBLEMS.values():
        assert isinstance(expr, str)
        assert isinstance(truth, dict) and truth
        assert half_width > 0.0


def test_truth_vector_sorts_names_and_matches_truth():
    expr, truth, _ = problems.PROBLEMS["exp_decay"]
    names, tv = problems.truth_vector(expr, truth)
    # fails if truth_vector stopped sorting the free symbols by name
    assert names == ["a", "b"]
    assert np.array_equal(tv, [3.0, -1.2])


def test_model_matches_truth_to_1e12():
    expr, truth, _ = problems.PROBLEMS["exp_decay"]
    names, tv = problems.truth_vector(expr, truth)
    f = problems.model(expr, names)
    x = np.linspace(0.0, 2.0, 11)
    got = f(x, *tv)
    expected = truth["a"] * np.exp(truth["b"] * x)
    assert np.max(np.abs(got - expected)) < 1e-12


def test_model_broadcasts_a_constant_over_x():
    # fails if the "+ 0.0 * x" broadcast guard were dropped: a model with
    # no dependence on x would then return a bare scalar
    f = problems.model("a", ["a"])
    x = np.array([1.0, 2.0, 3.0])
    out = f(x, 5.0)
    assert out.shape == x.shape
    assert np.allclose(out, 5.0)


def test_stream_jump_has_the_amplitude_step_drift_does_not():
    t, y, amp = problems.stream("jump", seed=0, noise=0.0)
    assert t.shape == y.shape == amp.shape
    # fails if the step moved off t == 30 s, 300 samples in at DT=0.1
    assert amp[299] == 2.0
    assert amp[300] == 3.5
    assert amp[0] == 2.0
    assert amp[-1] == 3.5
    _, _, amp_drift = problems.stream("drift", seed=0, noise=0.0)
    assert amp_drift[0] == 2.0
    assert amp_drift[-1] == pytest.approx(3.5, abs=0.01)
    # fails if drift collapsed to the same step shape as jump
    assert not np.array_equal(amp, amp_drift)


def test_stream_seed_is_reproducible():
    _, y0, _ = problems.stream("jump", seed=3)
    _, y1, _ = problems.stream("jump", seed=3)
    # fails if stream() drew a fresh, unseeded generator each call
    assert np.array_equal(y0, y1)


def test_stream_unknown_kind_raises_keyerror():
    with pytest.raises(KeyError):
        problems.stream("bogus", seed=0)


def test_truth_vector_missing_symbol_raises_keyerror():
    with pytest.raises(KeyError):
        problems.truth_vector("a*exp(b*t)", {"a": 1.0})


def test_truth_vector_unparsable_expr_raises_sympifyerror():
    with pytest.raises(sp.SympifyError):
        problems.truth_vector("a*(*t", {"a": 1.0})


def test_model_unparsable_expr_raises_sympifyerror():
    with pytest.raises(sp.SympifyError):
        problems.model("a*(*t", ["a"])
