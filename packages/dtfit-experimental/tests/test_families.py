"""The sixteen-family catalogue and the bridge to dtfit's accuracy corpus.

Each test names the mutation of families.py it fails under.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.parameter_estimation import backend
from dtfit_experimental.study import baselines as bl
from dtfit_experimental.study import families as F
from dtfit_experimental.study.montecarlo import noisy


def test_families_has_sixteen_unique_keys():
    # fails under a Family.key duplicated across FAMILIES entries
    assert len(F.FAMILIES) == 16
    assert len(set(fam.key for fam in F.FAMILIES)) == 16


def test_families_match_backend_source():
    """Every field matches the entry :data:`backend.MODELS` was copied from.

    Catches a transcription slip that :func:`test_clean_data_recovery`
    cannot: a truth or bound copied wrong from the source is still
    internally consistent with itself there, since that test both draws its
    data from and scores against the same (possibly wrong) family. Fails
    under any single field of a FAMILIES entry drifting from its
    backend.MODELS source, e.g. a truth value edited in one place only.
    """
    by_key = {m["key"]: m for m in backend.MODELS}
    for family in F.FAMILIES:
        model = by_key[family.key]
        assert family.domain == model["domain"]
        assert family.shape == model["shape"]
        assert family.expr == model["expr"]
        assert tuple(family.names) == tuple(model["names"])
        assert family.truth == model["true"]
        assert family.span == model["t"]
        assert list(family.p0) == model["p0"]
        assert [tuple(b) for b in family.bounds] == [
            tuple(b) for b in model["bounds"]
        ]
        assert family.osc == model.get("osc")


@pytest.mark.parametrize("family", F.FAMILIES, ids=lambda fam: fam.key)
def test_clean_data_recovery(family):
    """SciPy NLLS recovers each family's truth from noise-free data seeded
    at its own p0/bounds. Fails if a family's p0 or bounds were copied
    wrong: a bound narrowed past the truth, or a p0 far enough from the
    truth to miss the basin, both leave the recovered value off by more
    than the tolerance below. A truth-only slip is not caught here (the
    same truth both draws the data and scores the fit); that case is
    :func:`test_families_match_backend_source`'s job.
    """
    x = np.linspace(family.span[0], family.span[1], 220)
    clean = family.func(x, *[family.truth[k] for k in family.names])
    lo = [b[0] for b in family.bounds]
    hi = [b[1] for b in family.bounds]
    popt = bl.scipy_curve_fit(
        x, clean, family.func, list(family.p0), bounds=(lo, hi)
    )
    estimate = dict(zip(family.names, popt))
    # fails under a bound narrowed to exclude the family's own truth
    assert F.param_error(estimate, family) == pytest.approx(0.0, abs=1e-2)


def test_simulate_draws_noise_through_montecarlo_noisy():
    # fails under simulate() forwarding a different noise level or a
    # differently-seeded rng to montecarlo.noisy than the one it was given
    family = F.FAMILIES[0]
    rng_a = np.random.default_rng(123)
    x, y, clean = F.simulate(family, rng_a, n=220, noise=0.2)

    rng_b = np.random.default_rng(123)
    x_ref = np.linspace(family.span[0], family.span[1], 220)
    clean_ref = family.func(x_ref, *[family.truth[k] for k in family.names])
    y_ref, _ = noisy(clean_ref, 0.2, rng_b)

    np.testing.assert_array_equal(x, x_ref)
    np.testing.assert_array_equal(clean, clean_ref)
    np.testing.assert_array_equal(y, y_ref)


def test_param_error_is_median_relative_percent():
    # fails under np.median swapped for np.mean, or the *100.0 dropped
    family = F.FAMILIES[0]  # damped: A=2.0, w=3.0, z=0.15
    estimate = {"A": 2.2, "w": 3.0, "z": 0.15}  # A off by 10%
    assert F.param_error(estimate, family) == pytest.approx(0.0)
    estimate2 = {"A": 2.2, "w": 3.3, "z": 0.15}  # A, w both off by 10%
    assert F.param_error(estimate2, family) == pytest.approx(10.0)


def test_param_error_accepts_either_key_order():
    # fails under estimate[k] replaced by a positional lookup into
    # estimate.values(), since declared and sorted_order share every value
    # but not its position
    family = dataclasses.replace(
        F.FAMILIES[0], names=("w", "A", "z"),
        truth={"w": 3.0, "A": 2.0, "z": 0.15},
    )
    declared = {"w": 3.0, "A": 2.0, "z": 0.15}
    sorted_order = {k: declared[k] for k in sorted(declared)}
    assert list(declared) != list(sorted_order)
    assert F.param_error(declared, family) == pytest.approx(0.0)
    assert F.param_error(sorted_order, family) == pytest.approx(0.0)


def test_param_error_raises_on_missing_key():
    # fails under estimate[k] replaced by estimate.get(k, default)
    family = F.FAMILIES[0]
    with pytest.raises(KeyError):
        F.param_error({"A": 2.0, "w": 3.0}, family)


def test_catalog_returns_dtfit_accuracy_scenarios():
    # fails under the resolver returning [] (wrong path, missing corpus)
    scenarios = F.catalog()
    assert len(scenarios) > 0


def test_catalog_empty_when_resolver_points_one_directory_up(monkeypatch, capsys):
    # fails under the accuracy/scenarios.py existence guard removed
    real_experiments_dir = F.paths.experiments_dir

    def _one_up():
        return real_experiments_dir().parent

    monkeypatch.setattr(F.paths, "experiments_dir", _one_up)
    scenarios = F.catalog()
    assert scenarios == []
    assert "families.catalog" in capsys.readouterr().out


def test_catalog_seed_returns_corpus_own_seed():
    # fails under catalog_seed swapping its (p0, bounds) return order
    # Only some scenarios seed finite bounds (the rest get None); this picks
    # the first one that does, so both return values are exercised.
    for scenario in F.catalog():
        x, y, _ = scenario.make(0.0, seed=0)
        expected_p0, expected_bounds = scenario.model()._seed_arrays(x, y)
        if expected_bounds is not None:
            break
    else:
        pytest.fail("no scenario in the corpus seeds finite bounds")

    p0, bounds = F.catalog_seed(scenario, x, y)
    assert p0 == expected_p0
    assert bounds == expected_bounds


def test_catalog_leaves_sys_path_unchanged():
    # fails under catalog() inserting tests_dir on sys.path
    before = list(sys.path)
    F.catalog()
    assert sys.path == before
