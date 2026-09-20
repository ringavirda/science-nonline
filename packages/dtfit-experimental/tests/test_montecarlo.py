"""Monte-Carlo scaffolding: seeded draws, grids, contamination, image
efficiency, replicate summaries and the process pool.

Each test names the mutation of montecarlo.py it fails under.
"""

import numpy as np
import pytest
import threadpoolctl

from dtfit_experimental.study.montecarlo import (
    contaminate,
    generators,
    grid,
    image_efficiency,
    noisy,
    numeric_jacobian,
    pool_map,
    summarize,
)


def test_generators_draw_independent_reproducible_streams():
    a = generators(0, 2)
    assert len(a) == 2
    # fails if generators() reused one stream for every replicate
    x0 = a[0].standard_normal(5)
    x1 = a[1].standard_normal(5)
    assert not np.allclose(x0, x1)
    # fails under a non-reproducible seed source
    b = generators(0, 2)
    assert np.array_equal(x0, b[0].standard_normal(5))
    assert np.array_equal(x1, b[1].standard_normal(5))


def test_grid_uniform_pins_endpoints():
    x = grid("uniform", 50, span=2.0)
    assert x[0] == 0.0
    assert x[-1] == 2.0
    assert x.size == 50


def test_grid_clustered_puts_half_in_first_fifth():
    rng = np.random.default_rng(1)
    x = grid("clustered", 100, span=10.0, rng=rng)
    # fails if the endpoint clip were dropped
    assert x[0] == 0.0
    assert x[-1] == 10.0
    in_first_fifth = np.sum(x <= 2.0)
    # fails under any split fraction other than half
    assert in_first_fifth == 50


def test_grid_random_stays_in_span():
    rng = np.random.default_rng(2)
    x = grid("random", 30, span=5.0, rng=rng)
    assert x[0] == 0.0
    assert x[-1] == 5.0
    assert np.all((x >= 0.0) & (x <= 5.0))


def test_grid_rejects_unknown_kind():
    with pytest.raises(ValueError):
        grid("triangular", 10)


def test_noisy_scales_sigma_to_signal_range():
    rng = np.random.default_rng(3)
    clean = np.linspace(0.0, 4.0, 500)  # ptp == 4.0
    y, sigma = noisy(clean, 0.1, rng)
    assert sigma == pytest.approx(0.4)
    assert y.shape == clean.shape
    # fails if noisy() returned clean unchanged
    assert not np.allclose(y, clean)


def test_contaminate_replaces_exactly_rounded_count():
    rng = np.random.default_rng(4)
    y = np.zeros(197)
    y_out, idx = contaminate(y, rng=rng, fraction=0.1, sigma=1.0)
    # round(0.1 * 197) == 20; fails under int()/floor() truncation (19)
    assert idx.size == 20
    assert np.all(y_out[idx] != 0.0)
    untouched = np.setdiff1d(np.arange(197), idx)
    assert np.all(y_out[untouched] == 0.0)


def test_contaminate_burst_leaves_runs_of_five():
    rng = np.random.default_rng(5)
    y = np.zeros(200)
    _, idx = contaminate(y, rng=rng, fraction=0.1, sigma=1.0, burst=5)
    assert idx.size == 20
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    # fails if burst were ignored and the indices scattered singly
    assert all(run.size == 5 for run in runs)


def test_numeric_jacobian_matches_known_derivative():
    def f(x, a, b):
        return a * x + b

    x = np.linspace(0.0, 1.0, 20)
    jac = numeric_jacobian(f, x, [2.0, 3.0])
    assert jac.shape == (20, 2)
    # d/da = x, d/db = 1; fails under a biased step formula
    assert jac[:, 0] == pytest.approx(x, abs=1e-4)
    assert jac[:, 1] == pytest.approx(np.ones(20), abs=1e-4)


def test_image_efficiency_full_span_is_one():
    x = np.linspace(0.0, 1.0, 200)
    jac = np.stack([np.ones(200), x], axis=1)
    eff = image_efficiency(jac, x, basis="block", order=200)
    assert eff.ratio == pytest.approx([1.0, 1.0], abs=1e-6)


def test_image_efficiency_coarse_basis_matches_measured_ratio():
    x = np.linspace(0.0, 1.0, 200)
    jac = np.stack([np.ones(200), x], axis=1)
    eff = image_efficiency(jac, x, basis="block", order=8)
    # fails under a projector from the wrong grid or a swapped covariance
    assert eff.ratio == pytest.approx([0.98828299, 0.98439961], abs=1e-6)
    assert eff.ratio[0] < 1.0


def test_summarize_counts_nonfinite_instead_of_dropping_silently():
    estimates = np.array([
        [1.0, 2.0],
        [1.2, 1.8],
        [np.nan, 2.0],
        [0.8, 2.2],
    ])
    truth = np.array([1.0, 2.0])
    s = summarize(estimates, truth)
    # fails if summarize dropped the non-finite row without reporting it
    assert s.n_nonfinite == 1
    assert s.bias.shape == (2,)
    assert s.rmse.shape == (2,)
    # fails if the NaN row leaked into bias/rmse
    assert np.all(np.isfinite(s.bias))
    assert np.all(np.isfinite(s.rmse))


def test_summarize_percentiles_shape():
    rng = np.random.default_rng(6)
    estimates = rng.normal(loc=[1.0, 2.0], scale=0.1, size=(500, 2))
    s = summarize(estimates, np.array([1.0, 2.0]), percentiles=(5, 50, 95))
    assert s.percentiles.shape == (3, 2)
    assert s.n_nonfinite == 0


def _one_blas_thread(x):
    np.dot(np.eye(8), np.eye(8))  # loads the BLAS library into this worker
    info = threadpoolctl.threadpool_info()
    limits = [i["num_threads"] for i in info if i["user_api"] == "blas"]
    return x, bool(limits) and all(n == 1 for n in limits)


def test_pool_map_keeps_order_and_caps_blas_threads():
    items = list(range(8))
    out = pool_map(_one_blas_thread, items, workers=2)
    xs = [o[0] for o in out]
    # fails if pool_map used as_completed instead of map
    assert xs == items
    # fails if the BLAS thread cap were not held for the worker's life
    assert all(o[1] for o in out)
