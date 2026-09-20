"""The block adaptations: explicit edges, aggregated data, alignment.

Each test names the mutation of blocks.py it fails under. The Monte Carlo
tests are seeded and sized for a couple of seconds, not for a measurement
report.
"""

import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import least_squares

import dtfit
from dtfit._stats import information_criteria
from dtfit.image import Original
from dtfit.image.bases import BlockBasis, LegendreBasis, u_of
from dtfit_experimental import (
    AlignedFit,
    EdgeBlockBasis,
    aggregated_image,
    detect_jumps,
    fit_aggregated,
    fit_aligned,
)
from dtfit_experimental.blocks import SegmentBasis, coarsen


def test_equal_edges_reproduce_the_block_basis():
    # Fails if evaluate() drops the offset after the search (every window
    # shifted by one): the block basis is the equal-edge case of this one.
    u = np.random.default_rng(0).uniform(-1.3, 1.3, 200)
    for k in (1, 4, 7):
        b = EdgeBlockBasis(np.linspace(-1.0, 1.0, k + 1))
        assert b.n_coef == k
        assert np.array_equal(b.evaluate(u), BlockBasis(k).evaluate(u))


def test_windows_are_half_open_at_an_interior_edge():
    # Fails under searchsorted(side="left"), which would put the edge
    # sample in the window that ends there.
    b = EdgeBlockBasis([-1.0, -0.25, 1.0])
    phi = b.evaluate([-0.25 - 1e-9, -0.25, 1.0])
    assert phi[0].tolist() == [1.0, 0.0]
    assert phi[1].tolist() == [0.0, 1.0]
    assert phi[2].tolist() == [0.0, 1.0]


def test_interior_edges_go_to_the_window_on_their_right():
    # Fails under searchsorted(side="left"), and under a floor-based
    # index: at K = 6 and K = 12 the block basis rounds some of these u to
    # the window on their left.
    disagree = 0
    for k in (3, 6, 12):
        e = np.linspace(-1.0, 1.0, k + 1)
        u = e[1:-1]
        phi = EdgeBlockBasis(e).evaluate(u)
        assert phi.sum() == k - 1
        assert np.argmax(phi, axis=1).tolist() == list(range(1, k))
        disagree += int(not np.array_equal(phi, BlockBasis(k).evaluate(u)))
    assert disagree == 2


def test_u_outside_the_unit_interval_is_clamped():
    # Fails without the clip in evaluate(): an index out of range, or a
    # sample lost from the end window.
    b = EdgeBlockBasis([-1.0, 0.5, 1.0])
    phi = b.evaluate([-4.0, 4.0])
    assert phi[0].tolist() == [1.0, 0.0]
    assert phi[1].tolist() == [0.0, 1.0]


def test_a_dropped_interval_gives_a_zero_row():
    # Fails if evaluate() maps a dropped interval onto a kept column.
    b = EdgeBlockBasis([-1.0, -0.5, 0.5, 1.0], keep=[True, False, True])
    assert b.n_coef == 2
    phi = b.evaluate([-0.75, 0.0, 0.75])
    assert phi[0].tolist() == [1.0, 0.0]
    assert phi[1].tolist() == [0.0, 0.0]
    assert phi[2].tolist() == [0.0, 1.0]


def test_equality_and_hash_see_edges_and_keep():
    # Fails if to_dict or __hash__ carries only name and order: these three
    # bases all have order 2.
    a = EdgeBlockBasis([-1.0, 0.0, 1.0])
    b = EdgeBlockBasis([-1.0, 0.5, 1.0])
    c = EdgeBlockBasis([-1.0, 0.0, 0.5, 1.0], keep=[True, False, True])
    assert a != b and a != c and b != c
    assert len({hash(a), hash(b), hash(c)}) == 3
    assert a == EdgeBlockBasis([-1.0, 0.0, 1.0])
    assert hash(a) == hash(EdgeBlockBasis([-1.0, 0.0, 1.0]))


def test_each_edge_guard_rejects_its_own_input():
    # One input per guard, each passing every earlier guard; fails if a
    # guard is dropped or its order is changed.
    with pytest.raises(ValueError, match="at least 2 edges"):
        EdgeBlockBasis([-1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        EdgeBlockBasis([-1.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="from -1 to"):
        EdgeBlockBasis([-0.5, 0.0, 1.0])
    with pytest.raises(ValueError, match="one entry per interval"):
        EdgeBlockBasis([-1.0, 0.0, 1.0], keep=[True])
    with pytest.raises(ValueError, match="at least one interval"):
        EdgeBlockBasis([-1.0, 0.0, 1.0], keep=[False, False])


def test_on_requires_the_domain_ends():
    # Fails if on() rescales without checking the ends: the basis would
    # silently cover a different interval than the image's domain.
    with pytest.raises(ValueError, match="ends of the domain"):
        EdgeBlockBasis.on([0.5, 3.0, 9.0], (0.0, 10.0))
    b = EdgeBlockBasis.on([0.0, 3.0, 10.0], (0.0, 10.0))
    assert b.edges_on((0.0, 10.0)) == pytest.approx([0.0, 3.0, 10.0])
    lo, hi = b.windows_on((0.0, 10.0))
    assert lo.tolist() == [0.0, 3.0] and hi.tolist() == [3.0, 10.0]


def test_segment_basis_gram_is_block_diagonal():
    # Fails if evaluate() drops the segment mask: a sample would then also
    # light the neighbouring segment's columns, filling in the
    # off-diagonal blocks of G that a Legendre-per-segment basis must not
    # have.
    edges = [-1.0, -0.2, 0.5, 1.0]
    orders = [2, 1, 3]
    b = SegmentBasis(edges, orders)
    rng = np.random.default_rng(2)
    u = rng.uniform(-1.0, 1.0, 300)
    phi = b.evaluate(u)
    idx = np.clip(
        np.searchsorted(np.asarray(edges), u, side="right") - 1, 0, 2
    )
    bounds = np.cumsum([0] + [o + 1 for o in orders])
    g = phi.T @ phi
    for i in range(3):
        for j in range(3):
            if i != j:
                block = g[bounds[i]:bounds[i + 1], bounds[j]:bounds[j + 1]]
                assert np.all(block == 0.0)
    for j in range(3):
        rows = np.flatnonzero(idx == j)
        assert rows.size
        outside = np.ones(phi.shape[1], dtype=bool)
        outside[bounds[j]:bounds[j + 1]] = False
        assert np.all(phi[np.ix_(rows, outside)] == 0.0)


def test_on_with_one_segment_reproduces_legendre():
    # Fails if on()'s edge-to-u mapping per segment slips (an extra
    # rescale, or the segment's own domain used instead of the whole
    # one): with a single segment the basis must be exactly the plain
    # Legendre basis on the same domain.
    domain = (2.0, 12.0)
    x = np.linspace(*domain, 50)
    b = SegmentBasis.on([domain[0], domain[1]], domain, n_coef=6)
    ref = LegendreBasis(int(b.orders[0]))
    got = b.evaluate(u_of(x, *domain))
    want = ref.evaluate(u_of(x, *domain))
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_on_shares_coefficients_with_a_floor_of_two():
    # Fails if the max(2, ...) floor is dropped: the short first segment
    # here would otherwise get zero or one coefficient (no room for a
    # slope, or none at all).
    edges = [0.0, 0.1, 5.0, 10.0]
    domain = (0.0, 10.0)
    b = SegmentBasis.on(edges, domain, n_coef=9)
    shares = b.orders + 1
    assert np.all(shares >= 2)
    assert abs(int(shares.sum()) - 9) <= 1


_TAU = 4.37
_TWO_REGIME_TRUTH = (2.0, 0.6, 3.0, 0.9)


def _two_regime(t, a1, l1, a2, l2):
    t = np.asarray(t, dtype=float)
    return np.where(
        t < _TAU, a1 * np.exp(-l1 * t), a2 * np.exp(-l2 * (t - _TAU))
    )


def _num_jac(f, x, c, h=1e-6):
    c = np.asarray(c, dtype=float)
    cols = []
    for i in range(c.size):
        d = np.zeros_like(c)
        d[i] = h * max(1.0, abs(c[i]))
        cols.append((f(x, *(c + d)) - f(x, *(c - d))) / (2 * d[i]))
    return np.stack(cols, axis=1)


def _exact_efficiency(phi, jac):
    """The deterministic image-restriction ratio of probe_efficiency.py,
    independent of any noise draw."""
    keep = phi.any(axis=0)
    phi = phi[:, keep]
    q, _ = np.linalg.qr(phi)
    restricted = np.linalg.pinv(jac.T @ (q @ (q.T @ jac)))
    full = np.linalg.inv(jac.T @ jac)
    return np.diag(full) / np.diag(restricted)


def test_segment_on_the_switch_reaches_efficiency_where_legendre_does_not():
    # Fails if on() ignores the given interior edge (the edges ignored):
    # the segment basis would then behave like plain Legendre across the
    # whole domain, and a2's efficiency would fall to Legendre's ~0.66
    # instead of reaching 1.0.
    domain = (0.0, 10.0)
    x = np.linspace(*domain, 2000)
    jac = _num_jac(_two_regime, x, _TWO_REGIME_TRUTH)
    rng = np.random.default_rng(4)
    y = _two_regime(x, *_TWO_REGIME_TRUTH) + 0.01 * rng.standard_normal(x.size)
    original = Original(x, y, domain=domain)
    seg = SegmentBasis.on([domain[0], _TAU, domain[1]], domain, n_coef=16)
    result = dtfit.fit(
        _two_regime, original, basis=seg, p0=_TWO_REGIME_TRUTH,
        param_names=["a1", "l1", "a2", "l2"],
    )
    assert np.all(np.isfinite(result.coeffs))
    eff_seg = _exact_efficiency(seg.evaluate(u_of(x, *domain)), jac)
    eff_leg = _exact_efficiency(
        LegendreBasis(seg.n_coef - 1).evaluate(u_of(x, *domain)), jac
    )
    assert eff_seg[2] > 0.999
    assert eff_leg[2] < 0.7


EPOCH = 3.4
JUMP_DOMAIN = (0.0, 10.0)


def _jump_model(t, a, b, d):
    return a + b * t + d * (t >= EPOCH)


def test_an_edge_on_the_epoch_beats_straddling_windows():
    # Fails if on() or evaluate() ignores the given edges (uniform windows
    # at the same K straddle the epoch and lose the jump's information).
    x = np.linspace(*JUMP_DOMAIN, 400)
    truth = np.array([1.0, 0.3, 2.0])
    k = 4
    uniform = np.linspace(JUMP_DOMAIN[0], JUMP_DOMAIN[1], k + 1)
    moved = uniform.copy()
    moved[int(np.argmin(np.abs(uniform - EPOCH)))] = EPOCH
    aligned = EdgeBlockBasis.on(moved, JUMP_DOMAIN)
    rng = np.random.default_rng(3)
    err = {"aligned": [], "uniform": []}
    for _ in range(100):
        y = _jump_model(x, *truth) + rng.standard_normal(x.size)
        data = Original(x, y, domain=JUMP_DOMAIN)
        for name, basis in (("aligned", aligned), ("uniform", BlockBasis(k))):
            r = dtfit.fit(_jump_model, data, basis=basis, p0=truth,
                          param_names=["a", "b", "d"])
            err[name].append(r.coeffs[2] - truth[2])
    rmse = {k_: float(np.sqrt(np.mean(np.square(v)))) for k_, v in err.items()}
    assert rmse["uniform"] > 1.25 * rmse["aligned"]


CYCLE_DOMAIN = (0.0, 12.0)
CYCLE_NAMES = ["a", "c", "ph", "w"]
CYCLE_TRUTH = np.array([2.0, 1.0, 0.4, 2.0 * np.pi / 6.0])


def _cycle(t, a, c, ph, w):
    return c + a * np.sin(w * t + ph)


def _cycle_windows(n=4800, k=12):
    x = (np.linspace(CYCLE_DOMAIN[0], CYCLE_DOMAIN[1], n, endpoint=False)
         + 0.5 * (CYCLE_DOMAIN[1] - CYCLE_DOMAIN[0]) / n)
    edges = np.linspace(CYCLE_DOMAIN[0], CYCLE_DOMAIN[1], k + 1)
    idx = np.minimum(
        ((x - CYCLE_DOMAIN[0]) / (CYCLE_DOMAIN[1] - CYCLE_DOMAIN[0])
         * k).astype(int), k - 1)
    return x, edges, idx, np.bincount(idx, minlength=k)


def test_aggregated_image_carries_the_window_sum_least_squares():
    # Fails if S, G or sumsq is built from the window means instead of
    # the totals and the counts.
    x, edges, idx, counts = _cycle_windows()
    rng = np.random.default_rng(1)
    y = _cycle(x, *CYCLE_TRUTH) + 0.5 * rng.standard_normal(x.size)
    totals = np.bincount(idx, weights=y, minlength=counts.size)
    r = fit_aggregated(_cycle, edges, totals, counts, x=x, p0=CYCLE_TRUTH,
                       param_names=CYCLE_NAMES)

    def resid(p):
        model_sums = np.bincount(
            idx, weights=_cycle(x, *p), minlength=counts.size)
        return (model_sums - totals) / np.sqrt(counts)

    by_hand = least_squares(resid, CYCLE_TRUTH).x
    assert np.max(np.abs(by_hand - r.coeffs) / np.abs(CYCLE_TRUTH)) < 1e-6


def test_the_area_fit_is_unbiased_where_the_midpoint_reading_is_not():
    # Fails if the image carries the window means (S = totals / counts)
    # instead of the totals: what is solved is then no longer the
    # equal-areas least squares, and the bias on this cycle returns.
    x, edges, idx, counts = _cycle_windows()
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(1)
    area, mid = [], []
    for _ in range(30):
        y = _cycle(x, *CYCLE_TRUTH) + 0.5 * rng.standard_normal(x.size)
        totals = np.bincount(idx, weights=y, minlength=counts.size)
        area.append(fit_aggregated(
            _cycle, edges, totals, counts, x=x, p0=CYCLE_TRUTH,
            param_names=CYCLE_NAMES).coeffs)
        mid.append(least_squares(
            lambda p: _cycle(midpoints, *p) - totals / counts,
            CYCLE_TRUTH).x)
    area_bias = np.mean(np.asarray(area)[:, 0]) / CYCLE_TRUTH[0] - 1.0
    mid_bias = np.mean(np.asarray(mid)[:, 0]) / CYCLE_TRUTH[0] - 1.0
    assert mid_bias < -0.03
    assert abs(area_bias) < 0.005


def test_without_positions_the_window_means_are_read_as_areas():
    # Fails if the grid of the x=None path puts every sample of a window at
    # the window centre: the fit is then the midpoint reading, whose bias on
    # this cycle is what the test measures.
    x, edges, idx, counts = _cycle_windows()
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(1)
    blind, known, mid = [], [], []
    for _ in range(30):
        y = _cycle(x, *CYCLE_TRUTH) + 0.5 * rng.standard_normal(x.size)
        totals = np.bincount(idx, weights=y, minlength=counts.size)
        blind.append(fit_aggregated(
            _cycle, edges, totals, counts, p0=CYCLE_TRUTH,
            param_names=CYCLE_NAMES).coeffs)
        known.append(fit_aggregated(
            _cycle, edges, totals, counts, x=x, p0=CYCLE_TRUTH,
            param_names=CYCLE_NAMES).coeffs)
        mid.append(least_squares(
            lambda p: _cycle(midpoints, *p) - totals / counts,
            CYCLE_TRUTH).x)
    blind_bias = np.mean(np.asarray(blind)[:, 0]) / CYCLE_TRUTH[0] - 1.0
    known_bias = np.mean(np.asarray(known)[:, 0]) / CYCLE_TRUTH[0] - 1.0
    mid_bias = np.mean(np.asarray(mid)[:, 0]) / CYCLE_TRUTH[0] - 1.0
    assert mid_bias < -0.03
    assert abs(blind_bias) < 0.005
    assert blind_bias == pytest.approx(known_bias, abs=1e-3)
    image = aggregated_image([0.0, 1.0, 3.0], [1.0, 2.0], [2, 3])
    assert image.grid.positions() == pytest.approx(
        [0.25, 0.75, 1.0 + 1.0 / 3.0, 2.0, 2.0 + 2.0 / 3.0])


def test_aggregated_result_counts_windows_not_samples():
    # Fails if n_obs is left on the sample count: aic and bic would then
    # read a window-scale rss against the samples the caller never saw.
    x, edges, idx, counts = _cycle_windows()
    rng = np.random.default_rng(2)
    y = _cycle(x, *CYCLE_TRUTH) + 0.5 * rng.standard_normal(x.size)
    totals = np.bincount(idx, weights=y, minlength=counts.size)
    r = fit_aggregated(_cycle, edges, totals, counts, x=x, p0=CYCLE_TRUTH,
                       param_names=CYCLE_NAMES)
    aic, bic = information_criteria(r.rss, counts.size, len(CYCLE_NAMES))
    assert r.n_obs == counts.size
    assert r.aic == pytest.approx(aic)
    assert r.bic == pytest.approx(bic)


DECAY_DOMAIN = (0.0, 6.0)
DECAY_NAMES = ["a", "c", "tau"]
DECAY_TRUTH = np.array([5.0, 1.0, 1.5])


def _decay(t, a, c, tau):
    return c + a * np.exp(-t / tau)


def _decay_windows(n, k):
    span = DECAY_DOMAIN[1] - DECAY_DOMAIN[0]
    x = (np.linspace(DECAY_DOMAIN[0], DECAY_DOMAIN[1], n, endpoint=False)
         + 0.5 * span / n)
    edges = np.linspace(DECAY_DOMAIN[0], DECAY_DOMAIN[1], k + 1)
    idx = np.minimum(((x - DECAY_DOMAIN[0]) / span * k).astype(int), k - 1)
    return x, edges, idx, np.bincount(idx, minlength=k)


def test_aggregated_covariance_is_on_the_window_scale():
    # Fails if fit_aggregated drops the (n - p) / (K - p) rescale: the
    # covariance is then the sample-count one, whose coverage collapses.
    n, k, reps = 1200, 32, 1000
    p = len(DECAY_TRUTH)
    x, edges, idx, counts = _decay_windows(n, k)
    rng = np.random.default_rng(5)
    t_q = float(stats.t.ppf(0.975, k - p))
    hit, hit_raw = np.zeros(p), np.zeros(p)
    for _ in range(reps):
        y = _decay(x, *DECAY_TRUTH) + 0.5 * rng.standard_normal(n)
        totals = np.bincount(idx, weights=y, minlength=k)
        r = fit_aggregated(_decay, edges, totals, counts, x=x,
                           p0=DECAY_TRUTH, param_names=DECAY_NAMES)
        se = np.sqrt(np.diag(r.cov))
        miss = np.abs(r.coeffs - DECAY_TRUTH)
        hit += miss <= t_q * se
        hit_raw += miss <= t_q * se * np.sqrt((k - p) / (n - p))
    sem = np.sqrt(0.95 * 0.05 / reps)
    assert np.all(np.abs(hit / reps - 0.95) <= 3.0 * sem)
    assert np.all(hit_raw / reps < 0.5)


def test_aggregated_covariance_is_the_window_least_squares_one():
    # Fails under (n - p) / K in place of (n - p) / (K - p): at K = 8 with
    # three parameters that is a factor 0.625 on every variance.
    k, p = 8, len(DECAY_TRUTH)
    x, edges, idx, counts = _decay_windows(480, k)
    rng = np.random.default_rng(4)
    y = _decay(x, *DECAY_TRUTH) + 0.5 * rng.standard_normal(x.size)
    totals = np.bincount(idx, weights=y, minlength=k)
    r = fit_aggregated(_decay, edges, totals, counts, x=x, p0=DECAY_TRUTH,
                       param_names=DECAY_NAMES)
    a, c, tau = r.coeffs
    d = np.stack([np.exp(-x / tau), np.ones_like(x),
                  a * x / tau ** 2 * np.exp(-x / tau)], axis=1)
    jac = np.stack([np.bincount(idx, weights=d[:, j], minlength=k)
                    for j in range(p)], axis=1)
    exact = (r.rss / (k - p)) * np.linalg.inv(
        jac.T @ np.diag(1.0 / counts) @ jac)
    assert np.diag(r.cov) == pytest.approx(np.diag(exact), rel=1e-3)


def test_aggregated_covariance_is_none_with_a_window_per_parameter():
    # Fails if the K == p case is not turned into cov None: the rescale
    # divides by zero, or a floored (K - p) reports finite standard errors
    # for a fit with no residual freedom left.
    p = len(DECAY_TRUTH)
    x, edges, idx, counts = _decay_windows(300, p)
    rng = np.random.default_rng(0)
    y = _decay(x, *DECAY_TRUTH) + 0.5 * rng.standard_normal(x.size)
    totals = np.bincount(idx, weights=y, minlength=p)
    r = fit_aggregated(_decay, edges, totals, counts, x=x, p0=DECAY_TRUTH,
                       param_names=DECAY_NAMES)
    assert r.cov is None
    assert r.n_obs == p
    with pytest.raises(ValueError, match="no covariance"):
        r.stderr()


def test_aggregated_guards_reject_their_own_input():
    # One input per guard; fails if a guard is dropped, in particular the
    # per-window count check that keeps x and counts describing the same
    # aggregation.
    edges = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        aggregated_image([0.0, 2.0, 1.0], [1.0, 1.0], [1, 1])
    with pytest.raises(ValueError, match="one entry per window"):
        aggregated_image(edges, [1.0, 1.0, 1.0], [1, 1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        aggregated_image(edges, [1.0, 1.0], [2, -1])
    with pytest.raises(ValueError, match=r"window 1 has 1\.5"):
        aggregated_image(edges, [1.0, 1.0], [2, 1.5])
    with pytest.raises(ValueError, match="finite where the count"):
        aggregated_image(edges, [1.0, np.nan], [2, 1])
    with pytest.raises(ValueError, match="every window is empty"):
        aggregated_image(edges, [0.0, 0.0], [0, 0])
    with pytest.raises(ValueError, match="counts sum to"):
        aggregated_image(edges, [1.0, 1.0], [2, 2], x=[0.1, 0.2, 1.5])
    with pytest.raises(ValueError, match="every position must lie"):
        aggregated_image(edges, [1.0, 1.0], [2, 2],
                         x=[0.1, 0.2, 1.5, 2.9])
    with pytest.raises(ValueError, match="positions per window"):
        aggregated_image(edges, [1.0, 1.0], [2, 2],
                         x=[0.1, 0.2, 0.3, 1.5])


def test_empty_windows_are_dropped_from_the_basis():
    # Fails if keep is not passed to the basis, which would put a zero row
    # and column in the Gram, and if sumy is summed over every window
    # rather than the kept ones: here that carries the nan of the empty
    # window into tss.
    image = aggregated_image(
        [0.0, 1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [2, 0, 3])
    assert image.n_coef == 2
    assert image.basis.keep.tolist() == [True, False, True]
    assert image.S.tolist() == [4.0, 6.0]
    assert image.G.tolist() == [[2.0, 0.0], [0.0, 3.0]]
    assert image.n == 5
    assert image.sumy == 10.0
    assert image.sumsq == pytest.approx(4.0 ** 2 / 2 + 6.0 ** 2 / 3)
    assert image.grid.positions().size == 5


ALIGN_DOMAIN = (0.0, 10.0)
ALIGN_NAMES = ["a", "b", "c", "d"]
ALIGN_TRUTH = np.array([1.0, 0.3, 0.8, -0.5])


def _smooth(t, a, b, c, d):
    return (a + b * t + c * np.sin(2.0 * np.pi * t)
            + d * np.cos(2.0 * np.pi * t))


def _align_series(rng, epochs=(), steps=(), outliers=0.0):
    x = np.linspace(ALIGN_DOMAIN[0], ALIGN_DOMAIN[1], 2000)
    y = _smooth(x, *ALIGN_TRUTH) + rng.standard_normal(x.size)
    for e, s in zip(epochs, steps):
        y = y + s * (x >= e)
    if outliers:
        k = int(outliers * x.size)
        where = rng.choice(x.size, k, replace=False)
        y[where] += 20.0 * rng.choice([-1.0, 1.0], k)
    return x, y


def test_fit_aligned_finds_two_jumps_and_their_amplitudes():
    # Fails if the detector ranks z rather than |z|.
    rng = np.random.default_rng(11)
    epochs, steps = (2.37, 6.81), (10.0, -10.0)
    x, y = _align_series(rng, epochs, steps)
    out = fit_aligned(_smooth, Original(x, y, domain=ALIGN_DOMAIN),
                      n_windows=32, param_names=ALIGN_NAMES,
                      p0=np.zeros(4))
    assert isinstance(out, AlignedFit)
    fine_width = (ALIGN_DOMAIN[1] - ALIGN_DOMAIN[0]) / (8 * 32)
    assert out.epochs.size == 2
    assert np.max(np.abs(out.epochs - np.asarray(epochs))) < fine_width
    se = np.asarray([out.result.stderr()[f"step_{i + 1}"] for i in (0, 1)])
    assert np.all(np.abs(out.steps - np.asarray(steps)) < 3.0 * se)
    assert out.n_dropped == 0
    assert not out.basis.keep.all()


def test_a_jump_free_series_yields_the_plain_uniform_fit():
    # Fails if the threshold is not applied (any largest |z| accepted) or
    # if the no-detection path skips the coarse refit: the basis would then
    # be the fine one and the parameters far off.
    rng = np.random.default_rng(12)
    x, y = _align_series(rng)
    out = fit_aligned(_smooth, Original(x, y, domain=ALIGN_DOMAIN),
                      n_windows=32, threshold=5.0,
                      param_names=ALIGN_NAMES, p0=np.zeros(4))
    assert out.epochs.size == 0
    assert out.steps.size == 0
    assert out.basis.order == 32
    assert out.basis.keep.all()
    assert out.result.coeffs == pytest.approx(ALIGN_TRUTH, abs=0.2)


def test_clip_keeps_gross_outliers_out_of_the_image():
    # Fails if clip is ignored or applied after the image is built: the
    # 20-sigma outliers then stay in and the detector fires on them.
    rng = np.random.default_rng(14)
    x, y = _align_series(rng, outliers=0.02)
    data = Original(x, y, domain=ALIGN_DOMAIN)
    clipped = fit_aligned(_smooth, data, n_windows=32, clip=5.0,
                          param_names=ALIGN_NAMES, p0=np.zeros(4))
    plain = fit_aligned(_smooth, data, n_windows=32,
                        param_names=ALIGN_NAMES, p0=np.zeros(4))
    assert clipped.n_dropped == int(0.02 * x.size)
    assert plain.n_dropped == 0
    assert plain.epochs.size > 0
    assert clipped.epochs.size == 0


SH5_DOMAIN = (0.0, 2000.0)
SH5_EPOCH = 290.0


def _line(t, a, b):
    return a + b * t


def test_no_sample_of_a_flagged_window_enters_the_refit():
    # On this integer-timed record the two window rules disagree at seven
    # samples, x = 290 among them. Fails if the sample index in fit_aligned
    # is not the basis's own rule (a truncation index leaves x = 290 in the
    # fit while the basis drops it with the rest of window 29), and if the
    # flagged windows are kept in the fit at all (sel = all True).
    n = 2001
    x = np.arange(n, dtype=float)
    rng = np.random.default_rng(31)
    y = (10.0 + 0.01 * x + 8.0 * (x >= SH5_EPOCH)
         + rng.standard_normal(n))
    out = fit_aligned(_line, Original(x, y, domain=SH5_DOMAIN),
                      n_windows=25, fine=8, param_names=["a", "b"],
                      p0=[0.0, 0.0])
    fine_width = (SH5_DOMAIN[1] - SH5_DOMAIN[0]) / (8 * 25)
    assert out.epochs.size == 1
    assert abs(out.epochs[0] - SH5_EPOCH) < fine_width
    in_basis = out.basis.evaluate(u_of(x, *SH5_DOMAIN)).any(axis=1)
    assert int((~in_basis).sum()) == 10
    assert out.result.n_obs == int(in_basis.sum())
    assert out.n_dropped == 0


def test_fit_aligned_passes_n_min_to_the_detector():
    # Fails if n_min is not wired through to detect_jumps: with no fine
    # window holding a million samples the detector has nothing to work on
    # and the jump in this series must go unfound.
    rng = np.random.default_rng(17)
    x, y = _align_series(rng, (4.13,), (10.0,))
    data = Original(x, y, domain=ALIGN_DOMAIN)
    assert fit_aligned(_smooth, data, n_windows=32,
                       param_names=ALIGN_NAMES,
                       p0=np.zeros(4)).epochs.size == 1
    assert fit_aligned(_smooth, data, n_windows=32, n_min=10 ** 6,
                       param_names=ALIGN_NAMES,
                       p0=np.zeros(4)).epochs.size == 0


def test_fit_aligned_guards_reject_their_own_input():
    # One input per guard, each passing every earlier guard; fails if a
    # guard is dropped or its order is changed.
    rng = np.random.default_rng(15)
    x, y = _align_series(rng)
    data = Original(x, y, domain=ALIGN_DOMAIN)
    with pytest.raises(ValueError, match="n_windows and fine"):
        fit_aligned(_smooth, data, n_windows=0, param_names=ALIGN_NAMES,
                    p0=np.zeros(4))
    with pytest.raises(ValueError, match="max_iter"):
        fit_aligned(_smooth, data, n_windows=32, max_iter=0,
                    param_names=ALIGN_NAMES, p0=np.zeros(4))
    with pytest.raises(ValueError, match="too few for"):
        fit_aligned(_smooth, data, n_windows=32, clip=0.05,
                    param_names=ALIGN_NAMES, p0=np.zeros(4))


def test_detect_jumps_reads_the_amplitude_and_the_epoch():
    # The step sits a fifth into window 40, whose mean is therefore 3.2.
    # Fails without the non-maximum suppression around a detection, if the
    # amplitude is read from that mean rather than from the two
    # extrapolations, and if the epoch is placed at the window centre or at
    # frac instead of 1 - frac (four fifths in).
    kf, b0 = 64, 40
    edges = np.linspace(0.0, 10.0, kf + 1)
    width = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.full(kf, 20.0)
    rng = np.random.default_rng(7)
    resid = 4.0 * (np.arange(kf) > b0) + 0.02 * rng.standard_normal(kf)
    resid[b0] = 0.8 * 4.0 + 0.02 * rng.standard_normal()
    found = detect_jumps(counts, counts * centres, resid * counts, edges)
    assert len(found) == 1
    window, epoch, amplitude, z = found[0]
    assert window == b0
    assert epoch == pytest.approx(edges[b0] + 0.2 * width, abs=0.02 * width)
    assert amplitude == pytest.approx(4.0, abs=0.05)
    assert abs(z) > 5.0


def _ar1(rng, kf, phi=0.9):
    e = rng.standard_normal(kf)
    r = np.empty(kf)
    r[0] = e[0]
    for i in range(1, kf):
        r[i] = phi * r[i - 1] + e[i]
    return r


def test_self_scale_absorbs_the_colour_of_the_noise():
    # Fails if self_scale does not divide the statistic by its own spread:
    # these jump-free AR(1) residuals give three detections at threshold 5
    # without it.
    kf = 256
    edges = np.linspace(0.0, 10.0, kf + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.full(kf, 8.0)
    for seed in (2, 4, 5):
        r = _ar1(np.random.default_rng(seed), kf)
        args = (counts, counts * centres, r * counts, edges)
        assert len(detect_jumps(*args)) >= 1
        assert detect_jumps(*args, self_scale=True) == []


def test_self_scale_never_makes_a_detection_easier():
    # Fails if the divisor is not floored at 1: the spread of the statistic
    # on this white-noise record is 0.54, which would multiply every |z| by
    # 1.8 and turn the largest of them into a detection.
    kf = 96
    edges = np.linspace(0.0, 10.0, kf + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.full(kf, 16.0)
    r = np.random.default_rng(33).standard_normal(kf) / 4.0
    args = (counts, counts * centres, r * counts, edges)
    plain = detect_jumps(*args, threshold=1.4)
    assert plain
    assert detect_jumps(*args, threshold=1.4, self_scale=True) == plain
    assert detect_jumps(*args, threshold=2.0, self_scale=True) == []


def test_detect_jumps_needs_a_positive_threshold():
    # Fails if the guard is dropped: at threshold 0 the suppression loop
    # never reaches its break and the call does not return.
    kf = 16
    edges = np.linspace(0.0, 10.0, kf + 1)
    counts = np.full(kf, 5.0)
    with pytest.raises(ValueError, match="threshold must be > 0"):
        detect_jumps(counts, counts * 0.5 * (edges[:-1] + edges[1:]),
                     np.zeros(kf), edges, threshold=0.0)


def test_detect_jumps_returns_nothing_from_too_few_windows():
    # Fails if the guard on the number of windows holding n_min samples is
    # dropped: the noise scale is then a median of nothing, a RuntimeWarning
    # and a nan statistic.
    counts = np.array([10.0, 0.0, 0.0, 0.0])
    edges = np.linspace(0.0, 4.0, 5)
    sums_x = counts * 0.5 * (edges[:-1] + edges[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect_jumps(
            counts, sums_x, np.array([30.0, 0.0, 0.0, 0.0]), edges
        ) == []


def test_detect_jumps_needs_a_positive_noise_scale():
    # Fails if the guard on a zero noise scale is dropped: every z is then
    # a division by zero and the whole record reads as jumps.
    kf = 32
    edges = np.linspace(0.0, 10.0, kf + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.full(kf, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect_jumps(
            counts, counts * centres,
            counts * 4.0 * (centres >= 5.0), edges
        ) == []


def test_coarsen_never_spans_a_flagged_window():
    # Fails if coarsen merges across a flagged window.
    lab = coarsen(64, 8, [20, 21, 45])
    assert lab.tolist()[20:22] == [-1, -1]
    assert lab[45] == -1
    assert np.all(np.diff(lab[lab >= 0]) >= 0)
    for b in (20, 21, 45):
        assert lab[b - 1] != lab[b + 1]
    assert 6 <= len(set(lab[lab >= 0].tolist())) <= 10
    assert (coarsen(16, 4, []) == np.repeat(np.arange(4), 4)).all()


def test_coarsen_rejects_a_degenerate_shape():
    # One input per clause of the guard; fails if the guard is dropped,
    # which returns one coarse window and an empty labelling instead.
    with pytest.raises(ValueError, match="must be >= 1"):
        coarsen(64, 0)
    with pytest.raises(ValueError, match="must be >= 1"):
        coarsen(0, 4)
