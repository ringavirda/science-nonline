"""The block adaptations: explicit edges, aggregated data, alignment.

Each test names the mutation of blocks.py it fails under. The Monte Carlo
tests are seeded and sized for a couple of seconds, not for a measurement
report; the measured numbers live in the module docstring of blocks.py.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares

import dtfit
from dtfit.image import Original
from dtfit.image.bases import BlockBasis
from dtfit_experimental import (
    AlignedFit,
    EdgeBlockBasis,
    aggregated_image,
    detect_jumps,
    fit_aggregated,
    fit_aligned,
)
from dtfit_experimental.blocks import coarsen


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
    # Fails if aggregated_image puts every sample of a window at the
    # window's centre: the fit is then the midpoint reading, whose bias on
    # this cycle is what the test measures.
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


DECAY_DOMAIN = (0.0, 6.0)
DECAY_NAMES = ["a", "c", "tau"]
DECAY_TRUTH = np.array([5.0, 1.0, 1.5])


def _decay(t, a, c, tau):
    return c + a * np.exp(-t / tau)


def test_aggregated_covariance_is_on_the_window_scale():
    # Fails if fit_aggregated drops the (n - p) / (K - p) rescale: the
    # covariance is then the sample-count one, whose coverage collapses.
    n, k = 1200, 32
    span = DECAY_DOMAIN[1] - DECAY_DOMAIN[0]
    x = (np.linspace(DECAY_DOMAIN[0], DECAY_DOMAIN[1], n, endpoint=False)
         + 0.5 * span / n)
    edges = np.linspace(DECAY_DOMAIN[0], DECAY_DOMAIN[1], k + 1)
    idx = np.minimum(((x - DECAY_DOMAIN[0]) / span * k).astype(int), k - 1)
    counts = np.bincount(idx, minlength=k)
    rng = np.random.default_rng(5)
    hit, hit_raw = 0, 0
    reps = 300
    for _ in range(reps):
        y = _decay(x, *DECAY_TRUTH) + 0.5 * rng.standard_normal(n)
        totals = np.bincount(idx, weights=y, minlength=k)
        r = fit_aggregated(_decay, edges, totals, counts, x=x,
                           p0=DECAY_TRUTH, param_names=DECAY_NAMES)
        raw = dtfit.fit(_decay, aggregated_image(edges, totals, counts, x=x),
                        p0=DECAY_TRUTH, param_names=DECAY_NAMES)
        for res, tally in ((r, "hit"), (raw, "raw")):
            se = float(np.sqrt(np.diag(res.cov))[2])
            covered = abs(res.coeffs[2] - DECAY_TRUTH[2]) <= 1.96 * se
            if tally == "hit":
                hit += covered
            else:
                hit_raw += covered
    assert 0.90 <= hit / reps <= 0.98
    assert hit_raw / reps < 0.5


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
    # Fails if keep is not passed to the basis: a zero-count window would
    # put a zero row and column in the Gram.
    image = aggregated_image(
        [0.0, 1.0, 2.0, 3.0], [4.0, 0.0, 6.0], [2, 0, 3])
    assert image.n_coef == 2
    assert image.basis.keep.tolist() == [True, False, True]
    assert image.S.tolist() == [4.0, 6.0]
    assert image.G.tolist() == [[2.0, 0.0], [0.0, 3.0]]
    assert image.n == 5
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


def test_detect_jumps_reads_the_amplitude_and_the_epoch():
    # Fails without the non-maximum suppression around a detection, and
    # if the amplitude is read from the flagged window's own mean rather
    # than from the two extrapolations.
    kf = 64
    edges = np.linspace(0.0, 10.0, kf + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.full(kf, 20.0)
    sums_x = counts * centres
    rng = np.random.default_rng(7)
    resid = 4.0 * (centres >= 6.28) + rng.standard_normal(kf) * 0.05
    found = detect_jumps(counts, sums_x, resid * counts, edges)
    assert len(found) == 1
    window, epoch, amplitude, z = found[0]
    assert edges[window] <= 6.28 <= edges[window + 1]
    assert abs(epoch - 6.28) < edges[1] - edges[0]
    assert amplitude == pytest.approx(4.0, abs=0.2)
    assert abs(z) > 5.0


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
