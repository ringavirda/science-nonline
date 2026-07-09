"""Promoted map-reduce estimators (PartitionedLSI / PartitionedEAC).

These were validated in the experiment suite and promoted to the stable API, so
they are tested here against ``dtfit`` directly (not the experimental package).
"""

import numpy as np
import pytest

from dtfit import PartitionedLSI, PartitionedEAC


@pytest.fixture
def exp_stream():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 600)
    y = 1.0 * np.exp(0.9 * t) + rng.normal(0, 0.02, t.size)
    return t, y, (1.0, 0.9)


def test_partitioned_lsi_reduce_recovers(exp_stream):
    t, y, (a, b) = exp_stream
    acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
    for xc, yc in zip(np.array_split(t, 5), np.array_split(y, 5)):
        acc.update(xc, yc)
    r = acc.fit(p0=[1.0, 1.0])
    assert abs(r.coeffs[0] - a) < 0.2 and abs(r.coeffs[1] - b) < 0.2


def test_partitioned_lsi_sequential_equals_whole(exp_stream):
    # Carrying the boundary sample makes disjoint sequential chunks exact.
    t, y, _ = exp_stream
    whole = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
    whole.update(t, y)
    seq = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
    for xc, yc in zip(np.array_split(t, 5), np.array_split(y, 5)):
        seq.update(xc, yc)
    np.testing.assert_allclose(whole.spectrum(), seq.spectrum(), rtol=1e-9,
                               atol=1e-9)


def test_partitioned_lsi_merge_is_associative(exp_stream):
    # Parallel reduce: partitions that SHARE boundary samples merge exactly.
    t, y, _ = exp_stream
    whole = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
    whole.update(t, y)
    bnds = np.linspace(0, t.size, 5).astype(int)
    parts = []
    for k in range(4):
        lo, hi = bnds[k], bnds[k + 1]
        sl = slice(lo, hi + 1)  # include the next partition's first sample
        p = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
        p.update(t[sl], y[sl])
        parts.append(p)
    merged = parts[0]
    for p in parts[1:]:
        merged.merge(p)
    np.testing.assert_allclose(whole.spectrum(), merged.spectrum(), rtol=1e-9,
                               atol=1e-9)


def test_partitioned_eac_reduce_recovers(exp_stream):
    t, y, (a, b) = exp_stream
    acc = PartitionedEAC("a*exp(b*t)", "t", domain=(0, 3), n_windows=8)
    for xc, yc in zip(np.array_split(t, 5), np.array_split(y, 5)):
        acc.update(xc, yc)
    r = acc.fit(p0=[1.0, 1.0])
    assert abs(r.coeffs[1] - b) < 0.2


def test_partitioned_lsi_update_accepts_plain_lists(exp_stream):
    # ``update`` coerces array-likes up front: feeding list/tuple chunks must
    # not crash (the sample count once read ``.shape`` off the raw argument)
    # and must accumulate exactly the ndarray-fed state.
    t, y, _ = exp_stream
    ref = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
    alt = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 3), order=6)
    for k, (xc, yc) in enumerate(zip(np.array_split(t, 5), np.array_split(y, 5))):
        ref.update(xc, yc)
        if k % 2:  # alternate list and tuple chunks
            alt.update(tuple(xc), tuple(yc))
        else:
            alt.update(list(xc), list(yc))
    assert alt.n_samples == ref.n_samples == t.size
    np.testing.assert_array_equal(alt.spectrum(), ref.spectrum())


def test_partitioned_eac_update_accepts_plain_lists(exp_stream):
    t, y, _ = exp_stream
    ref = PartitionedEAC("a*exp(b*t)", "t", domain=(0, 3), n_windows=8)
    alt = PartitionedEAC("a*exp(b*t)", "t", domain=(0, 3), n_windows=8)
    for xc, yc in zip(np.array_split(t, 5), np.array_split(y, 5)):
        ref.update(xc, yc)
        alt.update(list(xc), list(yc))
    assert alt.n_samples == ref.n_samples == t.size
    np.testing.assert_array_equal(
        alt.fit(p0=[1.0, 1.0]).coeffs, ref.fit(p0=[1.0, 1.0]).coeffs
    )


def test_partitioned_lsi_accepts_scalar_chunks():
    """A 0-d/scalar chunk behaves exactly like the equivalent single-sample
    1-d chunk: it concatenates with the boundary carry and counts as 1."""
    x = np.linspace(0.0, 2.0, 21)
    y = 2.0 * np.exp(-1.1 * x)
    ref = PartitionedLSI("a*exp(-b*t)", "t", domain=(0.0, 2.0))
    ref.update(x, y)
    acc = PartitionedLSI("a*exp(-b*t)", "t", domain=(0.0, 2.0))
    acc.update(x[:10], y[:10])
    acc.update(np.array(x[10]), np.array(y[10]))   # 0-d scalar chunk
    acc.update(x[11:], y[11:])
    assert acc.n_samples == ref.n_samples
    assert np.allclose(acc.spectrum(), ref.spectrum())
