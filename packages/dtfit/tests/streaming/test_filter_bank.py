"""FilterBank -- a bank of independent streaming filters runs in parallel."""

from typing import Any

import numpy as np

from dtfit.streaming import EDAFilter, FilterBank


def _streams(K=5, n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, n)
    bs = np.linspace(0.5, 1.0, K)
    Y = np.column_stack(
        [1.0 * np.exp(b * t) + rng.normal(0, 0.02, n) for b in bs]
    )
    return t, Y, bs


def test_filter_bank_recovers_per_stream_params():
    t, Y, bs = _streams()
    bank = FilterBank.from_model(
        "a*exp(b*t)", "t", len(bs), p0=[1.0, 0.3], window_size=40,
        q_diag=[1e-4, 1e-3], r=0.3,
    )
    out = bank.run(t, Y, n_jobs=1)
    est_b = out["params"][:, 1]
    np.testing.assert_allclose(est_b, bs, atol=0.1)


def test_filter_bank_threaded_matches_serial():
    t, Y, bs = _streams()
    kw: dict[str, Any] = dict(
        p0=[1.0, 0.3], window_size=40, q_diag=[1e-4, 1e-3], r=0.3
    )
    a = FilterBank.from_model("a*exp(b*t)", "t", len(bs), **kw).run(t, Y, n_jobs=1)
    b = FilterBank.from_model("a*exp(b*t)", "t", len(bs), **kw).run(t, Y, n_jobs=4)
    np.testing.assert_allclose(a["params"], b["params"], rtol=1e-9, atol=1e-9)


def test_filter_bank_matches_standalone_filters():
    t, Y, bs = _streams(K=3)
    kw: dict[str, Any] = dict(
        p0=[1.0, 0.3], window_size=40, q_diag=[1e-4, 1e-3], r=0.3
    )
    bank = FilterBank.from_model("a*exp(b*t)", "t", 3, **kw)
    bank.run(t, Y, n_jobs=1)
    for k in range(3):
        flt = EDAFilter("a*exp(b*t)", "t", **kw)
        for s in range(t.size):
            flt.partial_fit(t[s], Y[s, k])
        np.testing.assert_allclose(bank[k].p, flt.p, rtol=1e-9, atol=1e-9)


def test_filter_bank_predict_and_readout_shapes():
    t, Y, bs = _streams(K=4)
    bank = FilterBank.from_model("a*exp(b*t)", "t", 4, p0=[1.0, 0.3],
                                 window_size=40)
    bank.run(t, Y, n_jobs=1)
    assert bank.params_array().shape == (4, 2)
    assert bank.predict(t[:6]).shape == (4, 6)
    assert bank.predict(t[:1]).shape == (4,)
    assert len(bank.params_) == 4
