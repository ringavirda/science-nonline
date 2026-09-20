"""study.processes: the ground-truth generators, the router's applicability
map and the real-series catalogue.
"""

import numpy as np
import pytest

from dtfit_experimental.study.classical_stochastic import garch_mle_persistence
from dtfit_experimental.study.processes import (
    gen_arfima,
    gen_ar1,
    gen_garch,
    gen_ar2_cycle,
    gen_trend_cycle,
    ROUTER_CASES,
    regime_matches,
    REAL_SERIES,
    suite_horizon,
)


def test_gen_ar1_and_gen_arfima_are_driven_by_their_parameter():
    # fails if phi stops driving gen_ar1 (e.g. a copy-paste that always
    # returns white noise) or d stops driving gen_arfima's fractional
    # integration (the MA(inf) truncation collapsing to identity)
    x = gen_ar1(4000, 0.7, np.random.default_rng(0))
    acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    assert abs(acf1 - 0.7) < 0.05

    from dtfit.stochastic import hurst_aggvar

    white = np.random.default_rng(1).standard_normal(4096)
    h_white = hurst_aggvar(white)["H"]
    long_memory = gen_arfima(4096, 0.3, np.random.default_rng(2))
    h_lm = hurst_aggvar(long_memory)["H"]
    assert h_lm > h_white


def test_gen_garch_persistence_matches_alpha_plus_beta():
    # fails if omega, alpha and beta are swapped in the call signature:
    # both parameter settings must recover alpha + beta, since a rotated
    # signature only breaks one of the two
    for omega, alpha, beta, persistence in [(0.05, 0.08, 0.90, 0.98),
                                             (0.05, 0.20, 0.60, 0.80)]:
        r = gen_garch(4000, omega, alpha, beta, np.random.default_rng(0))
        assert garch_mle_persistence(r) == pytest.approx(persistence, abs=0.05)


def test_gen_ar2_cycle_peaks_near_its_period():
    # fails on a phi1/phi2 sign slip, which moves the spectral peak off P
    x = gen_ar2_cycle(1500, 16.0, 0.97, np.random.default_rng(0))
    xx = x - x.mean()
    spec = np.abs(np.fft.rfft(xx)) ** 2
    freqs = np.fft.rfftfreq(xx.size, d=1.0)
    spec[0] = 0.0
    pk = freqs[int(np.argmax(spec))]
    period = 1.0 / pk if pk > 0 else float("inf")
    assert abs(period - 16.0) / 16.0 < 0.10


def test_gen_trend_cycle_shape_and_recovered_slope():
    # fails if the intercept and slope are swapped in the return
    t, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0,
                           np.random.default_rng(0))
    assert t.shape == y.shape == (600,)
    slope, _ = np.polyfit(t, y, 1)
    assert abs(slope - 0.02) / 0.02 < 0.3


def test_router_cases_are_seeded_and_do_not_leak_a_global_rng():
    # fails if a case's generator ignores its own seed and leaks a shared rng
    assert len(ROUTER_CASES) == 7
    for _, _, gen in ROUTER_CASES:
        a = np.asarray(gen(0))
        b = np.asarray(gen(0))
        c = np.asarray(gen(1))
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class _FakeModel:
    def __init__(self, regime, has_vol_clustering=False):
        self.regime = regime
        self.has_vol_clustering = has_vol_clustering


def test_regime_matches_accepts_the_true_label_and_rejects_a_wrong_one():
    # fails on regime_matches' own rejection logic, e.g. a missing
    # "not-a-real-regime" branch falling through to True
    labels = {
        "white noise": "white noise",
        "random walk": "random walk (unit root)",
        "mean-reverting": "mean-reverting (AR(1))",
        "long-memory": "long-memory (ARFIMA)",
        "cyclical": "cyclical (AR(2))",
        "trend+cycle": "trend+cycle",
    }
    for expected, regime in labels.items():
        assert regime_matches(_FakeModel(regime), expected)
        assert not regime_matches(_FakeModel(regime), "not-a-real-regime")
    assert regime_matches(_FakeModel("x", has_vol_clustering=True),
                          "vol-clustering")
    assert not regime_matches(_FakeModel("x", has_vol_clustering=False),
                              "vol-clustering")


def test_regime_matches_tracks_dtfits_actual_regime_labels():
    # fails if a regime string is renamed in dtfit.stochastic (e.g.
    # gates.py's "cyclical" -> "cyclicX", "seasonal" -> "seasonalX"),
    # since this fits real StochasticModel objects instead of the
    # _FakeModel strings the test above writes itself
    from dtfit.stochastic import fit_stochastic

    for _, expect, gen in ROUTER_CASES:
        model = fit_stochastic(gen(0))
        assert regime_matches(model, expect), (expect, model.regime)


def test_real_series_entries_load_or_return_none_with_one_line(capsys):
    # fails if a loader lets its exception propagate, or prints more than
    # one line
    for entry in REAL_SERIES:
        y = entry.load()
        out = capsys.readouterr().out
        if y is None:
            assert out.count("\n") == 1
        else:
            arr = np.asarray(y, dtype=float)
            assert arr.ndim == 1
            assert np.all(np.isfinite(arr))


def test_guarded_loader_returns_none_with_one_line_when_data_dir_is_missing(
    monkeypatch, capsys,
):
    # fails if _guarded's try/except is removed, or catches nothing
    monkeypatch.setenv("DTFIT_DATA", "/no/such/dtfit-data-dir")
    entry = next(e for e in REAL_SERIES if e.key == "usd_uah")
    y = entry.load()
    out = capsys.readouterr().out
    assert y is None
    assert out.count("\n") == 1


def test_suite_horizon_clamps_both_ends():
    # fails if the clamp's min/max are inverted
    assert suite_horizon(100) == 12
    assert suite_horizon(4000) == 40
