"""The twelve series loaders and the SERIES catalogue.

Each test names the mutation of series.py it fails under.
"""

from __future__ import annotations

import numpy as np
import pytest

from dtfit_experimental.study import series as S


def test_series_catalogue_has_twelve_entries():
    assert len(S.SERIES) == 12


# Fails when a loader's seed or sample count drifts. The golden first sample
# pins the seed: a reproducibility check across two calls of the same loader
# passes for any seed, so it alone would miss a drifted seed.
@pytest.mark.parametrize("loader, n, period, y0", [
    (S.load_rlc_transient, 360, 90.0, 0.0006838553450636834),
    (S.load_ac_harmonics, 360, 60.0, -0.00020480339596569536),
    (S.load_am_signal, 400, 400 / 9.0, 0.05480269679872269),
    (S.load_chirp, 400, 400 / 11.0, 0.020865593101145056),
])
def test_generated_waveforms_reproducible(loader, n, period, y0):
    from dtfit._signal import dominant_period

    y1 = loader()
    y2 = loader()
    assert y1.size == n
    np.testing.assert_array_equal(y1, y2)
    assert y1[0] == pytest.approx(y0)
    period_samp, strength = dominant_period(y1)
    assert strength > 0.05
    assert period_samp == pytest.approx(period, rel=0.05)


# Fails when a loader raises FileNotFoundError instead of returning None.
def test_loader_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "data_dir", lambda: tmp_path)
    assert S.load_uah() is None
    assert S.load_covid() is None


# Fails if load_ltsf catches the bad-name KeyError instead of letting it
# propagate.
def test_load_ltsf_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        S.load_ltsf("not-a-dataset")


# Fails if load_covid stops raising StopIteration when the series never
# crosses 500 (for example, a fixed sentinel index that silently clamps).
def test_load_covid_never_reaching_500_raises(monkeypatch):
    monkeypatch.setattr(S, "_csv", lambda name, col=1, start_row=1: np.array([1.0, 2.0, 3.0]))
    with pytest.raises(StopIteration):
        S.load_covid()
