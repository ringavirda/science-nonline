"""The twelve series loaders and the SERIES catalogue.

Each test names the mutation of series.py it fails under.
"""

from __future__ import annotations

import numpy as np
import pytest

from dtfit_experimental.study import series as S


def test_series_catalogue_has_twelve_entries():
    assert len(S.SERIES) == 12


# Fails when a loader's seed or sample count drifts.
@pytest.mark.parametrize("loader, n, period", [
    (S.load_rlc_transient, 360, 90.0),
    (S.load_ac_harmonics, 360, 60.0),
    (S.load_am_signal, 400, 400 / 9.0),
    (S.load_chirp, 400, 400 / 11.0),
])
def test_generated_waveforms_reproducible(loader, n, period):
    from dtfit._signal import dominant_period

    y1 = loader()
    y2 = loader()
    assert y1.size == n
    np.testing.assert_array_equal(y1, y2)
    period_samp, strength = dominant_period(y1)
    assert strength > 0.05
    assert period_samp == pytest.approx(period, rel=0.05)


# Fails when a loader raises FileNotFoundError instead of returning None.
def test_loader_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "data_dir", lambda: tmp_path)
    assert S.load_uah() is None
    assert S.load_covid() is None
