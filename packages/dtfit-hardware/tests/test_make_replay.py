"""Guards for the replay-header codegen: the round trip a mangled float
literal or a dropped array would break silently on the board."""
from __future__ import annotations

import numpy as np

from dtfit_hardware.tools import make_replay


def test_header_roundtrips_float32_exactly(tmp_path) -> None:
    # .9g is enough decimal digits to recover a float32 bit-for-bit; shorten
    # the format spec in render()/_lit() (e.g. to .5g) and this fails.
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 1300.0, 37).astype(np.float32)
    e = rng.uniform(-500.0, 500.0, 37).astype(np.float32)
    n = rng.uniform(-500.0, 500.0, 37).astype(np.float32)
    out = tmp_path / "replay_vec.h"
    out.write_text(make_replay.render(t, e, n, "synthetic.csv"), encoding="utf-8")

    t2, e2, n2 = make_replay.read(out)

    assert t2.dtype == np.float32
    assert np.array_equal(t, t2)
    assert np.array_equal(e, e2)
    assert np.array_equal(n, n2)


def test_render_records_array_length_and_source(tmp_path) -> None:
    # Dropping a row on the way in would silently ship a short replay: guard
    # RP_N against the array size and the source comment against the log name.
    t = np.array([0.0, 0.2, 0.4], dtype=np.float32)
    e = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    n = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    text = make_replay.render(t, e, n, "drive_5hz_20260702_182933.csv")

    assert "#define RP_N 3" in text
    assert "drive_5hz_20260702_182933.csv" in text
    out = tmp_path / "replay_vec.h"
    out.write_text(text, encoding="utf-8")
    t2, e2, n2 = make_replay.read(out)
    assert t2.size == e2.size == n2.size == 3
