"""Guards for the horizon/gap sweep: sweep_rows carries the arithmetic and
sweep() must format it without dropping or drifting from a column."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from dtfit_hardware import compare_real as C

HORIZONS = (2, 5, 10)
GAPS = (5, 15)


def _write_synthetic_log(path) -> str:
    # 200 fixes at 5 Hz, moving north at 20 km/h with a level, non-rotating
    # rig (rest never fires, since GPS speed alone already clears the rest
    # threshold), and a magnetometer reading a constant frame offset from
    # true course -- enough for the compass branch and every column to stay
    # finite past warm-up.
    n = 200
    dt = 0.2
    speed = 20.0 / 3.6
    t_ms = np.arange(n) * dt * 1000.0
    lat0, lon0 = 49.0, 30.0
    north_m = speed * np.arange(n) * dt
    lat = lat0 + north_m / 111320.0
    lon = np.full(n, lon0)
    head = ["t_ms", "lat", "lon", "alt_m", "ax", "ay", "az", "gx", "gy", "gz",
            "mx", "my", "mz", "fix", "sats", "hdop", "spd_kmph"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for i in range(n):
            w.writerow([t_ms[i], lat[i], lon[i], 0.0,
                        0.0, 0.0, 9.81, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1, 9, 1.0, 20.0])
    return str(path)


@pytest.fixture
def synthetic_log(tmp_path):
    return _write_synthetic_log(tmp_path / "synthetic_rig.csv")


def test_sweep_rows_returns_one_row_per_horizon_and_gap(synthetic_log) -> None:
    rows = C.sweep_rows(synthetic_log, horizons=HORIZONS, gaps=GAPS)
    assert len(rows) == len(HORIZONS) + len(GAPS)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["forecast"] * len(HORIZONS) + ["coast"] * len(GAPS)
    for r in rows[:len(HORIZONS)]:
        assert np.isfinite(r["gps_ctrl"])
        assert np.isfinite(r["gyro_gated"])
        assert np.isfinite(r["ct_ekf"])
        assert np.isfinite(r["kalman"])
    for r in rows[len(HORIZONS):]:
        assert np.isfinite(r["gps_ctrl"])
        assert np.isfinite(r["gyro_gated"])
        assert np.isfinite(r["ct_ekf"])


def test_sweep_text_reproduces_sweep_rows_numbers(synthetic_log) -> None:
    # every formatted column, not just the first two, must trace back to
    # sweep_rows: dropping or drifting gyro_compass, ct_ekf or kalman from
    # sweep()'s format string must fail this.
    rows = C.sweep_rows(synthetic_log, horizons=HORIZONS, gaps=GAPS)
    text = C.sweep(synthetic_log, HORIZONS, GAPS)
    for r in rows:
        if r["kind"] == "forecast":
            line = (f"  {r['h']:2d}  {r['gps_ctrl']:7.2f}  {r['gyro_gated']:9.2f}  "
                    f"{r['gyro_compass']:11.2f}  {r['ct_ekf']:7.2f}  {r['kalman']:6.2f}")
        else:
            line = (f"  {r['gap']:3d}  {r['gps_ctrl']:7.2f}  {r['gyro_gated']:9.2f}  "
                    f"{r['gyro_compass']:11.2f}  {r['ct_ekf']:7.2f}")
        assert line in text, f"sweep() text is missing the row {r}"


def test_sweep_rows_threads_adaptive_window(synthetic_log) -> None:
    # A dropped adaptive_window keyword on the way into imu_lsi_track would
    # make the fixed-window sweep a silent copy of the adaptive one.
    adaptive = C.sweep_rows(synthetic_log, horizons=(2,), gaps=(),
                            adaptive_window=True)
    fixed = C.sweep_rows(synthetic_log, horizons=(2,), gaps=(),
                         adaptive_window=False)
    assert adaptive[0]["gps_ctrl"] != fixed[0]["gps_ctrl"]


def test_sweep_rows_empty_without_imu_columns(tmp_path) -> None:
    path = tmp_path / "no_imu.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "lat", "lon", "alt_m", "fix"])
        for i in range(40):
            w.writerow([i * 200.0, 49.0 + i * 1e-5, 30.0, 0.0, 1])
    assert C.sweep_rows(str(path)) == []
    assert C.sweep(str(path)) == "sweep: no IMU columns"
