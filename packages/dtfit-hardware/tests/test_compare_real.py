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
    # A dropped adaptive_window keyword on the way into imu_track would
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


def _write_noisy_turn_log(path) -> str:
    # 160 fixes at 5 Hz along a quarter-circle at 20 km/h with 1 m of seeded
    # fix noise and the matching yaw rate on the gyro: enough curvature and
    # noise for every process-noise setting to move a forecast.
    n = 160
    dt = 0.2
    speed = 20.0 / 3.6
    rng = np.random.default_rng(3)
    psi = np.linspace(0.0, np.pi / 2.0, n)
    rate = (psi[1] - psi[0]) / dt
    radius = speed / rate
    east = radius * (1.0 - np.cos(psi)) + rng.normal(0.0, 1.0, n)
    north = radius * np.sin(psi) + rng.normal(0.0, 1.0, n)
    lat0, lon0 = 49.0, 30.0
    lat = lat0 + north / 111320.0
    lon = lon0 + east / (111320.0 * np.cos(np.radians(lat0)))
    head = ["t_ms", "lat", "lon", "alt_m", "ax", "ay", "az", "gx", "gy", "gz",
            "mx", "my", "mz", "fix", "sats", "hdop", "spd_kmph"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for i in range(n):
            w.writerow([i * dt * 1000.0, lat[i], lon[i], 0.0,
                        0.0, 0.0, 9.81, 0.0, 0.0, -rate,
                        0.0, 0.0, 0.0, 1, 9, 1.0, 20.0])
    return str(path)


@pytest.fixture
def noisy_turn_log(tmp_path):
    return _write_noisy_turn_log(tmp_path / "noisy_turn.csv")


def _rmse(rows, method, config):
    return [r["rmse"] for r in rows
            if r["method"] == method and r["config"] == config]


def test_config_rows_threads_every_knob(noisy_turn_log) -> None:
    # Each pair differs in one number only; a keyword dropped on the way into
    # imu_track, kalman_track or ekf_track makes its pair identical.
    rows = C.config_rows(
        noisy_turn_log, horizons=(2, 5),
        tracker=[(3, 12, 0.01), (5, 12, 0.01), (3, 20, 0.01), (3, 12, 1.0)],
        kalman_q=[0.05, 5.0], ekf=[(3.0, 0.8), (30.0, 0.8), (3.0, 80.0)])
    assert len(rows) == (4 + 2 + 3) * 2
    assert all(np.isfinite(r["rmse"]) for r in rows)
    base = _rmse(rows, "tracker", "order 3 window 12 q 0.01")
    assert base != _rmse(rows, "tracker", "order 5 window 12 q 0.01")
    assert base != _rmse(rows, "tracker", "order 3 window 20 q 0.01")
    assert base != _rmse(rows, "tracker", "order 3 window 12 q 1")
    assert _rmse(rows, "kalman", "q 0.05") != _rmse(rows, "kalman", "q 5")
    ekf = _rmse(rows, "ct_ekf", "q_acc 3 q_w 0.8")
    assert ekf != _rmse(rows, "ct_ekf", "q_acc 30 q_w 0.8")
    assert ekf != _rmse(rows, "ct_ekf", "q_acc 3 q_w 80")


def test_config_rows_shift_reaches_the_tracker_alone(noisy_turn_log) -> None:
    # A t_shift that never reaches imu_track leaves the tracker rows equal;
    # one that leaks into a baseline moves rows that take the sample period
    # alone.
    kw = dict(horizons=(2,), tracker=[(3, 12, 0.01)], kalman_q=[0.05],
              ekf=[(3.0, 0.8)])
    here = C.config_rows(noisy_turn_log, **kw)
    there = C.config_rows(noisy_turn_log, t_shift=5000.0, **kw)
    by = {r["method"]: r["rmse"] for r in here}
    moved = {r["method"]: r["rmse"] for r in there}
    assert by["tracker"] != moved["tracker"]
    assert by["kalman"] == moved["kalman"]
    assert by["ct_ekf"] == moved["ct_ekf"]


def _cfg(drive, config, rmse, method="tracker", h=2):
    return dict(drive=drive, method=method, config=config, h=h, rmse=rmse)


def test_leave_one_log_out_never_scores_on_the_held_out_log() -> None:
    # "b" is best on log 3 alone and "a" on the other two: a selection that
    # looks at the held-out log hands log 3 its own favourite.
    rows = [_cfg("1", "a", 1.0), _cfg("1", "b", 2.0),
            _cfg("2", "a", 1.0), _cfg("2", "b", 2.0),
            _cfg("3", "a", 5.0), _cfg("3", "b", 0.1)]
    picked = {r["drive"]: r for r in C.leave_one_log_out(rows)}
    assert picked["3"]["config"] == "a"
    assert picked["3"]["rmse"] == 5.0
    # logs 2 and 3 together favour "b" (geometric mean 0.45 against 2.2)
    assert picked["1"]["config"] == "b"


def test_leave_one_log_out_is_scale_free_and_per_method() -> None:
    # On an arithmetic mean the 100 m log would decide alone and pick "a";
    # the mean of logs weighs a halving on either log the same and picks "b".
    rows = [_cfg("long", "a", 100.0), _cfg("long", "b", 110.0),
            _cfg("short", "a", 2.0), _cfg("short", "b", 1.0),
            _cfg("held", "a", 1.0), _cfg("held", "b", 1.0),
            _cfg("long", "k", 3.0, method="kalman"),
            _cfg("short", "k", 3.0, method="kalman"),
            _cfg("held", "k", 3.0, method="kalman")]
    picked = [r for r in C.leave_one_log_out(rows) if r["drive"] == "held"]
    assert [(r["method"], r["config"]) for r in picked] == [
        ("tracker", "b"), ("kalman", "k")]


def test_leave_one_log_out_drops_a_configuration_that_diverged() -> None:
    # Skipping the non-finite row instead of disqualifying its configuration
    # would score "a" on its one good log and pick it.
    rows = [_cfg("1", "a", 0.5), _cfg("1", "b", 1.0),
            _cfg("2", "a", float("nan")), _cfg("2", "b", 1.0),
            _cfg("3", "a", 1.0), _cfg("3", "b", 1.0)]
    picked = {r["drive"]: r["config"] for r in C.leave_one_log_out(rows)}
    assert picked["3"] == "b"
