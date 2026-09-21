"""The GPS simulation and benchmark modules: the trajectory generator, the
dtfit and baseline trackers, the batch driver and the GSDC loader.

Each test names the mutation of gps.py / gps_benchmark.py it fails under.
"""

import csv
import math

import numpy as np
import pytest

from dtfit_experimental.study import gps
from dtfit_experimental.study import gps_benchmark as gb


def test_build_rig_carries_gps_sigma_and_turns_by_the_commanded_rate():
    plan = [(0.0, 0.0, 10.0, 0.0), (5.0, 0.5, 10.0, 0.0)]
    t, truth, fixes, gyro, rng = gps.build_rig(200, seed=0, plan=plan, gps_sigma=2.5)
    # fails if build_rig ignored gps_sigma (e.g. a hardcoded noise scale)
    assert abs(float((fixes - truth).std()) - 2.5) < 0.3
    # fails under a sign or scale slip in the coordinated-turn heading
    # integration: the true heading rate would then not match the plan's
    # commanded turn rate of 0.5 rad/s over the turn segment
    mask = t >= 6.0
    vel = np.diff(truth[mask], axis=0)
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    rate = np.diff(heading) / np.diff(t[mask])[:-1]
    assert rate == pytest.approx(0.5, abs=0.05)


def test_dtfit_track_rejects_the_retired_lowercase_kind():
    t, truth, fixes, gyro, rng = gps.build_rig(50, seed=1, gps_sigma=1.0)
    # fails if _axis_filters still fell through to "block" for any
    # kind other than "legendre" (the guard the retired name relies on)
    with pytest.raises(ValueError, match="legendre.*block"):
        gps.dtfit_track(t, fixes, (1,), kind="lsi")


def test_dtfit_track_adaptive_window_keyword_reaches_the_filter():
    t, truth, fixes, gyro, rng = gps.build_rig(200, seed=1, gps_sigma=1.0)
    sm_adaptive, *_ = gps.dtfit_track(t, fixes, (1,), adaptive_window=True)
    sm_fixed, *_ = gps.dtfit_track(t, fixes, (1,), adaptive_window=False)
    # fails if adaptive_window were dropped on the way to ImageFilter (always
    # True), which would make the fixed-window track a copy of the adaptive one
    assert not np.allclose(sm_adaptive, sm_fixed)
    # fails if partial_fit were never reached (a track frozen at its p0)
    assert not np.allclose(sm_fixed[0], sm_fixed[-1])


def test_kalman_and_ekf_recover_a_noise_free_constant_velocity_path():
    plan = [(0.0, 0.0, 10.0, 0.0)]
    t, truth, fixes, gyro, rng = gps.build_rig(200, seed=2, plan=plan,
                                               gps_sigma=0.0, gyro_sigma=0.0)
    sm_k, *_ = gps.kalman_track(t, fixes, (1,))
    sm_e, *_ = gps.ekf_track(t, fixes, gyro, (1,))
    warm = gps.WARMUP
    # fails under a wrong dt or transition (e.g. a hardcoded sample period
    # shorter than the one in t)
    assert float(np.max(np.abs(sm_k[warm:] - truth[warm:]))) < 0.02
    assert float(np.max(np.abs(sm_e[warm:] - truth[warm:]))) < 0.02


def test_kalman_track_carries_its_state_through_a_gap():
    plan = [(0.0, 0.0, 10.0, 0.0)]
    t, truth, fixes, gyro, rng = gps.build_rig(200, seed=2, plan=plan,
                                               gps_sigma=0.0, gyro_sigma=0.0)
    gapped = fixes.copy()
    gapped[100:120] = np.nan
    sm, pred, _ = gps.kalman_track(t, gapped, (1,))
    # fails if a missed fix only extrapolates for display and leaves the
    # filter state at the last fix: the first update after the gap then
    # starts 20 samples behind and lands tens of metres off
    assert float(np.max(np.abs(sm[100:125] - truth[100:125]))) < 0.05
    assert float(np.max(np.abs(pred[1][101:125] - truth[101:125]))) < 0.05


def test_run_batch_is_finite_and_reproducible_per_trial_index():
    results = gps.run_batch(2, 300)
    assert [j for j, _, _ in results] == [0, 1]
    for _, out, _ in results:
        # fails if any method's RMSE came out NaN or infinite
        assert all(math.isfinite(v) for v in out.values())
    _, out0a, _ = gps.batch_trial((0, 300))
    _, out0b, _ = gps.batch_trial((0, 300))
    _, out1, _ = gps.batch_trial((1, 300))
    # fails under a global rng leaking between trials: a fixed trial index
    # would then not reproduce the same result on a second call
    assert out0a == out0b
    # fails if every trial silently drew the same stream regardless of index
    assert out0a != out1


def _geodetic_to_ecef(lat_deg, lon_deg, alt=0.0):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    x = (n + alt) * math.cos(lat) * math.cos(lon)
    y = (n + alt) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - e2) + alt) * math.sin(lat)
    return x, y, z


def test_load_gsdc_drops_ground_truth_rows_outside_tol_ms(tmp_path):
    lat0, lon0 = 40.0, -75.0
    gnss_path = tmp_path / "gnss.csv"
    gt_path = tmp_path / "gt.csv"
    wls_times = list(range(0, 5000, 100))
    with open(gnss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utcTimeMillis", "WlsPositionXEcefMeters",
                                          "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"])
        w.writeheader()
        for tm in wls_times:
            x, y, z = _geodetic_to_ecef(lat0 + 1e-6 * tm / 1000.0, lon0)
            w.writerow(dict(utcTimeMillis=tm, WlsPositionXEcefMeters=x,
                            WlsPositionYEcefMeters=y, WlsPositionZEcefMeters=z))
    # one ground-truth epoch (10000) is far past the last WLS fix (4900), well
    # outside tol_ms, the rest sit exactly on a WLS epoch
    gt_times = [0, 1000, 2000, 3000, 4000, 10000]
    with open(gt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["UnixTimeMillis", "LatitudeDegrees",
                                          "LongitudeDegrees", "AltitudeMeters"])
        w.writeheader()
        for tm in gt_times:
            w.writerow(dict(UnixTimeMillis=tm, LatitudeDegrees=lat0 + 1e-6 * tm / 1000.0,
                            LongitudeDegrees=lon0, AltitudeMeters=0.0))
    t, truth, meas, labels = gb.load_gsdc(str(gnss_path), str(gt_path), tol_ms=500)
    # fails if the tol_ms gate were ignored: the out-of-range epoch would stay
    assert len(t) == len(gt_times) - 1


def test_run_methods_realdata_survives_an_outlier_and_guards_a_small_window():
    rng = np.random.default_rng(0)
    n = 60
    t = np.arange(n, dtype=float)
    truth = np.stack([t * 2.0, t * 0.5, np.zeros(n)], axis=1)
    meas = truth + rng.normal(0.0, 1.0, truth.shape)
    meas[30, 0] += 300.0
    res = gb.run_methods_realdata(t, meas, window=6)
    # fails if the IMM likelihood overflowed to NaN mixing probabilities on
    # the 300 m multipath spike
    for name, est in res.items():
        assert np.all(np.isfinite(est)), name
    # fails if a window too small to hold the image were let through silently
    with pytest.raises(ValueError, match="cannot hold an image"):
        gb.run_methods_realdata(t, meas, window=3)
