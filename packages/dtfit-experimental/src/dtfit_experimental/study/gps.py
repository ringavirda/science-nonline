"""Simulation and estimation infrastructure for the real-time GPS/inertial
trajectory experiment: a simulated maneuvering 3-D target tracked from a noisy
GPS fix stream and a 9-DOF IMU (3-axis gyro plus accelerometer) with realistic
dropouts and multipath glitches.

* the trajectory and rig generators: :func:`trajectory`, :func:`random_plan`,
  :func:`build_rig`, :func:`build_imu`;
* the dtfit integral trackers: :func:`dtfit_track` (the streaming Legendre
  image and the block image of :class:`dtfit.streaming.ImageFilter`) and the
  full-IMU strapdown :func:`imu_track` (external-regressor Legendre image
  fit), with the :class:`FusedCUSUM` maneuver detector;
* the established baselines: :func:`kalman_track` (constant-accel Kalman) and
  :func:`ekf_track` (gyro-aided coordinated-turn EKF);
* the scoring and batch helpers: :func:`rmse3`, :func:`roll_rmse`,
  :func:`match_onsets` and :func:`run_batch`.

The GPS is modelled at the fix level, truth plus noise: a NEO-M8N puts out no
more than that over NMEA, so the simulated stream mirrors the real rig's.
"""

from __future__ import annotations

import numpy as np

from dtfit.streaming import ImageFilter

from dtfit_experimental.study import baselines as bl
from dtfit_experimental.study import montecarlo

__all__ = [
    "DURATION", "MANEUVERS", "ONSETS", "GPS_SIGMA", "GYRO_SIGMA", "IMU_GYRO_SIGMA",
    "IMU_ACC_SIGMA", "IMU_WASH_TAU", "GRAVITY", "WARMUP", "MODELS",
    "MAG_SIGMA", "MAG_GAIN", "GYRO_BIAS",
    "trajectory", "random_plan", "rmse3", "build_rig", "build_imu", "build_mag",
    "controls", "dtfit_track", "kalman_track", "ekf_track",
    "strapdown_basis", "imu_track", "FusedCUSUM", "roll_rmse", "match_onsets",
    "exp_so3", "batch_trial", "run_batch",
]

# Flight plan: coordinated turns at piecewise-constant turn-rate, speed and
# climb. Heading psi integrates the turn-rate, (x, y) integrate speed*(cos, sin
# psi) and z integrates the climb-rate, so heading, speed and climb all change
# at the onsets. That leaves a maneuvering target with no closed-form per-axis
# formula, which is the honest hard case. The plan below is fully 3-D over 60 s
# and nine segments: sweeping and hard turns of varied rate, accelerations from
# 8 to 16 m/s, climbs and descents from +3 to -2.5 m/s.
DURATION = 60.0     # seconds (10 Hz GPS -> 600 epochs)
MANEUVERS = [
    (0.0,   0.00,  8.0,  2.0),   # straight climb-out
    (6.0,   0.45,  9.0,  2.0),   # sweeping left turn, accelerating, climbing
    (14.0,  0.00, 14.0,  0.0),   # roll out, level cruise
    (20.0, -0.60, 14.0, -2.5),   # hard right descending turn
    (28.0, -0.60, 10.0, -2.5),   # tighten and slow inside the turn
    (34.0,  0.00, 16.0,  1.0),   # roll out, accelerate, gentle climb
    (42.0,  0.80, 16.0,  0.0),   # hard left turn (high rate)
    (48.0,  0.00, 12.0,  3.0),   # roll out, steep climb
    (54.0, -0.35, 12.0, -1.5),   # final easing right turn, descend
]
ONSETS = [m[0] for m in MANEUVERS[1:]]    # true regime-change times
GPS_SIGMA = 1.5     # per-axis fix noise, m (~2.5 m CEP, a NEO-M8N's grade)
GYRO_SIGMA = 0.03   # gyro yaw-rate noise, rad/s
# The full 9-DOF IMU the Nano 33 BLE Sense carries: a dedicated 3-axis MEMS
# gyro, better than the crude yaw channel above, plus a 3-axis accelerometer.
IMU_GYRO_SIGMA = 0.015   # 3-axis gyro noise, rad/s (~0.9 deg/s, IMU-grade)
IMU_ACC_SIGMA = 0.05     # 3-axis accelerometer noise, m/s^2
IMU_WASH_TAU = 100.0     # accel washout time constant, in steps of dt
# A gentler washout leaves more of the low-frequency trajectory arc in the
# accel basis rather than handing it to the drift polynomial. The maneuvers
# live on a 6-8 s scale, so tau has to sit clear of that band: at 10 Hz, 100
# steps is 10 s, which lowers RMSE while staying bounded on random plans.
# Magnetometer, carried by both the NEO-M8N puck and the Nano's BMM150: an
# absolute heading reference that, unlike the gyro, does not drift. It is
# modelled at the heading level as the GPS is at the fix level, which is the
# honest output of a calibrated, tilt-compensated compass: the true course,
# plus a residual hard-iron and declination bias, plus noise.
MAG_SIGMA = 0.05     # compass heading noise, rad (~2.9 deg, tilt-compensated)
MAG_GAIN = 0.05      # complementary mag->yaw correction per step (tau ~ dt/gain ~ 2 s)
# A realistic MEMS yaw-rate bias: small while GPS anchors the track, but it
# integrates into a large heading error across a multi-second GPS dropout,
# which is the error the absolute compass bounds. :func:`build_imu` leaves it
# out unless asked for, so the canonical E1-E7 results are unaffected.
GYRO_BIAS = np.array([0.0, 0.0, 0.01])   # body yaw-rate bias, rad/s (~0.57 deg/s)
GRAVITY = np.array([0.0, 0.0, -9.81])    # world-frame gravity (ENU, z up)
WARMUP = 35         # skip the fill-window transient when scoring


def _cumtrapz(f, t):
    out = np.zeros_like(f, dtype=float)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))
    return out


def controls(t, plan=None):
    """The piecewise-constant turn-rate, speed and climb-rate a flight
    ``plan`` commands at sample times ``t`` (seconds), each holding from its
    onset to the next. ``plan`` defaults to the canonical :data:`MANEUVERS`.
    Returns ``(turn_rate, speed, climb_rate)``, each shape ``(n,)``, in rad/s,
    m/s and m/s."""
    plan = MANEUVERS if plan is None else plan
    t = np.asarray(t, float)
    om, v, zd = (np.zeros_like(t) for _ in range(3))
    for ts, o, s, z in plan:
        m = t >= ts
        om[m], v[m], zd[m] = o, s, z
    return om, v, zd


def trajectory(t, plan=None):
    """Integrate a flight ``plan``'s piecewise-constant controls to a 3-D path.

    Heading integrates the turn-rate, (x, y) integrate speed along the
    current heading and z integrates the climb-rate, all by trapezoidal
    quadrature over ``t``.

    Args:
        t: Sample times, seconds, shape ``(n,)``, increasing.
        plan: Flight plan, the ``(onset, turn_rate, speed, climb)`` segments
            :func:`controls` reads; defaults to :data:`MANEUVERS`.

    Returns:
        Position, metres, shape ``(n, 3)``, starting at ``(2.0, 0.0, 5.0)``.
    """
    om, v, zd = controls(t, plan)
    psi = _cumtrapz(om, t)
    x = 2.0 + _cumtrapz(v * np.cos(psi), t)
    y = 0.0 + _cumtrapz(v * np.sin(psi), t)
    z = 5.0 + _cumtrapz(zd, t)
    return np.stack([x, y, z], axis=-1)


def random_plan(seed, duration=DURATION):
    """Generate a random but realistic coordinated-turn flight plan from a
    seed: a sequence of ``(onset, turn-rate, speed, climb)`` segments covering
    ``duration``. Segments alternate stochastically between straight legs and
    left or right turns of varied rate, at random speed (8-16 m/s) and climb or
    descent (+/-2.5 m/s). The batch test (E6) can then score the approach
    over many distinct trajectories instead of the one hand-built path, so no
    value can be silently tuned to a single track."""
    rng = np.random.default_rng(seed)
    plan, ts = [], 0.0
    while ts < duration - 1e-9:
        if not plan or rng.random() < 0.45:
            om = 0.0                                   # straight leg
        else:
            om = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.8))  # turn
        v = float(rng.uniform(8.0, 16.0))
        zd = float(rng.uniform(-2.5, 2.5)) if rng.random() < 0.7 else 0.0
        plan.append((round(ts, 3), om, v, zd))
        ts += float(rng.uniform(4.0, 9.0))             # segment length
    return plan


def rmse3(a, b):
    """3-D position RMSE between two tracks.

    Args:
        a: Position, metres, shape ``(n, 3)``.
        b: Position, metres, shape ``(n, 3)``, same length as ``a``.

    Returns:
        The root-mean-square Euclidean distance between ``a`` and ``b``,
        metres.
    """
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def build_rig(n, seed=0, *, plan=None, gps_sigma=GPS_SIGMA, gyro_sigma=GYRO_SIGMA,
              glitch_frac=0.0, glitch_mag=12.0):
    """Simulate one pass of the rig: truth, GPS fixes at fix-level noise, and
    the gyro rate.

    ``plan`` selects the flight plan, defaulting to the hand-built canonical
    ``MANEUVERS``; pass a :func:`random_plan` output for the batch test.
    ``gps_sigma`` and ``gyro_sigma`` set the baseline noise. ``glitch_frac``
    injects multipath anomalies, a fraction of the fixes after the warm-up
    window corrupted by N(0, ``glitch_mag``) spikes, which is how the
    separate harsh scenario for the robustness test is built.

    Returns:
        ``(t, truth, fixes, gyro, rng)``: ``t`` the sample times (seconds,
        shape ``(n,)``); ``truth`` the exact position (metres, shape
        ``(n, 3)``); ``fixes`` the noisy GPS position, ``truth`` plus
        per-axis N(0, ``gps_sigma``) and any glitches (metres, shape
        ``(n, 3)``); ``gyro`` the noisy yaw-rate (rad/s, shape ``(n,)``); and
        ``rng`` the generator seeded from ``seed``, left advanced for a
        caller that wants more draws from the same stream.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, n)
    truth = trajectory(t, plan)
    fixes = truth + rng.normal(0, gps_sigma, truth.shape)
    if glitch_frac > 0:
        k = int(glitch_frac * (n - WARMUP))
        idx = rng.choice(np.arange(WARMUP, n), k, replace=False)
        fixes[idx] += rng.normal(0, glitch_mag, (k, 3))
    om, _, _ = controls(t, plan)
    gyro = om + rng.normal(0, gyro_sigma, t.shape)
    return t, truth, fixes, gyro, rng


# The trackers: a dtfit per-axis filter bank and the constant-accel Kalman,
# both handed only a generic local model with no maneuver knowledge, so they
# compete on equal footing.
#
# These are the local models the filter fits over its window. A
# constant-acceleration quadratic is the same class as the Kalman-CA baseline,
# so at best it matches that, and it corner-cuts fast turns into a late
# divergence; the cubic below carries the extra curvature a turn needs, which
# makes it the robust all-round default. The coordinated-turn model
# `c0+c1*t+c2*sin(c3*t+c4)` is nonlinear in its parameters, a circular arc
# being sinusoidal in time, so a linear Kalman-CA cannot represent it at all.
# That is dtfit's differentiator: it wins on the maneuvering segment, at the
# cost of slight overfit on straight runs.
MODELS = {
    "poly": dict(expr="c0 + c1*t + c2*t**2 + c3*t**3", rest=[0.0, 0.0, 0.0], order=4),
    "turn": dict(expr="c0 + c1*t + c2*sin(c3*t + c4)", rest=[0.0, 8.0, 0.6, 0.0], order=5),
}


def _axis_filters(fixes, kind="legendre", model="poly", robust=False, off=None,
                  adaptive_window=True):
    off = off or {}
    m = MODELS[model]
    nq = len(m["rest"]) + 1

    def p0(ax):
        return [float(fixes[0, ax])] + list(m["rest"])

    # The window image self-tunes to the local noise from the residual
    # variance, staying responsive on clean fixes and damping automatically
    # on the noisy, anomaly-heavy harsh stream, with no per-regime
    # hand-tuning; this applies to both bases below.
    if kind == "legendre":   # the Legendre spectrum, right for trajectories
        return [ImageFilter(m["expr"], "t", p0=p0(ax), window_size=15, order=m["order"],
                          q_diag=[1e-2] * nq, adaptive_window=adaptive_window,
                          robust=robust, drift_reset="inflate", **off, basis="legendre") for ax in range(3)]
    if kind == "block":
        return [ImageFilter(m["expr"], "t", p0=p0(ax), window_size=15,
                          order=nq, q_diag=[1e-2] * nq, adaptive_window=adaptive_window,
                          robust=robust, drift_reset="inflate", **off, basis="block") for ax in range(3)]
    raise ValueError(f'kind must be "legendre" or "block", got {kind!r}')


class FusedCUSUM:
    """Pool K streams' one-step residuals into a chi^2(K) NIS and CUSUM it.

    A maneuver moves several axes at once, so the fused statistic carries far
    more SNR than any single axis, which is why a per-axis test misses subtle
    coordinated turns. Residuals are standardized by a running EWMA scale,
    dtfit exposing no innovation covariance of its own, and the CUSUM then
    accumulates surprise above the degrees of freedom."""

    def __init__(self, k, *, ewma=0.05, slack=1.5, h=10.0, warmup=60):
        self.k, self.ewma, self.slack, self.h, self.warmup = k, ewma, slack, h, warmup
        self._sc2 = np.zeros(k)
        self._g = 0.0
        self._n = 0

    def update(self, residuals):
        r = np.asarray(residuals, float)
        if not np.all(np.isfinite(r)):
            return False
        z2 = np.where(self._sc2 > 0, r * r / np.where(self._sc2 > 0, self._sc2, 1.0), 0.0)
        self._sc2 = (1 - self.ewma) * self._sc2 + self.ewma * r * r
        self._n += 1
        if self._n <= self.warmup:
            return False
        self._g = max(0.0, self._g + float(z2.sum()) - self.k - self.slack)
        if self._g > self.h:
            self._g = 0.0
            return True
        return False


def dtfit_track(t, fixes, horizons=(10,), *, kind="legendre", model="poly", robust=False,
                fused=False, gyro=None, coast=True, coast_order=1,
                adaptive_window=True):
    """Online per-axis tracking with rolling h-step forecasts. Missing fixes
    (NaN rows) take no update and the track is extrapolated off the window.
    Returns ``(smoothed, pred, drift_times)``.

    ``coast`` selects how the track is extrapolated off the window's support,
    during a GPS gap and for the h-step forecast. ``True`` (the default)
    dead-reckons from the last in-window sample through
    :meth:`dtfit.streaming.ImageFilter.coast`, constant-velocity at
    ``coast_order=1`` and constant-acceleration at 2. ``False`` evaluates the
    fitted model itself through ``predict``, where a cubic diverges past the
    window. In-window smoothing is identical either way, so ``False`` is the
    matched control for the dropout and forecast scores.

    ``adaptive_window`` is threaded straight into the underlying
    :class:`ImageFilter`: ``True`` (the default) sizes the window from the
    data, ``False`` holds it fixed at ``window_size``.

    Raises:
        ValueError: ``kind`` is neither ``"legendre"`` nor ``"block"``."""
    n = t.size
    sm = np.zeros((n, 3))
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    drift: set[float] = set()
    off = dict(alpha=1e-12, cusum_h=float("inf")) if fused else {}
    flts = _axis_filters(fixes, kind, model, robust, off, adaptive_window)
    det = FusedCUSUM(3 + (gyro is not None)) if fused else None
    g_prev = None

    def _extrap(ax, tt):  # off-support extrapolation: coast vs raw model eval
        q = np.array([tt])
        if coast:
            return float(flts[ax].coast(q, order=coast_order)[0])
        return float(flts[ax].predict(q)[0])

    for i in range(n):
        miss = np.any(np.isnan(fixes[i]))
        for ax in range(3):
            if not miss:
                flts[ax].partial_fit(t[i], fixes[i, ax])
                if not fused and flts[ax].drift_flag_:
                    drift.add(round(float(t[i]), 2))
            sm[i, ax] = _extrap(ax, t[i])
        if det is not None and not miss:
            res = [flts[ax].last_residual_ for ax in range(3)]
            if gyro is not None:                 # add the gyro-rate change channel
                res.append((gyro[i] - g_prev) if g_prev is not None else 0.0)
                g_prev = gyro[i]
            if det.update(res):
                drift.add(round(float(t[i]), 2))
                for ax in range(3):
                    flts[ax].inflate(3.0)
        for h in horizons:
            if i + h < n:
                for ax in range(3):
                    pred[h][i + h, ax] = _extrap(ax, t[i + h])
    return sm, pred, sorted(drift)


def kalman_track(t, fixes, horizons=(10,), *, q=5e-2, adaptive=False):
    """Constant-acceleration Kalman baseline with rolling h-step forecasts.

    A row of ``fixes`` with any NaN is a missed fix: the filter takes a time
    update of one sample period with no measurement
    (:meth:`dtfit_experimental.study.baselines.KalmanCA.coast`), so state and
    covariance are carried through the gap and the next fix is assimilated
    from the coasted state.

    Args:
        t: Sample times, seconds, shape ``(n,)``, evenly spaced.
        fixes: GPS position, metres, shape ``(n, 3)``; a NaN row marks a
            missed fix.
        horizons: Forecast horizons, in samples.
        q: Process-noise scale of the constant-acceleration model.
        adaptive: When ``True``, a :class:`FusedCUSUM` on the innovations
            re-arms the covariance (``inflate(3.0)``, a 3x scale-up) whenever
            it fires.

    Returns:
        ``(smoothed, pred, drift_times)`` with the same shapes as
        :func:`ekf_track`: ``smoothed`` shape ``(n, 3)`` metres; ``pred`` a
        dict mapping each horizon ``h`` to an ``(n, 3)`` array of metres
        where ``pred[h][i + h]`` is the h-step forecast made at epoch ``i``
        (NaN before the first such forecast exists); ``drift_times`` the
        sorted sample times, seconds, where the detector fired.
    """
    kf = bl.KalmanCA(dim=3, dt=float(t[1] - t[0]), q=q, r=0.5)
    det = FusedCUSUM(3) if adaptive else None
    n = t.size
    sm = np.zeros((n, 3))
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    drift: set[float] = set()
    for i in range(n):
        if np.any(np.isnan(fixes[i])):
            sm[i] = kf.coast()
        else:
            sm[i] = kf.update(fixes[i])
            if det is not None and det.update(kf.last_residuals_):
                drift.add(round(float(t[i]), 2))
                kf.inflate(3.0)
        fc = kf.forecast(max(horizons))
        for h in horizons:
            if i + h < n:
                pred[h][i + h] = fc[h - 1]
    return sm, pred, sorted(drift)


def ekf_track(t, fixes, gyro, horizons=(10,), *, adaptive=False, q_acc=3.0,
              q_w=0.8):
    """Gyro-aided coordinated-turn EKF: the fair GPS+IMU recursive baseline. It
    sees the same information as the windowed gyro dead-reckoning, GPS position
    and gyro yaw-rate, but as a textbook EKF. Through a GPS gap it dead-reckons
    on the gyro rather than holding.

    Args:
        t: Sample times, seconds, shape ``(n,)``, evenly spaced.
        fixes: GPS position, metres, shape ``(n, 3)``; a NaN row marks a
            missed fix.
        gyro: Yaw rate about world up, rad/s, shape ``(n,)``; a non-finite
            sample repeats the last finite one.
        horizons: Forecast horizons, in samples.
        adaptive: When ``True``, a :class:`FusedCUSUM` on the innovations
            re-arms the covariance (``inflate(3.0)``) whenever it fires.
        q_acc: Variance of the planar acceleration
            :class:`dtfit_experimental.study.baselines.CTEKFGyro` holds over
            one sample period, m^2/s^4.
        q_w: Turn-rate random-walk process-noise density of the same
            filter, rad^2/s^3.

    Returns:
        ``(smoothed, pred, drift_times)``, shaped as :func:`kalman_track`
        returns them.
    """
    ekf = bl.CTEKFGyro(dt=float(t[1] - t[0]), r_gps=GPS_SIGMA ** 2,
                       r_gyro=GYRO_SIGMA ** 2, q_acc=q_acc, q_w=q_w)
    det = FusedCUSUM(3) if adaptive else None
    n = t.size
    sm = np.zeros((n, 3))
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    drift: set[float] = set()
    since = 0
    last_w = 0.0
    for i in range(n):
        w = float(gyro[i]) if np.isfinite(gyro[i]) else last_w
        last_w = w
        if np.any(np.isnan(fixes[i])):
            since += 1
            sm[i] = ekf.coast(w)                 # IMU-aided dead-reckoning
        else:
            since = 0
            sm[i] = ekf.update(fixes[i], w)
            if det is not None and det.update(ekf.last_residuals_):
                drift.add(round(float(t[i]), 2))
                ekf.inflate(3.0)
        fc = ekf.forecast(max(horizons))
        for h in horizons:
            if i + h < n:
                pred[h][i + h] = fc[h - 1]
    return sm, pred, sorted(drift)


# The full 9-DOF IMU (3-axis gyro plus 3-axis accelerometer), strapdown-fused
# through the external-regressor Legendre image filter. This is the richer model the floor
# argument calls for: the accelerometer adds the acceleration actually sensed
# (speed changes, centripetal and normal load) that a gyro-only constant-speed
# model assumes away, and the 3-axis gyro gives the full 3-D attitude, banked
# turns and climb included.
def _euler_R(psi, th, phi):
    """Body->world rotation from yaw/pitch/roll (ZYX)."""
    cz, sz = np.cos(psi), np.sin(psi)
    cy, sy = np.cos(th), np.sin(th)
    cx, sx = np.cos(phi), np.sin(phi)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def exp_so3(w, dt):
    """Rodrigues exponential of a body-rate increment ``w`` (rad/s, shape
    ``(3,)``) over ``dt`` seconds. Returns the ``(3, 3)`` rotation matrix a
    constant rate ``w`` integrates to; the identity below a 1e-12 rad rotation
    angle."""
    th = w * dt
    a = float(np.linalg.norm(th))
    if a < 1e-12:
        return np.eye(3)
    k = th / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def _log_so3(dR, dt):
    """Body rate that integrates ``dR`` over ``dt`` (the exact inverse of
    :func:`exp_so3`), so the simulated gyro re-integrates to the true attitude."""
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    a = float(np.arccos(c))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return a / (2.0 * np.sin(a)) * v / dt


def build_mag(t, truth, *, mag_sigma=MAG_SIGMA, bias=0.0, seed=0):
    """Simulate a tilt-compensated compass. The absolute world heading (yaw)
    the module reports is the true course ``atan2(v_y, v_x)``, plus a residual
    hard-iron and declination ``bias``, plus ``N(0, mag_sigma)`` noise.
    Modelled at the heading level as the GPS is at the fix level, this is the
    honest output of a calibrated magnetometer, and the one channel a gyro-only
    strapdown lacks: an attitude reference that does not drift. Returns
    ``mag_heading`` of shape ``(n,)`` in radians."""
    rng = np.random.default_rng(seed)
    dt = float(t[1] - t[0])
    vel = np.gradient(truth, dt, axis=0)
    psi = np.arctan2(vel[:, 1], vel[:, 0])
    return psi + bias + rng.normal(0, mag_sigma, psi.shape)


def build_imu(t, truth, *, gyro_sigma=IMU_GYRO_SIGMA, acc_sigma=IMU_ACC_SIGMA,
              gyro_bias=None, mag_sigma=MAG_SIGMA, mag_bias=0.0, seed=0):
    """Simulate the rig's full 9-DOF IMU consistently with the truth
    trajectory: a 3-axis gyro (body angular rate), a 3-axis accelerometer
    (specific force in body) and a 3-axis magnetometer, returned as the
    absolute compass heading it resolves to (see :func:`build_mag`). The rig
    always carries the magnetometer, so it is part of the IMU rather than an
    optional add-on.

    Attitude comes from flight kinematics: heading from the velocity, pitch
    from the climb angle, roll from the coordinated-turn bank
    ``atan2(v*psi_dot, g)``. The gyro rate is the exact relative rotation via
    the matrix log, so integrating it reproduces the attitude, and the
    accelerometer is ``R^T (a_world - g)``, what a strapped-down sensor reads.
    ``gyro_bias`` (rad/s in the body frame, for instance :data:`GYRO_BIAS`)
    adds a constant rate offset, the realistic MEMS error that integrates into
    heading drift through a GPS dropout and that the compass bounds. Returns
    ``(R0, gyro, accel, mag_heading)``, where ``R0`` is the initial attitude,
    the alignment a real rig gets at start-up."""
    rng = np.random.default_rng(seed)
    dt = float(t[1] - t[0])
    m = t.size
    vel = np.gradient(truth, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    vh = np.hypot(vel[:, 0], vel[:, 1]) + 1e-9
    psi = np.arctan2(vel[:, 1], vel[:, 0])
    pitch = np.arctan2(vel[:, 2], vh)
    bank = np.arctan2(vh * np.gradient(np.unwrap(psi), dt), 9.81)
    R = np.array([_euler_R(psi[i], pitch[i], bank[i]) for i in range(m)])
    omega = np.zeros((m, 3))
    for i in range(m - 1):
        omega[i] = _log_so3(R[i].T @ R[i + 1], dt)
    omega[-1] = omega[-2]
    f_body = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), acc - GRAVITY)
    gyro = omega + rng.normal(0, gyro_sigma, omega.shape)
    if gyro_bias is not None:
        gyro = gyro + np.asarray(gyro_bias, float)
    accel = f_body + rng.normal(0, acc_sigma, f_body.shape)
    mag = build_mag(t, truth, mag_sigma=mag_sigma, bias=mag_bias, seed=seed + 1)
    return R[0], gyro, accel, mag


def _wrap(a):
    """Wrap an angle (or array) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_of(R):
    """World-frame yaw (heading) of a body->world rotation ``R``."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def strapdown_basis(t, gyro, accel, R0, *, tau=IMU_WASH_TAU,
                    mag_heading=None, mag_gain=0.0):
    """Strapdown integration of the IMU into a per-axis position basis ``S``.

    Integrate the gyro into attitude, rotate the accelerometer into the world,
    remove gravity, then double-integrate to a position. A raw double integral
    drifts without bound (the classic INS problem: a gravity leak of ~3 m/s^2
    from any attitude error becomes thousands of metres), and a windowed filter
    cannot absorb that. So the integration is a leaky washout of time constant
    ``tau``, which keeps the basis bounded and drift-free while retaining the
    maneuver content the accelerometer senses. The residual smooth drift is
    mopped up by the model's polynomial-drift terms. Returns ``S`` of shape
    ``(n, 3)``.

    Given ``mag_heading``, a complementary yaw correction of strength
    ``mag_gain`` nudges the integrated attitude toward the absolute compass
    heading each step (``R <- Rz(mag_gain * delta_yaw) R``). The gyro still
    supplies the smooth, high-rate attitude and the magnetometer only bounds
    its slow yaw drift. This runs independently of the GPS, so it keeps the
    dead-reckoned heading honest through a GPS dropout, where a gyro bias would
    otherwise curve the coast off-course."""
    dt = float(t[1] - t[0])
    R = R0.copy()
    m = t.size
    S = np.zeros((m, 3))
    v = np.zeros(3)
    s = np.zeros(3)
    a = dt / tau
    use_mag = mag_heading is not None and mag_gain > 0.0
    for i in range(m):
        aw = R @ accel[i] + GRAVITY
        R = R @ exp_so3(gyro[i], dt)
        if use_mag:
            e = _wrap(float(mag_heading[i]) - _yaw_of(R)) * mag_gain
            c, sn = np.cos(e), np.sin(e)
            R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]) @ R
        v = (1.0 - a) * v + aw * dt
        s = (1.0 - a) * s + v * dt
        S[i] = s
    return S


def imu_track(t, fixes, gyro, accel, R0, horizons=(10,), *, window=28,
                  drift="c2*tt**2", mag_heading=None, mag_gain=MAG_GAIN, S=None,
                  adaptive_window=True, order=6, q=1e-2):
    """Full-IMU GPS fusion, run per axis entirely through dtfit's Legendre image filter.

    The strapdown basis ``S`` (gyro attitude plus accelerometer, washed out) is
    fed to :class:`the Legendre filter` as an external regressor, making the per-axis
    model ``c0 + c1*t + (polynomial drift) + S``. The accelerometer supplies
    the sensed motion shape, the polynomial absorbs the residual INS drift and
    the GPS anchors the absolute trajectory, all fused by the integral
    Legendre-spectrum measurement. Missing fixes (NaN rows) coast on the IMU
    basis. Returns ``(smoothed, pred)``.

    Passing ``mag_heading``, with ``mag_gain`` such as :data:`MAG_GAIN`, folds
    an absolute compass into the strapdown attitude (see
    :func:`strapdown_basis`): the gyro-integrated heading is anchored to
    magnetic north so the IMU coast does not yaw away during a GPS dropout. It
    adds nothing while GPS is healthy, the fix already pinning the heading, so
    its whole gain is concentrated in the gaps.

    The drift compensator is a quadratic (``c2*tt**2``) rather than a cubic.
    With the accelerometer already supplying the motion shape, a cubic drift
    term overfits the GPS noise on clean fixes and, having no data to anchor
    it, extrapolates explosively while coasting through a gap. The quadratic is
    both more accurate on clean smoothing and far more stable during dropouts.
    It is paired with a ``window`` of 28 samples, which the order-6 projection
    wants room for.

    ``adaptive_window`` is threaded straight into the underlying
    :class:`ImageFilter`: ``True`` (the default) sizes the window from the
    data, ``False`` holds it fixed at ``window``. ``order`` is the Legendre
    image order and ``q`` the process-noise variance added to every
    parameter at each update (``q_diag=[q] * n_params``)."""
    n = t.size
    sm = np.zeros((n, 3))
    ax = ["Sx", "Sy", "Sz"]
    # ``S`` may arrive precomputed, e.g. from a rest-aided real-IMU strapdown
    # that estimates bias online; otherwise build the clean-IMU basis as the
    # sim does.
    if S is None:
        S = strapdown_basis(t, gyro, accel, R0, mag_heading=mag_heading, mag_gain=mag_gain)
    nq = 2 + (drift.count("c") if drift else 0)

    def expr(a):
        if drift:
            return f"c0 + c1*tt + {drift} + {ax[a]}"
        return f"c0 + c1*tt + {ax[a]}"

    flts = [ImageFilter(expr(a), "tt", regressors=ax[a],
                      p0=[float(fixes[0, a])] + [0.0] * (nq - 1), window_size=window,
                      order=order, q_diag=[q] * nq, adaptive_window=adaptive_window,
                      drift_reset="inflate", basis="legendre") for a in range(3)]
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    for i in range(n):
        miss = np.any(np.isnan(fixes[i]))
        for a in range(3):
            if not miss:
                flts[a].partial_fit(t[i], fixes[i, a], regressors={ax[a]: S[i, a]})
            sm[i, a] = float(flts[a].predict(np.array([t[i]]),
                                             regressors={ax[a]: S[i, a]})[0])
        # Forecast by extrapolating the IMU-sensed motion: hold the smoothed
        # estimate's local velocity, a finite difference of the fused track,
        # forward. The model's polynomial drift-compensation terms are a local
        # nuisance fit, so evaluating the model at a future time would extrapolate
        # them and blow up. The smoothed-track velocity already carries the
        # accelerometer's information without that pathology.
        for h in horizons:
            if i >= 1 and i + h < n:
                pred[h][i + h] = sm[i] + h * (sm[i] - sm[i - 1])
    return sm, pred


def roll_rmse(pred_h, truth, mask=None):
    """RMSE of one horizon's rolling forecast against truth.

    Only samples with a defined forecast, past :data:`WARMUP` and, if given,
    inside ``mask``, are scored.

    Args:
        pred_h: One horizon's forecast, metres, shape ``(n, 3)``, as
            ``pred[h]`` from :func:`dtfit_track`, :func:`kalman_track` or
            :func:`ekf_track`; NaN rows (no forecast yet) are excluded.
        truth: True position, metres, shape ``(n, 3)``.
        mask: Optional boolean array, shape ``(n,)``, further restricting the
            scored samples.

    Returns:
        The RMSE (metres) over the scored samples, or NaN if none remain.
    """
    m = ~np.isnan(pred_h[:, 0])
    m[:WARMUP] = False
    if mask is not None:
        m &= mask
    return rmse3(pred_h[m], truth[m]) if m.any() else float("nan")


def match_onsets(flags):
    """Score a detector's flag times against the true maneuver onsets.

    An onset in :data:`ONSETS` counts as caught if some flag falls in its
    ``[onset - 0.3, onset + 1.5]`` window (seconds); a flag outside every
    onset's window counts as a false alarm.

    Args:
        flags: Detector flag times, seconds, such as the ``drift_times``
            returned by :func:`dtfit_track`, :func:`kalman_track` or
            :func:`ekf_track`.

    Returns:
        ``(caught, false_alarms, median_latency)``: ``caught`` the count of
        onsets with a matching flag; ``false_alarms`` the count of flags
        matching no onset; ``median_latency`` the median, over caught
        onsets, of (earliest matching flag - onset) in seconds, or NaN if no
        onset was caught.
    """
    caught = sum(any(o - 0.3 <= f <= o + 1.5 for f in flags) for o in ONSETS)
    fa = sum(not any(o - 0.3 <= f <= o + 1.5 for o in ONSETS) for f in flags)
    lat = []
    for o in ONSETS:
        hit = [f for f in flags if o - 0.3 <= f <= o + 1.5]
        if hit:
            lat.append(min(hit) - o)
    return caught, fa, (float(np.median(lat)) if lat else float("nan"))


def batch_trial(arg):
    """One random-trajectory trial for the random-plan batch test, at module
    level so a process pool can pickle it. ``arg`` is ``(trial_index, n)``,
    ``n`` the number of simulated GPS epochs. Returns
    ``(trial_index, {method: smoothing RMSE}, sample-trajectory-or-None)``,
    the sample trajectory carried for the first six trials only.

    The plan, GPS-noise and IMU-noise seeds are drawn as three independent
    children of a per-trial :class:`numpy.random.SeedSequence`, so they are
    decorrelated streams rather than correlated integer offsets, which could
    alias structure across one trial's plan, fixes and IMU. Each child is
    realized as a concrete integer seed through ``generate_state``, since
    ``build_imu`` requires an integer seed and derives the mag seed from it as
    ``seed + 1``."""
    j, n = arg
    children = np.random.SeedSequence(j).spawn(3)
    s_plan, s_rig, s_imu = (int(c.generate_state(1)[0]) % (2 ** 31) for c in children)
    plan = random_plan(seed=s_plan)
    t, truth, fixes, gyro, _ = build_rig(n, seed=s_rig, plan=plan)
    R0, gy3, ac3, mg3 = build_imu(t, truth, seed=s_imu)
    msk = np.ones(n, bool)
    msk[:WARMUP] = False
    out = {
        "raw": rmse3(fixes[msk], truth[msk]),
        "legendre": rmse3(dtfit_track(t, fixes, (1,), kind="legendre")[0][msk], truth[msk]),
        "imu": rmse3(imu_track(t, fixes, gy3, ac3, R0, (1,),
                                   mag_heading=mg3)[0][msk], truth[msk]),
        "ekf": rmse3(ekf_track(t, fixes, gyro, (1,))[0][msk], truth[msk]),
        "kal": rmse3(kalman_track(t, fixes, (1,))[0][msk], truth[msk]),
    }
    return j, out, (truth if j < 6 else None)


def run_batch(n_traj, n):
    """Run ``n_traj`` independent :func:`batch_trial` calls, each simulating
    ``n`` GPS epochs, through
    :func:`dtfit_experimental.study.montecarlo.pool_map` (one BLAS thread a
    worker), with no serial fallback. ``batch_trial`` is importable at module
    level so the pool's workers, spawned fresh rather than forked, can
    import and pickle it. Calling this from inside a daemonic worker process
    raises ``AssertionError("daemonic processes are not allowed to have
    children")``, since ``ProcessPoolExecutor`` cannot start its own workers
    there. Returns the trials' results in trial order."""
    args = [(j, n) for j in range(n_traj)]
    return montecarlo.pool_map(batch_trial, args)
