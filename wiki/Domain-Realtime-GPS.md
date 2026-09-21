# Domain -- Real-time GPS and inertial trajectory tracking

Smooth, forecast and coast a moving target's position from GPS fixes, with and
without an IMU, through gaps and multipath glitches, against the recursive
trackers a navigation engineer would reach for first.

**Notebooks:**
[33_gps_simulation](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/33_gps_simulation.ipynb)
(a simulated maneuvering target with a 9-DOF IMU),
[34_gps_benchmark](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/34_gps_benchmark.ipynb)
(two literature trajectories with closed-form truth, two GSDC 2022 trips with
RTK truth). Recorded drives and the chip: [GPS hardware rig](Domain-Realtime-GPS-Hardware).

## Routes and baselines

- **dtfit:** the streaming window tracker per axis in the Legendre and the
  block basis (a cubic or quadratic in time), dead-reckoning through a gap from
  the last in-window sample, IMU channels as external regressors, the full-IMU
  strapdown form, the reweighted (robust) window, the fused maneuver detector.
- **Baselines:** a constant-acceleration Kalman filter, a gyro-aided
  coordinated-turn EKF, an IMM, a Huber-hardened Kalman filter.

## What is measured

**Smoothing.** With the full IMU the Legendre tracker smooths at or below the
CT-EKF. GPS-only, the Legendre tracker and Kalman-CA are at parity: 1.622 m
against 1.687 m on the single flight, 1.524 m against 1.508 m in the median
over 24 random flights, split 13 to 11. On the literature trajectories clean
smoothing is near parity as well (coordinated turn 1.26 against 1.27 m;
figure-8 1.14 against 0.99 m). The block tracker trails the Legendre one on a
smoothly curving path.

**Through a gap.** Gyro-aided trackers stay below GPS-only ones at every gap
length. Of the GPS-only trackers the Legendre one, dead-reckoning at constant
velocity, stays 6 to 19 percent below Kalman-CA inside the gap and 2 to 21
percent below its own extrapolated cubic. On the coordinated turn it is level
with Kalman-CA (8.40 against 8.65 m); extrapolating the fitted cubic instead
costs 11 and 28 percent more on the two literature trajectories.

**Multipath.** Reweighting the window beats the plain fit at every glitch
fraction, widest at 5 percent (2.02 against 2.82 m). On the real urban-canyon
trip the reweighted fit and the Huber Kalman are the only routes that recover
the track, an order of magnitude ahead of their plain versions (20 and 43
times), and the reweighted fit stays within 7 percent of the Huber Kalman at
the shortest window.

**Maneuver detection.** The fused detector needs the gyro-rate channel to
lead: with it, it wins on catch rate and latency and raises more false alarms
than the CT-EKF's own residual test; on position alone it catches fewer onsets.

## Where dtfit loses

- The 10-step forecast goes to the CT-EKF: its coupled turn state rolls
  forward more accurately than a per-axis extrapolation.
- Scored over the whole track with dropouts scattered as single missing
  samples, the GPS-only Legendre tracker is at parity with Kalman-CA at 10
  percent missing (1.82 against 1.79 m) and behind at 20 percent (2.07 against
  1.88 m). The in-gap advantage does not carry to a flight's average.
- Through a gap on a sustained turn the turn-aware IMM and CT-EKF coast at
  about half the error of everything else; on the figure-8 Kalman-CA leads and
  the dead-reckoned cubic is 37 percent behind it.
- On the GSDC highway trip Kalman-CA (2.87 m) is ahead of the cubic at every
  window (3.18 m at best).
- Under a synthetic glitch the Huber Kalman leads the reweighted fit by 15 and
  39 percent, and its margin grows with the window (20 percent at window 8, 50
  at 12) on the urban-canyon trip.
- The reweighted route has a window floor of twice the parameter count: at 6
  fixes under a cubic its error is erratic in the clip threshold. On a clean
  trip reweighting buys nothing and costs up to 0.36 m.
- The plain fit is no better than Kalman-CA under glitches at any fraction.

## Reading it

On a position stream the window tracker is a smoother at parity with
Kalman-CA that identifies a model instead of a state. Its measured advantages
are inside a gap (dead reckoning from the window) and under multipath (the
reweighted window); a turn-aware filter wins wherever the turn is the signal.
