# Domain -- The GPS hardware rig

The streaming tracker on real silicon and real fixes: an Arduino Nano 33 BLE
Sense (nRF52840, Cortex-M4F) with a NEO-M8N GPS, logging raw fixes, the 9-axis
IMU and an on-chip float32 estimate to an SD card and over BLE.

**Notebook:**
[rig](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-hardware/experiments/rig.ipynb)
in `packages/dtfit-hardware`. It reruns without the board: the serial captures
of the on-chip runs and the compile sizes are tracked, and the notebook
recomputes the float64 reference and compares. Flashing and capturing is a
written procedure in the notebook. The recorded drives are not part of the
repository; the notebook reports error tables only.

## Routes and baselines

- **dtfit:** the GPS-only Legendre window tracker with the adaptive and the
  fixed window; `LocalTimeFilter`, the same tracker carried in the time of its
  newest fix; the on-chip filter (window 15, Legendre order 5, a two-parameter
  model per axis) in float32.
- **Baselines:** Kalman-CA and a CT-EKF on the same fixes.

No surveyed truth exists for the three logs (a 1 Hz drive, a 5 Hz drive, a
walk): the scores are forecast against the later fix and coast against
held-out fixes.

## What is measured

**Forecast.** At a common configuration CT-EKF and Kalman-CA win the shortest
horizons (two samples on the 5 Hz drive and the walk, three on the 5 Hz drive)
and past that the tracker wins: 5.43 m against 6.18 and 8.62 m for the two
baselines at ten samples on the 5 Hz drive, 19.21 m against 31.18 and 53.47 m
on the walk. With each method's configuration chosen on the other two logs the
tracker leads Kalman-CA on the 5 Hz drive and the walk everywhere but the
walk's two-sample horizon, which is level; against CT-EKF it is 7 percent
behind at two samples, level at three and ahead at ten.

**The adaptive window** wins both driving logs at every horizon and dropout;
on the walk it wins the short horizon and loses the 10- and 15-sample
dropouts to the fixed window (15.14 against 14.74 m, 21.83 against 21.13 m).

**The clock origin.** The absolute-time tracker is not blind to the clock's
zero: shifting a log's time stamps by 10000 s moves its two-sample RMSE from
1.73 m to 8.91 m on the 5 Hz drive. Moving the polynomial's origin onto every
new fix, exact for a polynomial, removes that: with the window held no shift
moves a score. The re-centred tracker takes the 1 Hz drive from 5.85 m to 4.14
m at two samples, level with Kalman-CA and ahead of it from three; it is 1 to
7 percent behind the absolute-clock tracker on the 5 Hz drive and a fifth
behind it on the walk.

**On the chip.** Across the three sketches flash use is 9 to 39 percent and
RAM 17 to 28 percent of the board; the filter's struct is 152 bytes and an
update 104 microseconds. Its float32 output agrees with a float64 replay of
the same algebra to millimetres over a full drive.

## Where dtfit loses

- On the 1 Hz drive CT-EKF and Kalman-CA beat the GPS-only tracker at every
  horizon with a common configuration; chosen per log the tracker's place
  against Kalman-CA starts about five samples out, and against CT-EKF the grid
  gives it none.
- Under dropouts CT-EKF coasts better than either window setting at every gap
  of both driving logs and at the walk's two shorter gaps.
- The metre-scale gap between the chip's output and the library's own filter
  is the damped-step guard and the jump test the chip does not run.

Three recorded drives, one board, one frozen on-chip configuration.
