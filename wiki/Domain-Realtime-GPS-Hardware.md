# Domain -- Real-time GPS / inertial trajectory (on-silicon hardware rig)

The **on-silicon twin** of the simulated [`realtime_gps`](Domain-Realtime-GPS)
domain. Everything device-specific lives in the third package,
[`dtfit-hardware`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-hardware/README.md):
an Arduino Nano 33 BLE Sense + u-blox NEO-M8N GPS running the on-MCU streaming LSI
filter, a USB/BLE host telemetry link, a real-log comparison harness, and a
phone-side BLE monitor app. It runs the same E1/E2/E3/E5 story the simulation
tells, but on **real hardware and real logged drives** -- the engineering
counterpart of the planned embedded paper.

> The living report is the notebook
> [`realtime_gps_hw.ipynb`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-hardware/src/dtfit_hardware/realtime_gps_hw.ipynb)
> (50 cells, 12 figures). This page is the narrative summary; the notebook has the
> figures and tables.

## The rig

| Piece | What |
|---|---|
| Board | Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4F @ 64 MHz) + onboard BMI270 IMU / BMM150 mag |
| GPS | u-blox NEO-M8N (single-frequency), UART, configured to 5 Hz RMC+GGA over UBX at boot |
| Firmware | `nano_lsi_log` -- polls the IMU at high rate, integrates a gravity-aligned yaw, runs the on-MCU float32 streaming LSI, and logs GPS+IMU+estimate to SD and BLE |
| Host link | `backend.py` -- find / flash / capture / log over USB (`pyserial`) and BLE (`bleak`), wrapping the bundled Arduino CLI |
| Harness | `compare_real.py` -- scores dtfit's integral trackers against Kalman-CA / CT-EKF on captured logs |
| Phone app | `mobile/dtfit-monitor` -- a React Native app showing GPS fix / sats / speed / IMU / on-MCU LSI live over the `dtfit-gps` GATT service; recording is a phone toggle over a BLE control characteristic |

Build: [BOM + wiring](https://github.com/ringavirda/science-nonline/blob/main/papers/embedded_hardware_bom.md)
· [beginner build guide](https://github.com/ringavirda/science-nonline/blob/main/papers/embedded_nano_build_guide.md).

## What is measured (and why)

A real single-frequency drive has **no external ground truth**, so the harness
scores the two metrics that are well-defined from the data itself (the no-RTK
plan), all in local-ENU metres about the first fix (small values keep the
per-window fit conditioned, the same reason the firmware runs in ENU):

- **Forecast RMSE (E1)** -- at each step the tracker predicts the fix `h` steps
  ahead; truth is the actually-observed future fix.
- **Dropout-coasting RMSE (E2)** -- blank synthetic gaps, let each tracker coast,
  and score the coasted estimate against the *held-out* real fixes (GPS+IMU
  methods dead-reckon; GPS-only ones extrapolate their local model).
- **Float32-vs-float64 (E5)** -- when the log carries the on-MCU `est_lat`/`est_lon`,
  the logged float32 estimate vs a float64 dtfit replay on the same raw fixes. The
  rigorous bit-faithful check is the golden vector in `tools/embed_lsi.py`
  (`cross_check()` = **4.4e-16**: the on-MCU filter is demonstrably the dtfit
  method, not a lookalike).

Public benchmarks with real truth (**comma2k19** highway, **UrbanNav** deep-urban)
add the E1/E2/E3 comparison against absolute ground truth.

## Results

**The final 5 Hz car drive (7.3 km, up to 116 km/h).** Raising the GPS to 5 Hz
shrinks the fixed 15-sample LSI window from 15 s to ~3 s -- the *real* version of
the shorter-window win:

- **On-MCU estimate ~4x tighter at speed** (136 -> 33 m @ 100 km/h).
- **dtfit LSI-cubic (with coasting) now BEATS Kalman-CA** at 5 Hz: forecast 6.1 vs
  8.6 m, coast 4.5 vs 5.9 m. This is a **regime flip** -- at 1 Hz (an earlier
  drive) LSI *trailed* Kalman on the fast car run because its velocity is the
  edge-derivative of a noisy per-window polynomial refit, a worse prior than
  Kalman's process-noise-smoothed constant-acceleration state for smooth
  high-speed motion. 5 Hz oversamples the <1 Hz car dynamics enough to close it.
- **IMU fusion is real but situational.** The IMU rides on jumper wires and wobbles
  (mount vibration ~15x the parked noise floor, isotropic), so accelerometer
  strapdown is dead -- but a complementary filter (real gyro slow-anchored to the
  GPS course) cleans the heading **82deg -> 9deg**, and the cleaned gyro then
  carries position through GPS dropouts that span a **turn** (-13% vs GPS-only on
  turn-gaps; a wash on straights, so random gap placement averages to ~0).

**comma2k19 (public highway benchmark, absolute truth).** On 22 constant-accel
highway segments, **dtfit beats Kalman against absolute ground truth**: forecast
2.3 vs 2.9 m @ 2 s, coast 5.3 vs 7.4 m @ 5 s gap (the raw u-blox is already ~0.5 m
median). Organic glitch rate here is **0** -- open-sky highway is too clean to
exercise the robustness path, which is why an urban set was needed.

**UrbanNav (deep-urban canyons, SPAN-CPT truth) -- the honest E3 finding.** Organic
urban multipath is **temporally correlated / sustained, not isolated spikes**
(error humps span tens of seconds as the car passes buildings). A *pointwise*
robust filter (winsorized dtfit) therefore does **not** rescue it -- robust ~= raw
~= Kalman at the NLOS epochs (~110 m on a single-freq M8T-class receiver). HDOP
barely predicts the error (multipath != geometry), so gating fails too. So the
integral-filter **robustness win is scoped to isolated outliers** (on the rig,
injected-glitch 2.4 vs 26 m); sustained urban NLOS is a tight-coupling / 3D-map /
receiver-quality regime. Side finding: **the receiver dominates in a canyon** --
same drive, dual-frequency F9P 2.4 m median vs single-freq M8T 29 m / 280 m max.

## On-MCU cost & footprint

Measured on the M4F via the `nano_lsi_onboard` demo (which replays the on-boot
golden vector and prints its own cost): **~267 us/update, sub-kilobyte state**,
float32 bit-faithful to the float64 reference. The information/Woodbury form keeps
the only matrix inverses at `N x N` (N=2 params) rather than the `M x M` (M=6
coefficients) innovation covariance -- a real embedded win.

> **Energy-per-estimate was cut, honestly.** Both INA226 current-meter modules were
> dead (never ACK on an I2C bus proven good by the GPS magnetometer), so the
> on-MCU cost story rests on **us/update + RAM**, not joules.

## Reading it

- **Deployable, and confirmed on silicon.** The embedded footprint numbers are no
  longer desktop projections -- the port exists and self-reports its cost.
- **Where the integral filter helps on real data:** low-dynamics (static /
  pedestrian) tracking, isolated-glitch robustness, long-gap coasting through
  maneuvers, and -- once the GPS oversamples the dynamics (5 Hz) -- forecast and
  coasting at speed too.
- **Where a recursive coupled-state filter keeps the edge:** smooth high-speed
  motion at low GPS rate, and multi-step forecasting.
- **Honest limits:** a wobbly jumper-wire mount kills accelerometer strapdown; a
  single-frequency receiver dominates the error budget in a canyon; sustained
  urban NLOS is out of scope for a pointwise robust filter.

See the [simulated `realtime_gps` domain](Domain-Realtime-GPS) for the controlled
E1-E7 study this reproduces on silicon, and the
[embedded-control domain](Domain-Embedded-Control) for the streaming-filter
footprint story.
