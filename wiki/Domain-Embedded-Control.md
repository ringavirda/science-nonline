# Domain -- Embedded real-time control

Identify and track the parameters of a plant online, sample by sample, with a
bounded cost per step and a fixed state, through noise, anomalies and missing
samples, and flag a change in a parameter when it happens.

**Notebooks:**
[21_filter_form](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/21_filter_form.ipynb),
[24_embedded_footprint](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/24_embedded_footprint.ipynb).
The chip itself is measured on the [hardware rig](Domain-Realtime-GPS-Hardware).

## Routes and baselines

- **dtfit:** `ImageFilter` on the window image in the Legendre and the block
  basis, with the adaptive and the fixed window, the clipped (reweighted)
  window, a chi-square test of the window innovation on every sample, and
  `dtfit_experimental.streaming.FilterBank` for several axes.
- **Baselines:** an extended Kalman filter with the parameters as its state,
  scored at a static setting, at a tracking setting and over a grid of twelve
  settings, with and without a 3-sigma gate; recursive least squares; a
  constant-acceleration Kalman filter; a sliding-window NLLS refit; the batch
  NLLS fit of the whole record as the floor.

## Plants

| plant | application | model | parameters |
|---|---|---|---|
| damped_osc | control, vibration | `A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)` | 3 |
| ac_sine | power monitoring | `c + A*sin(w*t)` | 3 |
| first_order | RC, thermal, DC motor | `K*(1-exp(-t/tau))` | 2 |
| ca_traj | trajectory | `c0 + c1*t + c2*t**2` | 3 |

Eight seeds, 5 percent noise unless a sweep says otherwise; the score is the
mean relative parameter error over the seeds, in percent.

## Accuracy: the process noise decides, not the basis

| | damped_osc | ac_sine | first_order | ca_traj |
|---|---|---|---|---|
| window filter, tuning matched to static parameters | 0.60 | 0.38 | 0.55 | 3.16 |
| EKF at a static setting | 0.58 | 0.25 | 1.30 | 2.26 |
| filter / best of twelve EKF settings | 1.03 | 1.55 | 1.80 | 1.46 |
| image of the whole record (equals batch NLLS) | 0.21 | 0.25 | 0.16 | 2.27 |

Matched to a plant whose parameters do not move, the window filter is at the
accuracy of an EKF given the same assumption: parity on the damped oscillator,
ahead on the first-order rise, behind on the sinusoid and on the trajectory,
where the static EKF is recursive least squares and sits on the batch floor.
The filter runs one setting and is given no measurement variance; it stays
within a factor of two of the best of twelve EKF settings on every plant, and
no single EKF setting is within 1.5 times that best on all four. The fit from
the image of the whole record equals batch NLLS and is 2.8 to 3.5 times better
than the best online estimator on the two plants with a transient: for
parameters that do not move, the running image is the estimator to use, and
the window filter is for parameters that may move.

With the window floor held equal the two bases enter the 10 percent band 1.5 to
1.9 steps after the first measurement: convergence speed is the window floor,
not the basis.

## Anomalies and missing samples

Robustness is the reweighting of the window. At 10 percent anomalies the
clipped filter keeps all 8 runs under 5 percent error; a 3-sigma-gated EKF
keeps 2 or 3 of 8, because a run the gate loses is locked out for good; an
ungated EKF goes to hundreds of percent. A dropout is a thinner window, not a
missed time update: through 20 percent of samples lost the matched filter
reads 0.60, 0.53 and 0.65 percent against the static EKF's 0.64, 0.56 and 0.66.

## A jump in a parameter

| | flagged of 96 | latency | false alarms |
|---|---|---|---|
| chi-square test of the window innovation, every sample | 96 | within four samples | 0 |
| the filter's own detector, fixed window | 76 | 0.8 to 5.8 s by the phase of the jump | -- |
| the filter's own detector, adaptive window | 53 | -- | -- |

The filter's own detector is consulted once per window, so its latency is
where the jump lands in that stride, and under the adaptive window it misses
the jumps the window has already absorbed. What the adaptive window buys is
the return to the new value: 2.4 s against the fixed window's 5.3 s, and
clearing the window on the flag (`ImageFilter.rearm()`) brings either to 0.8
s. On a fault that hits three axes at once the test rate is the whole gap: a
test on every sample flags in 1 step, the once-a-window test in 19, and a
Kalman bank that carries no model of the signal never flags. Per-axis tests on
every sample match the pooled test at every fault size tried.

## Cost and state

| | per step | note |
|---|---|---|
| window filter, either basis | 107 to 133 us | worst step 0.2 to 0.3 ms |
| EKF | 13.1 us | |
| sliding-window refit | 29 to 121 us in the mean | worst step 0.9 to 15 ms; needs an optimizer on the device |
| Kalman-CA, one axis | 8.4 us | tracks a position, identifies nothing |

The state is a fixed struct that does not grow with the stream: 212 bytes in
float32 for a quadratic axis at a window of 15, 492 for the damped oscillator
at 50, 636 for a three-axis tracker -- 31 percent of an ATmega328's 2 KB and
under 2 percent of a Cortex-M0+, a Cortex-M4F, an ESP32 or an nRF52840. The
struct model reads 184 bytes for the rig's configuration against 152 measured
on the chip, the 32-byte difference named field by field. A FLOP-count estimate
of the update is 3.5 times low against the timed nRF52840 (29.8 against 104.3
us); carried to every part, that error leaves the 32-bit parts inside a 10 Hz
epoch and puts the 8-bit part at 110.7 ms, outside it. No 8-bit chip was
timed.

## Where dtfit loses

- To the static EKF on the sinusoid and the trajectory, and from 10 percent
  noise up on the damped oscillator.
- On cost: an EKF step is about ten times cheaper, a Kalman-CA step cheaper
  still.
- A tuning inherited from a tracking configuration loses to the EKF everywhere
  it is scored, by eight to nine times on the damped oscillator: the matched
  tuning is the right one only where the parameters are static.
- The filter's own once-a-window detector, against a test on every sample.

Every plant is synthetic, the noise white and the anomaly one spike shape;
the microsecond figures are one shared desktop's and are read as ratios.
