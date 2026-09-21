# FilterBank & FusedChiSquareDetector -- experimental tooling

`FilterBank` (many streaming filters driven in lockstep) and
`FusedChiSquareDetector` (a pooled multi-stream fault test built on them)
live in `dtfit_experimental.streaming` -- experiment harness tooling, not
part of the stable `dtfit` surface. See
[the experimental adaptations API](Experimental-Adaptations-API).

The stable equivalent of the fused test is summing several
[`ImageFilter`](Methods-Legendre-Filter) instances' `nis_`: each is
chi-square under its model, so the sum is chi-square with the summed
degrees of freedom, giving the pooled test more degrees of freedom and
power than any one filter's innovation alone when the change is spread
evenly over the streams; on the three-axis damping fault of experiment
21, carried mostly by one axis, per-axis tests on every sample were as
fast. See
[several streams](API-Streaming#several-streams).

On a flag the detector re-arms every filter: `inflate=` multiplies each
`P` by the given factor and keeps the windows, `rearm=True` calls each
filter's `rearm()` instead, which applies the filter's own `drift_reset`
and collapses its adaptive window. A bank of one filter is the per-sample
innovation test on a single stream.
