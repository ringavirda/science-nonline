# ImageFilter(basis="block") -- the block-basis alias

> Numeric **online** method. Source:
> [`streaming/filter.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/streaming/filter.py).
> Invoke via `ImageFilter(expr, var, p0=, window_size=, ..., basis="block")` then
> `flt.partial_fit(t, y)` per sample; `flt.predict(x)`, `flt.params_`.

`ImageFilter(basis="block")` is [`ImageFilter`](Methods-Legendre-Filter) with `basis` fixed
to `"block"` -- the streaming twin of batch [EAC](Methods-EAC). Its window
measurement is the block image: window sums against a diagonal Gram, the
cheapest per-sample statistic of the two bases and the one an embedded
target runs (see [the embedded tool](Domain-Embedded-Control)). Its
sibling [`ImageFilter(basis="legendre")`](Methods-Legendre-Filter) fixes the Legendre basis
instead, whose spectral measurement resolves an oscillatory plant's
shape and frequency directly.

Everything else -- the whitened window-image measurement, the
information-form update, drift detection through the shared
[`DriftDetector`](API-Streaming#driftdetector), the adaptive window,
robust winsorization, `result()`'s calibrated covariance and coasting --
is `ImageFilter`'s and is described once, on
[its method page](Methods-Legendre-Filter).

## Where it is best applied

Use `ImageFilter(basis="block")` for monotone or saturating plants where the block image's
low per-sample cost matters, or as the streaming path an embedded target
runs. For an oscillatory plant use [`ImageFilter(basis="legendre")`](Methods-Legendre-Filter).
For an accurate *static* batch fit use [LSI](Methods-LSI) or
[EAC](Methods-EAC).
