# Streaming

`ImageFilter` tracks parameters online on the sliding window's image, one
sample at a time (`partial_fit`), at bounded per-update cost. `ImageFilter(basis="legendre")`
and `ImageFilter(basis="block")` fix its basis to Legendre and block. Start from the
`.tracking()` / `.robust()` presets. `result()` reads the current window
off as a calibrated batch fit; `filter.coast(...)` dead-reckons through
measurement dropouts.

```python
from dtfit import ImageFilter

flt = ImageFilter(basis="legendre").tracking("a*exp(b*x)", "x")
for xi, yi in zip(x, y):
    flt.partial_fit(xi, yi)
print(flt.params_)      # latest estimate
```

::: dtfit.ImageFilter

::: dtfit.ImageFilter(basis="legendre")

::: dtfit.ImageFilter(basis="block")

::: dtfit.streaming.DriftDetector
