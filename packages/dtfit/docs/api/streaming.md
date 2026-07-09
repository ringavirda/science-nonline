# Streaming

Online (`partial_fit`) parameter trackers -- the streaming twins of the batch
fitters. Feed samples one at a time; the filter maintains a running parameter
estimate (and covariance) that adapts as the parameters drift. Start from the
`.tracking()` / `.robust()` presets. `FilterBank` drives many independent streams
at once; `filter.coast(...)` dead-reckons through measurement dropouts.

```python
from dtfit import LSIFilter

flt = LSIFilter.tracking("a*exp(b*x)", "x", param_names=("a", "b"))
for xi, yi in stream:
    flt.partial_fit(xi, yi)
    print(flt.params_)      # latest estimate
```

::: dtfit.EACFilter

::: dtfit.LSIFilter

::: dtfit.FilterBank
