# dtfit

**D**ifferential-**t**ransformation **fit**ting: nonlinear smoothing and
forecasting on time-series / Big Data, using methods built in the scheme of
differential (non-Taylor) transformations. Developed as part of a PhD
dissertation on mathematical models of nonlinear smoothing and prediction.

## Installation

```bash
pip install -e .            # core
pip install -e '.[viz]'     # + matplotlib plotting helpers
pip install -e '.[dev]'     # + test/lint tooling
```

The reference development environment is a conda env (`science-nonline`); install
into it with the commands above.

## Quick start

```python
import numpy as np
import dtfit as dt

x = np.linspace(0, 10, 400)
y = ...  # your observations

# Batch fit, numeric (no polynomial pre-fit needed):
result = dt.nonline_fit("a*atan(w*x)", "x", method="eda", data_x=x, data_y=y)
print(result.coeffs)

# ...or through the pipeline:
model = dt.Model()
model.use(dt.FittingNonlineMw("a0 + a1*x + a2*exp(a3*x)", "x", method="lsi"))
model.fit(x, y)
y_hat = model.predict(x)
```

### Real-time / streaming

```python
flt = dt.EqualAreasFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50)
for t, y in stream:           # bounded-cost per-sample update
    flt.partial_fit(t, y)
print(flt.params_)            # tracks time-varying parameters
```

## Methods

- **LSI** (`method="lsi"`) — least-squares integral; numeric integral-OLS in the
  differential-transformation scheme (successor to DSBI).
- **EDA** (`method="eda"`) — equal differential areas / equal areas; numeric,
  integration-based and noise-robust (successor to DSBE).
- **EqualAreasFilter** — recursive/online EDA with NIS drift detection for
  real-time tracking.
- **DSB** (`method="dsb"`) — symbolic differential spectra balance; kept as the
  analytical reference (requires a polynomial fit first in the pipeline).

Each method's mathematical grounding (in differential / non-Taylor
transformations), full algorithm, optimizations, guards, applicability, usage
figures and comparison tables are documented under
[docs/methods/](docs/methods/README.md).

Core dependencies: numpy, scipy, sympy, scikit-learn. Plotting helpers require
the optional `viz` extra (matplotlib).
