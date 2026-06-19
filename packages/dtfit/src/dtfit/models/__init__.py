"""dtfit.models -- a framework for picking and constructing the right model.

The domain studies concluded that *choosing the structurally-correct model is the
whole game*. This package makes that choice ergonomic:

- a **catalog** of named families (:func:`logistic`, :func:`gaussian`,
  :func:`damped_oscillation`, ...), each a :class:`Model` that reads its own
  ``p0``/``bounds`` off the data, so you pick structure, not strings::

      from dtfit import models
      fit = models.logistic().fit(x, y)          # self-seeded

- **composition** with ``+`` (e.g. a trend plus a cycle); the second component is
  seeded on the detrended residual of the first::

      fit = (models.linear() + models.sine()).fit(x, y)

- **inference**: :func:`suggest_models` shortlists candidates by data shape, fits
  them, and ranks the families by AIC::

      for s in suggest_models(x, y)[:3]:
          print(s.name, s.r2, s.aic)

Families are grouped by ``category`` (trend / growth / decay / sigmoid /
saturating / peak / oscillatory) -- see :data:`CATALOG`.
"""

from ._model import Model
from ._suggest import suggest_models, Suggestion
from ._catalog import (
    CATALOG,
    all_models,
    # trend
    linear,
    quadratic,
    cubic,
    power_law,
    logarithmic,
    sqrt_law,
    # growth
    exponential,
    exp_growth_offset,
    # decay / relaxation
    exp_decay,
    exp_decay_offset,
    first_order,
    biexponential,
    stretched_exponential,
    # sigmoid
    logistic,
    gompertz,
    weibull_cdf,
    tanh_step,
    # saturating / rational
    michaelis_menten,
    hill,
    # peak
    gaussian,
    lorentzian,
    double_gaussian,
    # oscillatory
    sine,
    damped_oscillation,
    fourier_series,
)

__all__ = [
    "Model",
    "suggest_models",
    "Suggestion",
    "CATALOG",
    "all_models",
    "linear",
    "quadratic",
    "cubic",
    "power_law",
    "logarithmic",
    "sqrt_law",
    "exponential",
    "exp_growth_offset",
    "exp_decay",
    "exp_decay_offset",
    "first_order",
    "biexponential",
    "stretched_exponential",
    "logistic",
    "gompertz",
    "weibull_cdf",
    "tanh_step",
    "michaelis_menten",
    "hill",
    "gaussian",
    "lorentzian",
    "double_gaussian",
    "sine",
    "damped_oscillation",
    "fourier_series",
]
