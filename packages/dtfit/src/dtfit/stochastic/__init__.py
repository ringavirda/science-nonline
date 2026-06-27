"""Stochastic-series characterization, forecasting, generation and tracking.

dtfit fits a deterministic ``y = f(t; theta)``; a genuinely random series has no
such ``f``. The way to *use* dtfit on it is to fit the **deterministic
functionals** of the process -- its autocovariance, spectrum, aggregated variance
and trend/cycle -- whose forms (damped exponentials/cosines, power laws) are
exactly the shapes dtfit excels at, and read the process's parameters out of them.

Layered API:

* **estimators** -- :func:`hurst_aggvar` / :func:`hurst_spectral` (long memory),
  :func:`ar1_reversion` (mean reversion), :func:`garch_persistence` (volatility),
  :func:`cycle_period` (stochastic cycle), :func:`decompose_trend_cycle`; each
  recovers a stochastic-model parameter by feeding a functional to
  ``fit_lsi`` / ``fit_eac``;
* **batch** -- :func:`fit_stochastic` composes the routes behind significance
  gates into a single :class:`StochasticModel` that detects the regime, forecasts
  by backtest model selection, and *generates* fresh realizations
  (:meth:`StochasticModel.simulate`);
* **streaming** -- :class:`StochasticFilter`, the per-input online twin of the
  second-order stage (EWMA autocovariances read by the EAC equal-areas criterion),
  with a fused change-point detector.

The model-framework wrapper :class:`dtfit.Stochastic` exposes the batch solution
through the ``.fit(x, y)`` / ``.predict`` convention of :class:`dtfit.Model`.
"""

from dtfit.stochastic._model import (
    sample_acf,
    hurst_aggvar,
    hurst_spectral,
    ar1_reversion,
    garch_persistence,
    cycle_period,
    decompose_trend_cycle,
    fit_stochastic,
    StochasticModel,
    FORECASTERS,
)
from dtfit.stochastic._filter import StochasticFilter

__all__ = [
    "sample_acf",
    "hurst_aggvar",
    "hurst_spectral",
    "ar1_reversion",
    "garch_persistence",
    "cycle_period",
    "decompose_trend_cycle",
    "fit_stochastic",
    "StochasticModel",
    "FORECASTERS",
    "StochasticFilter",
]
