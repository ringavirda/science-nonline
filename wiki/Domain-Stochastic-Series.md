# Domain -- Stochastic series

Characterise a series that has no deterministic model: read its regime (random
walk, mean reversion, long memory, trend, cycle, volatility clustering),
estimate the parameters of that regime, forecast accordingly, and do it from a
fixed-size statistic that merges.

**Notebooks:**
[23_stochastic_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/23_stochastic_image.ipynb)
(synthetic processes),
[36_stochastic_real](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/36_stochastic_real.ipynb)
(seven real records). Method and API: [Methods-Stochastic](Methods-Stochastic),
[API-Stochastic](API-Stochastic).

## Routes and baselines

- **dtfit:** the second-order image (lagged sums, dyadic block sums, a
  fixed-grid DFT, trend sums), the estimators read from it, the regime router,
  the merged forecaster, the streaming filter, `simulate`.
- **Baselines:** a classical twin of the whole tier -- ADF, OLS AR(1), the
  periodogram, R/S and DFA, a GARCH quasi-MLE -- and for the forecasts the
  random walk, ETS, AR(1) and ARIMA.

## What is measured

- **The image merges exactly** once the scale budget is pinned: a record
  imaged whole and imaged in blocks then merged agree to floating-point noise.
- **Estimators by route.** Mean reversion and volatility persistence are at
  parity with OLS AR(1) and the GARCH quasi-MLE. The cycle period is decided by
  whether the true period lands on a periodogram bin: on one the classical
  route is exact, off it dtfit wins.
- **The regime router** ties its classical twin at 96.4 percent over 56 cases,
  with one miss in common (an AR(2) cycle read as mean-reverting). A
  finite-order AR process never routes as long memory, including the
  persistent AR(1) at 0.9 whose pre-veto Hurst statistic clears the threshold
  in every draw.
- **The streaming filter** flags a persistence jump and a volatility switch at
  flat per-sample cost with its state under a fixed cap; the generator
  round-trips every regime.
- **Seven real records.** Every detected regime matches the literature's
  reading: GDP, two FX levels and the T-bill as random walks, CO2 as trend plus
  a seasonal cycle, sunspots as a cycle in the 8 to 14 year band, the Nile as
  trend. The held-out forecast never trails the random walk (worst ratio
  1.0000) and beats it by more than half on CO2 and GDP. USD/UAH is a
  random-walk level with long memory in its absolute returns that four
  estimators agree on.

## Where dtfit loses

- The two Hurst read-outs trail R/S and DFA.
- The forecast loses on the AR(2) cycle and on GARCH(1,1), where both routes
  read the same regime and the textbook forecaster that regime selects beats
  the one the merged model falls back to; on mean reversion it ties.
- ETS, AR(1) and ARIMA each win a real series outright.
- The cycle gate's significance test assumes a white-noise null and opens far
  more often on red noise; its effect-size floor and two other constants are
  hand-set. On the Nile the 19.7-year cycle is the 1899 level shift read
  through a line, and the significance test is what rejects it; the router has
  no test that a level shift beats a slope, so the Nile's trend is still
  reported.
- The Nile's Hurst read-out is not stable across the record: adding the 12
  held-out years moves DFA from 0.88 to 0.61.

## Reading it

Near a random walk there is nothing to win, and the measure is whether the
router declines to invent structure: on the real records it never trails
persistence. The image form adds what the classical estimators do not have --
one fixed-size statistic that merges across blocks and machines and feeds
every read-out.
