# Domain -- Forecasting

Extrapolate a series past its record by fitting a declared structure to the
training part, and find out how much knowing the structure is worth.

**Notebook:**
[32_forecasting](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/32_forecasting.ipynb).

## Routes and baselines

- **dtfit:** a declared model per series fitted from the image; the blind
  `auto_forecast` search over its five classes (logistic, linear,
  linear-seasonal, polynomial, random walk).
- **Baselines:** random walk, seasonal naive, drift, polynomial extrapolation,
  Holt-Winters ETS, Theta, (S)ARIMA, an MLP; for the long-horizon protocol the
  published DLinear, TimesNet and Time-LLM numbers.

## What is measured

Twelve series: eight measured (COVID-19 Ukraine, USD/UAH, sunspots, Mauna Loa
CO2, El Nino SST, Nile flow, ETTh1 oil temperature, a weather channel) and four
generated from a known equation (an RLC transient, AC with harmonics, an AM
signal, a linear chirp). The score is nRMSE on a held-out tail.

- **A governing equation is worth an order of magnitude.** On the four
  generated waveforms the declared structure beats the blind search by 14.0,
  5.9, 22.9 and 25.6 times.
- **Without one it is worth close to nothing.** On the eight measured series
  the declared structure wins four, ties two to six figures and loses two (El
  Nino SST 13.15 against 12.68, sunspots 35.10 against 31.60).
- **The negative control holds.** On all three pairs tried the wrong structure
  is worse in RMSE and flagged by its in-sample R2 (chirp 0.9983 against
  0.0442).
- **Traffic, twelve hourly sensors.** The image fit, the same fit polished by a
  pointwise solve and a plain pointwise fit land within 0.05 percent of each
  other (9.681, 9.687 and 9.687 percent median nRMSE): parity on a real record.

## Where dtfit loses

- Random-walk persistence beats the declared structure on USD/UAH, a crash and
  partial recovery no smooth trend-plus-cycle form captures, and beats the
  blind route on four series (USD/UAH 205.4 against 50.9, the RLC transient,
  ETTh1, weather).
- Against the published long-horizon benchmark a trend-plus-seasonal
  heuristic is not competitive: at H=96 its MSE is 4.8, 5.0 and 3.2 times the
  best published number on ETTh1, ETTm1 and weather, and worse than repeating
  the last value. That heuristic is written inline in the notebook and calls
  nothing from dtfit; the section bounds what a structural extrapolation can
  do on those datasets, not the library.
- nRMSE normalised by the holdout's own range makes a shorter holdout read as
  harder on seven of the twelve series; the notebook reports it as a property
  of the metric.

## Reading it

dtfit forecasts by identifying parameters of a structure the user states. Where
the structure is physics the forecast is an order of magnitude better than a
blind search; where it is a guess, dtfit sits among the established forecasters
and the random walk, ahead on some series and behind on others.
