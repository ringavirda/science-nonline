# experiments — real-data validation

Validation of the `dtfit` methods against **real, downloaded** time series
(no synthetic signals). The datasets are chosen to match the dissertation's
stated domain — nonlinear smoothing/forecasting on **currency / economic /
pandemic** series — and the methods' target shape: models that are nonlinear
in their parameters (exponential / transcendental).

## Run

```bash
python -m dtfit_experimental.experiments.download_data      # fetch datasets into experiments/data/
python -m dtfit_experimental.experiments.validate_methods   # fit + report metrics
python -m dtfit_experimental.experiments.benchmark          # method docs: figures + comparison tables
```

`benchmark.py` additionally writes the per-method figures into the repo's
`wiki/figures/` and prints the comparison tables shown on the
[method docs](https://github.com/ringavirda/science-nonline/wiki/Methods) wiki
pages. It needs the optional `bench` extra (matplotlib) on top of the core deps.

Only `numpy`, `scipy`, `scikit-learn` are needed (already `dtfit` deps);
downloads use the stdlib. No API keys.

## Datasets

| file | source | what / why |
|------|--------|------------|
| `data/usd_uah_2014_2015.csv` | [National Bank of Ukraine open API](https://bank.gov.ua/) (`NBU_Exchange/exchange_site`) | USD/UAH official daily rate, 2014-2015 hryvnia crisis (7.99 → 24.0). Sustained, roughly exponential depreciation through two devaluation regimes. Currency domain. |
| `data/covid_ukraine_confirmed.csv` | [JHU CSSE COVID-19 time series](https://github.com/CSSEGISandData/COVID-19) | Cumulative confirmed cases, Ukraine. The early-2020 take-off is the textbook exponential-in-parameters signal. Pandemic domain. |

`rate_per_unit` is used for the FX series (NBU historically quoted USD per 100
units; `rate_per_unit` is the per-1-USD rate).

## Experiments & findings

**1. COVID-19 exponential growth — `y = a·exp(b·t)`, batch fit + forecast.**
A 4-week take-off window (~16× range, where a pure exponential is
well-conditioned). LSI and EAC recover the growth on par with a SciPy
`curve_fit` baseline (in-sample **R² ≈ 0.97 / 0.98**, MAPE ≈ 5%). Forecasting
the held-out last 20% gives **negative R²** for *all three* methods (including
SciPy) — this is the genuine, expected hardness of exponential extrapolation
(error compounds), not a method defect.

**1b. NonlineRegressor (scikit-learn wrapper)** on the same window: `fit` /
`score` / `cross_val_score` all work end-to-end on real data (4-fold CV
R² ≈ 0.78), confirming the sklearn integration.

**2a. USD/UAH exponential depreciation trend — batch LSI/EAC.** A single
exponential through the whole 2-year crisis captures the depreciation trend at
**MAPE ≈ 12%** (R² ≈ 0.6); the residual is the two discrete devaluation jumps a
single smooth exponential cannot represent. Honest, interpretable.

**2b. EACFilter streaming — online tracking, bounded cost per sample.**
The recursive filter tracks the 8 → 30 depreciation online at lag-1
**MAPE ≈ 9%** with O(window·params) cost per update. One-step-ahead it does
**not** beat the naive random walk (RW MAPE ≈ 0.7%) — the well-known result
that daily FX is near-RW; the filter's value is bounded-latency adaptive
tracking, not beating RW one step out. Its **two-sided drift detector**
(NIS for sudden jumps + a two-sided CUSUM for sustained shifts, both on a
self-standardized innovation) flags the **Feb-2015 free-float** (≈16 → 28) as
a structural break. Detection of shifts in *both* directions (up and down) is
covered by the unit tests on controlled step signals (`tests/test_streaming.py`).

## Notes

- LSI/EAC need a **modest dynamic range** (their empirical spectrum is a
  Maclaurin fit around 0); an extreme range is ill-conditioned. The COVID
  window is sized accordingly.
- Domains are normalized to `[0, 1.5]` and series scaled to O(1) before
  fitting; these are invertible rescalings and do not affect R².
