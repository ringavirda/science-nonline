# experiments — the dtfit validation suite

Evidence that the `dtfit` EAC/LSI methods and their adaptations work. The suite
has **three layers**; the notebooks are the canonical study, the root scripts are
auxiliary tools.

## Layout

```
experiments/
  cases/        # per-ADAPTATION studies — one optimization / structural idea per
                # folder (01_control_systems … 10_fused_partitioned_batched),
                # scored on the promotion matrix. See cases/REPORTS.md and the
                # per-feature deep-dives in cases/analysis/.
  domains/      # per-APPLICATION-DOMAIN studies — the validated levers merged and
                # run against the baselines a practitioner in that domain actually
                # uses (big_data, embedded_control, forecasting, parameter_estimation,
                # realtime_gps, stochastic_series). See domains/DOMAINS.md.
  common/       # the shared pure-compute library both trees import (metrics,
                # baselines, datasets, plotting used by notebooks).
  data/         # bundled real-data CSVs (populated by download_data.py).
  *.py          # the root CLI tools below.
```

`cases/` and `domains/` are a **deliberate two-axis design**, not a duplication:
`cases/` isolates one adaptation at a time to feed the promotion decision;
`domains/` merges the promoted levers and measures them end-to-end per domain.
Each experiment folder holds a `backend.py` (pure compute — the single source of
truth for its simulation/estimation/data infra) and a Jupyter notebook that
imports it and produces the report (tables, figures, narrative). Run a notebook
directly, or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb
```

## Root CLI tools

```bash
python -m dtfit_experimental.experiments.download_data   # fetch datasets into data/
python -m dtfit_experimental.experiments.benchmark       # method docs: wiki/figures + comparison tables
python -m dtfit_experimental.experiments.accuracy_explore # recovery-accuracy sweep over the test SCENARIOS
python -m dtfit_experimental.experiments.validate_methods # quick real-data smoke test (COVID/FX)
python -m dtfit_experimental.experiments.streaming_lsi_benchmark  # LSIFilter vs EACFilter micro-benchmark
```

- **`benchmark.py`** renders the per-method figures into the repo's `wiki/figures/`
  and prints the comparison tables shown on the [Methods](https://github.com/ringavirda/science-nonline/wiki/Methods)
  wiki pages (needs the optional `bench` extra: matplotlib).
- **`validate_methods.py`** is a lightweight smoke check on two real series
  (COVID-19 growth, USD/UAH depreciation); the rigorous forecasting/parameter
  studies live in `domains/forecasting` and `domains/parameter_estimation`.
- **`accuracy_explore.py`** drives the recovery-accuracy corpus (the Phase-0
  measurement feeding the promotion-gate thresholds).

Only `numpy`, `scipy`, `scikit-learn` are needed for the core scripts (already
`dtfit` deps); downloads use the stdlib, no API keys. Plotting/notebooks need the
`bench` extra.

## Datasets (used by `validate_methods` / `download_data`)

| file | source | what / why |
|------|--------|------------|
| `data/usd_uah_2014_2015.csv` | [National Bank of Ukraine open API](https://bank.gov.ua/) (`NBU_Exchange/exchange_site`) | USD/UAH official daily rate, 2014-2015 hryvnia crisis (7.99 → 24.0). Sustained, roughly exponential depreciation through two devaluation regimes. Currency domain. |
| `data/covid_ukraine_confirmed.csv` | [JHU CSSE COVID-19 time series](https://github.com/CSSEGISandData/COVID-19) | Cumulative confirmed cases, Ukraine. The early-2020 take-off is the textbook exponential-in-parameters signal. Pandemic domain. |

`rate_per_unit` is used for the FX series (NBU historically quoted USD per 100
units; `rate_per_unit` is the per-1-USD rate).

## Notes

- LSI/EAC need a **modest dynamic range** (their empirical spectrum is a Maclaurin
  fit around 0); an extreme range is ill-conditioned, so windows are sized
  accordingly and domains are normalized to `[0, 1.5]` / series scaled to O(1)
  before fitting (invertible rescalings that do not affect R²).
- The study tier is exempt from the mypy gate and ruff-relaxed; it is run
  notebook-by-notebook, not imported as a library. Nothing in the importable
  `dtfit_experimental` tier imports from here.
