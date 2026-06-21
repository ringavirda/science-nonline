# science-nonline — dtfit monorepo

Differential-transformation fitting: nonlinear smoothing and forecasting via
differential / non-Taylor transformations. Developed as part of a PhD
dissertation on mathematical models of nonlinear smoothing and prediction.

This repository is a **monorepo of two distributions**:

| package | path | what it is |
|---|---|---|
| **`dtfit`** | [`packages/dtfit`](packages/dtfit) | the stable, published library — the public API (`NonlineRegressor`, `EDAFilter`, `fit_lsi`/`fit_eda`/`fit_dsb`, `PartitionedLSI`/`PartitionedEDA`, streaming filters, the `models` framework + `suggest_models`, `auto_estimate`/`auto_forecast`, diagnostics). |
| **`dtfit-experimental`** | [`packages/dtfit-experimental`](packages/dtfit-experimental) | experimental EDA/LSI adaptations + the full experiment / validation suite. Depends on `dtfit`; **never ships inside the `dtfit` wheel**. |

The dependency is one-directional: `dtfit-experimental` → `dtfit`. When an
experimental adaptation proves itself across the experiment suite it is
**promoted** — physically moved into `dtfit` (as `PartitionedLSI` / `PartitionedEDA`
were) and re-exported from there, never imported back out of the experimental
package.

## Install (editable, from the repo root)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate   |   Linux/mac: source .venv/bin/activate

pip install -e packages/dtfit                          # stable library
pip install -e 'packages/dtfit[dev]'                   # + test/lint tooling

# optional — the experimental adaptations + experiment suite:
pip install -e packages/dtfit-experimental
pip install -e 'packages/dtfit-experimental[bench]'    # + suite baselines/plotting
```

## Layout

```
packages/
  dtfit/                     # stable library (pyproject, src/dtfit, tests, build_native.py)
  dtfit-experimental/        # experimental package (pyproject, src/dtfit_experimental, tests)
docs/guides/notebooks/       # runnable tutorial notebooks
```

See each package's README for details:
[`packages/dtfit/README.md`](packages/dtfit/README.md) ·
[`packages/dtfit-experimental/README.md`](packages/dtfit-experimental/README.md).
The full documentation -- guides, API reference, method math, and validation --
lives in the [project wiki](https://github.com/ringavirda/science-nonline/wiki).
