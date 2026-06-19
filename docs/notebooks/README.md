# dtfit guide notebooks

A visual, runnable tour of the **dtfit** public API. Each notebook is executed
with its figures and outputs embedded, so it reads as a reference even without
running it — and every cell runs end-to-end if you do.

## Read in order

| # | Notebook | Covers |
|---|----------|--------|
| 01 | [Quickstart](01_quickstart.ipynb) | The core idea, a first `fit_lsi`, the `FittingResult` (params, uncertainty, prediction bands), `auto_estimate` |
| 02 | [Fitting methods](02_fitting_methods.ipynb) | `fit_lsi` (+ oscillatory recipe, `fft_frequency_seed`), `fit_eda` / `fit_eda_adaptive`, `fit_dsb` (+ `find_degree`), comparing fits |
| 03 | [Models & automatic fitting](03_models_and_auto.ipynb) | The model catalog (`models.*`, `CATALOG`), self-seeding `Model.fit`, composition with `+`, `suggest_models`, `auto_estimate`, `auto_forecast` |
| 04 | [scikit-learn estimator](04_sklearn_estimator.ipynb) | `NonlineRegressor` — `fit`/`predict`/`score`, `Pipeline`, `GridSearchCV`, `cross_val_score` |
| 05 | [Streaming trackers](05_streaming.ipynb) | `EDAFilter` / `LSIFilter` (`partial_fit`, drift detection), `FilterBank`, `FusedChiSquareDetector` |
| 06 | [Scaling](06_scaling.ipynb) | `fit_many` (+ `FittingProblem`, `BatchFittingResult`), `project_spectra` / `fit_lsi_batched`, `PartitionedLSI` / `PartitionedEDA` / `PartitionedBatchLSI` (one-pass & map-reduce) |
| 07 | [Diagnostics](07_diagnostics.ipynb) | `fit_report`, `residual_diagnostics`, `FitDisplay` / `ResidualsDisplay`, `enable_logging`, `to_dict` / `from_dict` |

Between them the notebooks exercise the entire `dtfit` public namespace.

## Running them

```bash
pip install "dtfit[viz]"        # the library + matplotlib
pip install jupyterlab          # or use VS Code's notebook UI
jupyter lab                     # open any notebook and run all cells
```

## Regenerating

The notebooks are generated and executed from a single script, so they stay in
sync with the API and reproduce deterministically (fixed RNG seed):

```bash
pip install nbformat nbclient ipykernel
python -m ipykernel install --user --name dtfit-docs   # one-time kernel for execution
python docs/notebooks/build.py                          # rebuild + execute all
python docs/notebooks/build.py 03 05                    # only notebooks 03 and 05
```

Edit the cell definitions in [`build.py`](build.py) — not the `.ipynb` files —
then re-run it.
