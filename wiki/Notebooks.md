# dtfit guide notebooks

A visual, runnable tour of the **dtfit** public API. Each notebook is executed
with its figures and outputs embedded, so it reads as a reference even without
running it -- and every cell runs end-to-end if you do.

## Read in order

| # | Notebook | Covers |
|---|----------|--------|
| 01 | [Quickstart](Notebook-01-Quickstart) | The core idea, a first `fit_lsi`, the `FittingResult` (params, uncertainty, prediction bands), `auto_estimate` |
| 02 | [Fitting methods](Notebook-02-Fitting-Methods) | `fit_lsi` (+ oscillatory recipe, `fft_frequency_seed`), `fit_eda` / `fit_eda_adaptive`, `fit_dsb` (+ `find_degree`), comparing fits |
| 03 | [Models & automatic fitting](Notebook-03-Models-and-Auto) | The model catalog (`models.*`, `CATALOG`), self-seeding `Model.fit`, composition with `+`, `suggest_models`, `auto_estimate`, `auto_forecast` |
| 04 | [scikit-learn estimator](Notebook-04-Sklearn-Estimator) | `NonlineRegressor` -- `fit`/`predict`/`score`, `Pipeline`, `GridSearchCV`, `cross_val_score` |
| 05 | [Streaming trackers](Notebook-05-Streaming) | `EDAFilter` / `LSIFilter` (`partial_fit`, drift detection), `FilterBank`, `FusedChiSquareDetector` |
| 06 | [Scaling](Notebook-06-Scaling) | `fit_many` (+ `FittingProblem`, `BatchFittingResult`), `project_spectra` / `fit_lsi_batched`, `PartitionedLSI` / `PartitionedEDA` / `PartitionedBatchLSI` (one-pass & map-reduce) |
| 07 | [Diagnostics](Notebook-07-Diagnostics) | `fit_report`, `residual_diagnostics`, `FitDisplay` / `ResidualsDisplay`, `enable_logging`, `to_dict` / `from_dict` |

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
python docs/guides/notebooks/build.py                          # rebuild + execute all
python docs/guides/notebooks/build.py 03 05                    # only notebooks 03 and 05
```

Edit the cell definitions in [`build.py`](https://github.com/ringavirda/science-nonline/blob/main/docs/guides/notebooks/build.py) -- not the `.ipynb` files --
then re-run it.
