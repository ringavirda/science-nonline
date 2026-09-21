# dtfit-experimental

Experimental structural adaptations of the `dtfit` EAC / LSI methods, plus the
full experiment / validation suite. This is a **separate distribution** that
depends on the stable [`dtfit`](../dtfit) package: it never ships inside the published
`dtfit` wheel, so the public API stays lean while new method modifications are
prototyped and evaluated here.

## What lives here

- **`dtfit_experimental`** -- the experimental adaptations that remain in trial:
  `InformationFilter`, the inverse-covariance fusion primitive, and the image
  bases `FourierBasis`, `ChebyshevBasis`, `LaguerreBasis` (reached through
  `dtfit.fit(basis=...)`). These build directly on `dtfit`'s internals
  (`dtfit._core._spectral`, `dtfit._core._backend`, `dtfit._symbolic`, `dtfit._stats`).
  Notebook 19 (`experiments/method/19_adaptations_in_trial.ipynb`) measures each
  against the plain route and carries its verdict.
- `weak_ode` -- weak-form ODE identification: rate laws linearized by clearing denominators or eliminating hidden states, fit by least squares with no ODE solve and no p0 (`weak_operators`, `fit_logistic`, `fit_michaelis_menten`, `fit_lotka_volterra_prey`), plus `seed_nlls`, which solves the ODE and refines the weak estimate by nonlinear least squares to full accuracy.
- `local_time` -- `LocalTimeFilter`, the window image filter for a polynomial trend carried in the time of its newest sample: the origin moves onto every new sample through the exact coefficient map `shift_matrix`, so the estimate does not depend on the clock's zero and the process noise is stated per unit of time. Measured on the recorded rig logs in `packages/dtfit-hardware/experiments/rig.ipynb`.
- **`dtfit_experimental.streaming`** -- `FilterBank` and the fused chi-square
  detector over `dtfit`'s filters: experiment tooling rather than library API.
- **`dtfit_experimental.study`** -- what the notebooks import: baselines,
  simulators, dataset loaders, metrics, cost measurement, plotting and notebook
  helpers, imported module by module.
- **`experiments/`** -- the experiments themselves, outside `src/`: one notebook
  per experiment under `method/`, `technology/` and `realdata/`, the tables they
  export under `results/`, and the downloaded datasets under `data/` (ignored).
  [`experiments/README.md`](experiments/README.md) is the index: the claim of
  each notebook, its runtime and the data it needs.

When an adaptation proves effective across enough domains it is **promoted into
stable `dtfit`** and physically moved there; it is then imported from `dtfit`,
not from here. Already promoted: the oscillatory recipe
(`dtfit.fit(..., freq_param=...)` with `dtfit.image.fft_frequency_seed`).
Notebook 19 carries the verdict on each adaptation still in trial.

## Install

From the repo root (both packages are editable; this one pulls in `dtfit`):

```bash
pip install -e packages/dtfit                    # stable dtfit
pip install -e packages/dtfit-experimental       # this package
pip install -e "packages/dtfit-experimental[bench]"  # + matplotlib/torch/statsmodels/pandas
```

## Run the experiments

Each notebook is one experiment and its report: the claim at the top, the code
that measures it, the numbers it prints and the prose that reads them. There is
no generator; a notebook is edited by hand and re-executed.

```bash
python -m dtfit_experimental.study.download_data                # fetch datasets

# open and re-run interactively
jupyter lab experiments/method/12_discrete_image.ipynb

# or execute headless (writes outputs in place)
jupyter nbconvert --to notebook --execute --inplace \
    experiments/method/12_discrete_image.ipynb
```

`DTFIT_QUICK=1` shrinks every notebook to a run of seconds; `tests/test_notebooks.py`
executes each committed notebook that way. `DTFIT_DATA` moves the dataset
directory. A section on real data says where the data comes from and skips with
a visible message when it is absent.
