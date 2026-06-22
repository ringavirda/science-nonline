# dtfit-experimental

Experimental structural adaptations of the `dtfit` EAC / LSI methods, plus the
full experiment / validation suite. This is a **separate distribution** that
depends on the stable [`dtfit`](../dtfit) package: it never ships inside the published
`dtfit` wheel, so the public API stays lean while new method modifications are
prototyped and evaluated here.

## What lives here

- **`dtfit_experimental`** — the experimental adaptations that remain in trial
  (`fit_lsi_basis`, `fit_joint`, `boosted_fit`). These build directly on `dtfit`'s
  internals (`dtfit.methods`, `dtfit._spectral`, `dtfit._backend`). (The
  overlapping-window ensemble `ensemble_fit` has been **promoted** into stable
  `dtfit`.)
- **`dtfit_experimental.experiments`** — the experiment suite: `cases/` (each
  adaptation in isolation), `domains/` (per-application-domain validation against
  the established baselines), shared `common/` framework, and `data/`.

When an adaptation proves effective across enough domains it is **promoted into
stable `dtfit`** and physically moved there; it is then imported from `dtfit`,
not from here. Already promoted: `PartitionedLSI` / `PartitionedEAC` (#1), the
GEMM-batched `fit_lsi_batched` / `project_spectra` and fused multi-channel
`PartitionedBatchLSI`, the adaptive-window `fit_eac_adaptive` (#6), the LSI
**oscillatory recipe** (`dtfit.fit_lsi(oscillatory=…, freq_param=…)` +
`dtfit.fft_frequency_seed`), and the multi-axis `FusedChiSquareDetector`.

## Install

From the repo root (both packages are editable; this one pulls in `dtfit`):

```bash
pip install -e packages/dtfit                    # stable dtfit
pip install -e packages/dtfit-experimental       # this package
pip install -e "packages/dtfit-experimental[bench]"  # + matplotlib/torch/statsmodels/pandas
```

## Run the suites

```bash
python -m dtfit_experimental.experiments.download_data          # fetch datasets
python -m dtfit_experimental.experiments.cases.run_suite        # per-adaptation cases
python -m dtfit_experimental.experiments.domains.run_domains    # per-domain validation
```

Both runners share a single generalized driver
(`dtfit_experimental.experiments._runner`); pass `--quick` for a smoke run,
`--jobs N` to cap worker processes (`--jobs 1` = serial). Each suite writes a
`report.md` + `figures/` per experiment and regenerates its index
(`cases/REPORTS.md`, `domains/DOMAINS.md`). See
[`src/dtfit_experimental/experiments/README.md`](src/dtfit_experimental/experiments/README.md)
for the real-data validation details.
