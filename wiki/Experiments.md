# Experiments -- the notebooks behind every number

Every number the experiment and domain pages quote comes from a notebook under
[`packages/dtfit-experimental/experiments/`](https://github.com/ringavirda/science-nonline/tree/main/packages/dtfit-experimental/experiments).
A notebook is one experiment and its report: the claim at the top with what
would falsify it, the code that measures it, the numbers it prints, the reading
of those numbers, and the limits that were not tested. A notebook is edited by
hand and re-executed; nothing generates it, and the pages here are summaries
that link to it.

## Layout

```
packages/dtfit-experimental/
  experiments/
    README.md       # the index: claim, runtime and data needed, per notebook
    method/         # 11-19: what the image is and where each form of the method stops
    technology/     # 21-24: the filter, the reduce at scale, the stochastic image, the footprint
    realdata/       # 31-36: reference datasets, forecasting, GPS, the two archives, real stochastic series
    results/        # the tables the notebooks export, small CSV files, tracked
    data/           # downloaded datasets, not tracked
  src/dtfit_experimental/study/   # what the notebooks import: baselines, simulators, loaders, metrics
packages/dtfit-hardware/experiments/rig.ipynb   # the recorded drives and the on-chip filter
```

## Running

```bash
pip install -e "packages/dtfit-experimental[bench]"
python -m dtfit_experimental.study.download_data     # fetch the datasets, once

jupyter lab packages/dtfit-experimental/experiments/method/12_discrete_image.ipynb
# or headless, outputs written in place
jupyter nbconvert --to notebook --execute --inplace \
    packages/dtfit-experimental/experiments/method/12_discrete_image.ipynb
```

`DTFIT_QUICK=1` shrinks every notebook to a run of seconds; the package's
`tests/test_notebooks.py` executes each committed notebook that way, and a
check cell that no longer supports its claim fails the run. `DTFIT_DATA` moves
the dataset directory. A section on real data names its source and skips with
a visible message when the data is absent.

## Index

The summaries: [method studies](Experiments-Method) for notebooks 11 to 19,
the [domain pages](Domains) for the rest.

### method/

| Notebook | Establishes |
|---|---|
| [11_source_forms](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/11_source_forms.ipynb) | the three source criteria (spectrum balance, monomial integral, equal areas) against the image route, and where each stops |
| [12_discrete_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/12_discrete_image.ipynb) | the image is the discrete statistic: zero bias and the pointwise fit's error on uniform, clustered and random grids, against the quadrature reading of the same criterion |
| [13_two_bases](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/13_two_bases.ipynb) | Legendre against block: reconstruction by coefficient count, Gram conditioning by order and grid, efficiency |
| [14_robust_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/14_robust_image.ipynb) | robustness is the reweighted statistic in either basis; the plain block fit has none; bursts |
| [15_families](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/15_families.ipynb) | sixteen model families and the 25-scenario accuracy corpus against NLLS: recovery, noise sweep, forecast, sampling regimes, model mismatch |
| [16_weak_form](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/16_weak_form.ipynb) | weak-form ODE identification against finite differences and the solved fit; the seeded fit; where it breaks |
| [17_evolution_matrix](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/17_evolution_matrix.ipynb) | every historical stage of the method on one problem set: batch, streaming and map-reduce |
| [18a_block_basis_synthetic](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/18a_block_basis_synthetic.ipynb) | where the block basis is ahead, needed, at parity and behind: known epochs, window alignment, aggregated data, gapped grids, stored numbers |
| [18b_block_basis_real_data](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/18b_block_basis_real_data.ipynb) | the same map on 120 GPS station series and on hourly temperature read as 3 h and 6 h means |
| [19_adaptations_in_trial](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/19_adaptations_in_trial.ipynb) | each adaptation of `dtfit_experimental` against the plain route on its own signal class, with a verdict |

### technology/

| Notebook | Establishes | Summary |
|---|---|---|
| [21_filter_form](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/21_filter_form.ipynb) | the filter on the window image against a parameter-space EKF: accuracy by tuning, anomalies, dropouts, jump detection, cost | [Embedded control](Domain-Embedded-Control) |
| [22_scale](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/22_scale.ipynb) | exact merge, a lost shard, float32 numerics, memory against record length, `fit_many`, the GPU projection, the surrogate trap | [Big data](Domain-Big-Data) |
| [23_stochastic_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/23_stochastic_image.ipynb) | the second-order image: estimators by route, the regime router over 56 cases, the online filter, the generator | [Stochastic series](Domain-Stochastic-Series) |
| [24_embedded_footprint](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/24_embedded_footprint.ipynb) | state size, step cost and worst step of the filter against Kalman and refitting, by microcontroller | [Embedded control](Domain-Embedded-Control) |

### realdata/

| Notebook | Establishes | Summary |
|---|---|---|
| [31_reference_datasets](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/31_reference_datasets.ipynb) | NIST StRD from both certified starts, Puromycin, two real growth rates | [Parameter estimation](Domain-Parameter-Estimation) |
| [32_forecasting](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/32_forecasting.ipynb) | twelve series against the forecasting baselines, the LTSF protocol, traffic | [Forecasting](Domain-Forecasting) |
| [33_gps_simulation](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/33_gps_simulation.ipynb) | a simulated maneuvering target from GPS fixes and a 9-DOF IMU against Kalman-CA and a coordinated-turn EKF | [Real-time GPS](Domain-Realtime-GPS) |
| [34_gps_benchmark](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/34_gps_benchmark.ipynb) | two literature trajectories and two GSDC trips with RTK truth, where the baselines win | [Real-time GPS](Domain-Realtime-GPS) |
| [35_archive_showcase](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/35_archive_showcase.ipynb) | the NOAA and NGL archives reduced on a workstation and refitted on a Raspberry Pi 5 | [Archive showcase](Domain-Image-Showcase) |
| [36_stochastic_real](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/36_stochastic_real.ipynb) | the stochastic router on seven real records against the random walk and the classical forecasters | [Stochastic series](Domain-Stochastic-Series) |

### hardware

| Notebook | Establishes | Summary |
|---|---|---|
| [rig](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-hardware/experiments/rig.ipynb) | three recorded drives against Kalman-CA and CT-EKF, and the on-chip float32 filter: cost, memory, agreement with float64 | [GPS hardware rig](Domain-Realtime-GPS-Hardware) |

## Baselines

The established methods each experiment is scored against live in
`dtfit_experimental.study.baselines` and
`dtfit_experimental.study.classical_stochastic`; [Baselines](Experimental-Baselines)
documents them. NLLS is the reference throughout: where theory gives the image
no advantage the notebooks report parity as parity.
