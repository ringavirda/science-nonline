# experiments/domains — per-domain validation of the merged dtfit methods

`experiments/cases/` answers *"does each EAC/LSI adaptation work in isolation?"*
(one optimization or structural idea per folder, scored on the promotion matrix).
This suite answers the next question a practitioner asks:

> **For each real application domain, what is the best combination of the
> methods, and does the merged pipeline actually hold up in a realistic
> setting?**

Each domain folder collects the levers that *cleared* their isolated evaluation,
**merges them into one pipeline**, and runs it on a realistic workload — testing
validity (does it recover the right answer?), applicability (does it cover the
domain's real axes?) and usefulness (does the merge beat the obvious baseline /
earn its complexity?).

Each notebook has a **"Methods under test (dtfit)"** section explaining exactly
what each method does, a **"Baseline methods"** section listing the established
domain-standard methods compared against, and includes **real-data** tests.

Each domain is a self-contained folder with a `backend.py` (the compute — the
single source of truth for its simulation/estimation/data) and a Jupyter
notebook (the report: tables, figures, narrative) plus its `figures/`.

## Domains

| domain | dtfit methods tested | compared against | data |
|--------|----------------------|------------------|------|
| [`forecasting/`](forecasting/forecasting.ipynb) | LSI, EAC, #2 Fourier-LSI, #5 boosting, auto-merged pipeline | random walk, seasonal-naïve, drift, poly-extrap, Holt-Winters ETS, Theta, (S)ARIMA, MLP, LSTM | 12 series × 2 horizons (structurally-correct model per series): 8 measured (COVID, USD/UAH, sunspots, CO₂, El Niño, Nile, ETTh1, weather) + **4 physics/signal waveforms** (RLC ring-down transient, AC + harmonics, AM carrier, linear chirp) |
| [`parameter_estimation/`](parameter_estimation/parameter_estimation.ipynb) | LSI, EAC, #6 adaptive-EAC, #3 ensemble, #4 joint, merged selector | SciPy NLLS (LM), robust NLLS (soft-L1), MLP, Gaussian process | 16 nonlinear model families + applicability map; noise & outlier sweeps; sparse/transient/short-record/multi-channel; real COVID & USD/UAH rate recovery |
| [`big_data/`](big_data/big_data.ipynb) | GEMM batch (`fit_lsi_batched`), fused streaming `PartitionedBatchLSI`, distributed `merge` (#1), streaming `EACFilter` | per-channel SciPy NLLS, vectorised polynomial `lstsq`, sklearn `SGDRegressor.partial_fit`, recursive least squares | 4 multi-channel panels + **real 321-channel** electricity; GB-scale memory wall, numerical stability, mergeability, online cost |
| [`embedded_control/`](embedded_control/embedded_control.ipynb) | `EACFilter`, `LSIFilter`, `FilterBank` + fused χ² detector, `inflate` | Extended Kalman Filter, Recursive Least Squares, constant-accel Kalman, sliding-window refit | 4 plant shapes + applicability map; robustness (noise/outliers/dropout); multi-axis fault detection; deployable footprint; **real USD/UAH** streaming |
| [`realtime_gps/`](realtime_gps/realtime_gps.ipynb) | streaming `LSIFilter`/`EACFilter` (external regressors), full-IMU strapdown fused inside LSI, fused NIS/CUSUM detector | constant-accel Kalman, gyro-aided coordinated-turn EKF | simulated 9-DOF rig (3-D maneuvering target, GPS fixes, 3-axis gyro + accelerometer) with dropouts & multipath glitches |
| [`realtime_gps_hw/`](realtime_gps_hw/README.md) | the same streaming LSI/EAC trackers running **on the Arduino Nano 33 BLE M4F** over a live GPS+IMU stream (hardware twin of `realtime_gps`) | constant-accel Kalman / CT-EKF on the same captured logs; the PC reference (bit-identical) | real captured rig logs + public RTK/INS datasets (comma2k19, UrbanNav) |
| [`stochastic_series/`](stochastic_series/stochastic_series.ipynb) | `fit_stochastic`, `StochasticModel.simulate`, `StochasticFilter`, the functional estimators (Hurst / AR(1) / GARCH / cycle), vendored ADF (**promoted to `dtfit.stochastic`**) | OLS/GPH, R/S, DFA (Hurst), lag-1 ACF, FFT-peak, random walk, drift, AR(1), ARIMA, ETS, Theta, seasonal-naïve, `LSIFilter` (filter cost) | 6 process families (ARFIMA, AR(1)/OU, GARCH, AR(2) cycle, trend+cycle) recovered vs known truth + regime router; forecast skill; **7-series real gallery** (Nile, sunspots, CO₂, GDP, T-bill, USD/UAH, LTSF-FX); generative round-trip; streaming tracking + break detection |

## Run

```bash
pip install -e '.[bench]'              # matplotlib, torch, statsmodels, pandas, jupyter
python build_native.py                 # build the GIL-released C kernels

# open a notebook and re-run it interactively
jupyter lab experiments/domains/forecasting/forecasting.ipynb

# or execute headless (writes outputs + figures in place)
jupyter nbconvert --to notebook --execute --inplace \
    experiments/domains/forecasting/forecasting.ipynb
```

Each notebook carries a config block of knobs near the top (sized for a few-minute
run by default; comments show how to scale up). The backends reuse
`experiments/common` (metrics, baselines, datasets, plotting) and the real
datasets in `experiments/data/`. [`DOMAINS.md`](DOMAINS.md) indexes the notebooks.

## Reporting tone

The merges are conservative — they compose only the validated levers and exclude
the ones that did not generalize — and every report keeps the experiment suite's
honest-negative tone: daily FX stays near random-walk, the LTSF gap to deep
models is predictable global structure (not noise), joint fitting buys parsimony
not accuracy on clean channels, streaming trades throughput for bounded memory,
the GPU helps only resident data, and online change-detection is bounded by
measurement SNR rather than the algorithm.
