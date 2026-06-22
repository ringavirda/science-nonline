# experiments/domains -- per-domain validation of the merged dtfit methods

`experiments/cases/` answers *"does each EAC/LSI adaptation work in isolation?"*
(one optimization or structural idea per folder, scored on the promotion matrix).
This suite answers the next question a practitioner asks:

> **For each real application domain, what is the best combination of the
> methods, and does the merged pipeline actually hold up in a realistic
> setting?**

Each domain folder collects the levers that *cleared* their isolated evaluation,
**merges them into one pipeline**, and runs it on a realistic workload -- testing
validity (does it recover the right answer?), applicability (does it cover the
domain's real axes?) and usefulness (does the merge beat the obvious baseline /
earn its complexity?).

Each report has a **"Methods under test (dtfit)"** section explaining exactly
what each method does, a **"Baseline methods"** section listing the established
domain-standard methods compared against, and includes **real-data** tests.

## Domains

| domain | dtfit methods tested | compared against | data |
|--------|----------------------|------------------|------|
| [`forecasting/`](Domain-Forecasting) | LSI, EAC, #2 Fourier-LSI, #5 boosting, auto-merged pipeline | random walk, seasonal-naive, drift, poly-extrap, Holt-Winters ETS, Theta, (S)ARIMA, MLP, LSTM | 12 series x 2 horizons (structurally-correct model per series): 8 measured (COVID, USD/UAH, sunspots, CO_2, El Nino, Nile, ETTh1, weather) + **4 physics/signal waveforms** (RLC ring-down transient, AC + harmonics, AM carrier, linear chirp) |
| [`parameter_estimation/`](Domain-Parameter-Estimation) | LSI, EAC, #6 adaptive-EAC, #3 ensemble, #4 joint, merged selector | SciPy NLLS (LM), robust NLLS (soft-L1), MLP, Gaussian process | 16 nonlinear model families + applicability map; noise & outlier sweeps; sparse/transient/short-record/multi-channel; real COVID & USD/UAH rate recovery |
| [`big_data/`](Domain-Big-Data) | GEMM batch (`fit_lsi_batched`), fused streaming `PartitionedBatchLSI`, distributed `merge` (#1), streaming `EACFilter` | per-channel SciPy NLLS, vectorised polynomial `lstsq`, sklearn `SGDRegressor.partial_fit`, recursive least squares | 4 multi-channel panels + **real 321-channel** electricity; GB-scale memory wall, numerical stability, mergeability, online cost |
| [`embedded_control/`](Domain-Embedded-Control) | `EACFilter`, `LSIFilter`, `FilterBank` + fused chi^2 detector, `inflate` | Extended Kalman Filter, Recursive Least Squares, constant-accel Kalman, sliding-window refit | 4 plant shapes + applicability map; robustness (noise/outliers/dropout); multi-axis fault detection; deployable footprint; **real USD/UAH** streaming |

## Run

```bash
pip install -e '.[bench]'              # matplotlib, torch, statsmodels, pandas
python build_native.py                 # build the GIL-released C kernels

python -m experiments.domains.run_domains          # full run -> reports + DOMAINS.md
python -m experiments.domains.run_domains --quick  # fast smoke run

python -m experiments.domains.forecasting.run      # a single domain (also: --quick)
```

Reuses `experiments/common` (ReportWriter, metrics, baselines, plotting) and the
real datasets in `experiments/data/`, so the reports are consistent with the
experiment suite. The index and per-merge table are regenerated into
[`DOMAINS.md`](Domains-Reports) on every suite run.

## Reporting tone

The merges are conservative -- they compose only the validated levers and exclude
the ones that did not generalize -- and every report keeps the experiment suite's
honest-negative tone: daily FX stays near random-walk, the LTSF gap to deep
models is predictable global structure (not noise), joint fitting buys parsimony
not accuracy on clean channels, streaming trades throughput for bounded memory,
the GPU helps only resident data, and online change-detection is bounded by
measurement SNR rather than the algorithm.
