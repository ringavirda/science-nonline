# dtfit experiment suite -- reports index

Each experiment is a self-contained folder with a `backend.py` (the compute) and a Jupyter notebook (the report) plus its `figures/`.


## Experiments

| # | experiment | report | status | runtime (s) |
|---|---|---|---|---|
| 1 | Control-systems system identification | [01_control_systems/report.md](Case-01-Control-Systems) | ok | 2 |
| 2 | Big data / streaming (scaling law) | [02_big_data_streaming/report.md](Case-02-Big-Data-Streaming) | ok | 5 |
| 3 | Noise & robustness sweep | [03_noise_robustness/report.md](Case-03-Noise-Robustness) | ok | 4 |
| 4 | Real-world forecasting (train/holdout) | [04_realworld_forecasting/report.md](Case-04-Realworld-Forecasting) | ok | 1 |
| 5 | GPS positioning & trajectory forecast | [05_gps_trajectory/report.md](Case-05-GPS-Trajectory) | ok | 1 |
| 6 | LTSF benchmark vs published R&D results | [06_benchmark_ltsf/report.md](Case-06-Benchmark-LTSF) | ok | 5 |
| 7 | Parallel scaling & architecture adaptability | [07_parallel_scaling/report.md](Case-07-Parallel-Scaling) | ok | 17 |
| 8 | GEMM-batched projection throughput (CPU/GPU) | [08_gpu_batched_projection/report.md](Case-08-GPU-Batched-Projection) | ok | 1 |
| 9 | Embedded footprint (latency, memory, MCU fit) | [09_embedded_footprint/report.md](Case-09-Embedded-Footprint) | ok | 0 |
| 10 | Fused map-reduce + GEMM-batched LSI (multi-channel big data) | [10_fused_partitioned_batched/report.md](Case-10-Fused-Partitioned-Batched) | ok | 3 |

## Architecture-adaptation effectiveness matrix

Each novel EAC/LSI adaptation, scored across the experiments that exercised it (win / partial / loss / n/a). **Promotion gate**: a clear win on >= 2 distinct application domains.

| adaptation | per-domain result | decision |
|---|---|---|
| #1 map-reduce LSI/EAC (PartitionedLSI/EAC) | big data: win, parallel: win, forecasting: n/a | PROMOTE -- exact one-pass distributed estimator; enables the big-data scaling law and the parallel map-reduce. Clears the gate. |
| #2 pluggable orthogonal basis (fit_lsi_basis) | control: partial, forecasting: partial, ltsf: loss | Keep experimental -- Fourier basis expresses periodic models cleanly, but a window-local seasonal term only helped the cleanest periodic LTSF series (electricity) and hurt elsewhere (a 96-point period estimate drifts out of phase over long horizons); it did not beat the tuned forecasting baselines. |
| #3 overlapping-window ensemble (ensemble_fit) | noise/outliers: partial, gps: n/a | Keep experimental -- helps EAC at low outlier rates but unstable once many windows are corrupted; LSI's built-in smoothing is the more reliable robustness route. |
| #4 joint multi-channel fit (fit_joint) | control: loss, gps: n/a | Keep experimental -- on cleanly-identifiable channels the dedicated solver already wins; value is parameter parsimony / consistency, not accuracy. |
| #5 stage-wise boosting (boosted_fit) | forecasting: win, ltsf: n/a | Keep experimental -- a clear win on CO2 (trend+season) but only one domain demonstrated; promote if a second domain confirms. |
| #6 adaptive-window EAC (fit_eac(window_mode="curvature")) | transient fit: win | Promoted into dtfit as fit_eac(window_mode="curvature") -- domain-validated on localized transients/peaks. |

## Promotion outcome

- **Promoted to the stable API:** the map-reduce estimators (`PartitionedLSI` / `PartitionedEAC`, adaptation #1) -- re-exported from `dtfit` and documented; they cleared the gate (big-data + parallel). Curvature-adaptive-window EAC (adaptation #6) was also promoted, as the `fit_eac(window_mode="curvature")` path of the stable `fit_eac` (domain-validated on transients/peaks).

- **Kept experimental in `dtfit_experimental`:** #2-#5, with the honest per-experiment findings above. They remain available and documented but did not (yet) clear the >=2-domain promotion gate.


