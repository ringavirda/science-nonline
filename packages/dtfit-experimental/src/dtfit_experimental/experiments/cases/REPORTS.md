# dtfit experiment suite — reports index

Each case is a self-contained folder with a `backend.py` (the compute — the
single source of truth for its simulation/estimation/data) and a Jupyter
notebook (the report: tables, figures, narrative) plus its `figures/`. Open a
notebook in Jupyter and re-run it, or execute it headless with
`jupyter nbconvert --to notebook --execute --inplace <notebook>`.


## Experiments

| # | experiment | notebook |
|---|---|---|
| 1 | Control-systems system identification | [01_control_systems.ipynb](01_control_systems/01_control_systems.ipynb) |
| 2 | Big data / streaming (scaling law) | [02_big_data_streaming.ipynb](02_big_data_streaming/02_big_data_streaming.ipynb) |
| 3 | Noise & robustness sweep | [03_noise_robustness.ipynb](03_noise_robustness/03_noise_robustness.ipynb) |
| 4 | Real-world forecasting (train/holdout) | [04_realworld_forecasting.ipynb](04_realworld_forecasting/04_realworld_forecasting.ipynb) |
| 5 | GPS positioning & trajectory forecast | [05_gps_trajectory.ipynb](05_gps_trajectory/05_gps_trajectory.ipynb) |
| 6 | LTSF benchmark vs published R&D results | [06_benchmark_ltsf.ipynb](06_benchmark_ltsf/06_benchmark_ltsf.ipynb) |
| 7 | Parallel scaling & architecture adaptability | [07_parallel_scaling.ipynb](07_parallel_scaling/07_parallel_scaling.ipynb) |
| 8 | GEMM-batched projection throughput (CPU/GPU) | [08_gpu_batched_projection.ipynb](08_gpu_batched_projection/08_gpu_batched_projection.ipynb) |
| 9 | Embedded footprint (latency, memory, MCU fit) | [09_embedded_footprint.ipynb](09_embedded_footprint/09_embedded_footprint.ipynb) |
| 10 | Fused map-reduce + GEMM-batched LSI (multi-channel big data) | [10_fused_partitioned_batched.ipynb](10_fused_partitioned_batched/10_fused_partitioned_batched.ipynb) |

## Architecture-adaptation effectiveness matrix

Each novel EAC/LSI adaptation, scored across the experiments that exercised it (win / partial / loss / n/a). **Promotion gate**: a clear win on ≥ 2 distinct application domains.

| adaptation | per-domain result | decision |
|---|---|---|
| #1 map-reduce LSI/EAC (PartitionedLSI/EAC) | big data: win, parallel: win, forecasting: n/a | PROMOTE — exact one-pass distributed estimator; enables the big-data scaling law and the parallel map-reduce. Clears the gate. |
| #2 pluggable orthogonal basis (fit_lsi_basis) | control: partial, forecasting: partial, ltsf: loss | Keep experimental — Fourier basis expresses periodic models cleanly, but a window-local seasonal term only helped the cleanest periodic LTSF series (electricity) and hurt elsewhere (a 96-point period estimate drifts out of phase over long horizons); it did not beat the tuned forecasting baselines. |
| #3 overlapping-window ensemble (ensemble_fit) | noise/outliers: partial, gps: n/a | PROMOTED (specialized tool) — on densely-contaminated data it is the whole-window rejection route that stays stable where `fit_eac`'s in-fit robust loss diverges, so it ships in stable `dtfit` as `ensemble_fit` — the complement to the robust loss, not a general-purpose default. |
| #4 joint multi-channel fit (fit_joint) | control: loss, gps: n/a | Keep experimental — on cleanly-identifiable channels the dedicated solver already wins; value is parameter parsimony / consistency, not accuracy. |
| #5 stage-wise boosting (boosted_fit) | forecasting: win, ltsf: n/a | Keep experimental — a clear win on CO2 (trend+season) but only one domain demonstrated; promote if a second domain confirms. |
| #6 adaptive-window EAC (`fit_eac(window_mode="curvature")`) | transient fit: win | PROMOTED — recovers localized-transient parameters well; folded into stable `dtfit.fit_eac` as `window_mode="curvature"` (there is no separate `fit_eac_adaptive` symbol). |

## Promotion outcome

- **Promoted to the stable API:** the map-reduce estimators (`PartitionedLSI` / `PartitionedEAC`, #1), the overlapping-window ensemble (`ensemble_fit`, #3 — the complementary whole-window-rejection robustness tool), and adaptive-window EAC (#6 — folded into `dtfit.fit_eac` as `window_mode="curvature"`), all re-exported from `dtfit` and documented. (Later work also promoted the GEMM-batched projection `fit_lsi_batched`, the fused `FusedChiSquareDetector`, the LSI oscillatory recipe, and the stochastic-series solution.)

- **Kept experimental in `dtfit_experimental`:** #2 (`fit_lsi_basis`), #4 (`fit_joint`) and #5 (`boosted_fit`), with the honest per-experiment findings above — they have not (yet) cleared the ≥2-domain promotion gate.