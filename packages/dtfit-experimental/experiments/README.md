# Experiments

Every number this study quotes comes from a notebook here. Each notebook is
one experiment and its report: open it, read the claim at the top, run it,
and compare the numbers it prints against the prose that follows.

## Index

Runtime is the full run on a workstation; "data" is what has to be on disk
beyond what `python -m dtfit_experimental.study.download_data` fetches.

### method/

| Notebook | Claim | Runtime | Data |
|---|---|---|---|
| 11_source_forms | The three source criteria (spectrum balance, monomial integral, equal areas) against the current image route: the balance's adequacy indicators part ways past a best degree, the monomial weight matrix loses definiteness by order 12, the integral criterion fails past one period, the truncated equal-areas form is biased outside the small-angle range, and its variance trade holds only inside the polynomial base | under 10 seconds | none |
| 12_discrete_image | The image is the discrete statistic: S and G are exactly the basis products on the samples, a fit on it reproduces the pointwise fit on a uniform, a clustered and a random grid, reading the same criterion by quadrature loses that parity through the end samples and through uneven density, and order above what order_for reports is free | about a minute | none |
| 13_two_bases | h versus p refinement: Legendre wins a smooth target, block is exact on a step only where a window boundary falls on the jump and at parity with Legendre otherwise, the Gram conditioning by order and grid, the cost of restricting a linear fit to either basis | a few seconds | none |
| 14_robust_image | Where the robustness of the image comes from: at 10 percent scattered 8-sigma contamination the plain block fit tracks plain NLLS and plain Legendre within 11 percent (4.77 against 5.30 and 5.11 percent median parameter error), and reweighting the image brings either basis under a tenth of that (0.44 and 0.45 percent); on clean data the reweighted fit's curve RMSE is 1.007 to 1.040 times the plain fit's for Legendre and 0.999 to 1.061 for block over five blocks of 60 draws; under a contiguous burst robust Legendre stays flat over run lengths 1 to 40 (1.18 to 1.35 times clean NLLS) while robust block grows 2.9 times, no window count from 4 to 48 beats Legendre once the burst is long, and at burst 40 on 48 windows 4.9 windows on average are more than half outliers | about half a minute | none |
| 15_families | Recovery across sixteen model families and dtfit's 25-scenario corpus: the median parameter-error ratio to NLLS is 1.000 for Legendre and 1.025 for block at 5 percent noise and stays inside 0.997 to 1.000 and 0.985 to 1.055 from 2 to 30 percent, while the list of families trailing by a tenth does not reproduce from one noise draw to the next; the automatic basis returns Legendre on all sixteen; the historical integral least squares costs 7.74 times NLLS on the damped oscillation where the image costs 1.01, and direct equal areas 1.153 in the corpus median; held-out forecasts of five routes sit within 1.2 percent at the median; none of three sampling regimes (concentrated transient, sparse irregular, short record) separates the image from NLLS beyond the scatter over four seed blocks; a structurally wrong model is rescued by no estimator, and the weakest mismatch (a biexponential fitted as one decay) shows in the residual autocorrelation, 4.1 times the correct fit's, not in the R2 gap of 0.012 | about three minutes | none |
| 16_weak_form | Weak-form identification of an ODE law from samples, with no start and no ODE solve: the logistic recovers to 0.082 percent at 2 percent noise; against a solved nonlinear least-squares fit started near the truth and handed the true initial state it is 1.69 to 4.39 times less accurate at every one of 9 law and noise cells and 30 to 200 times faster; it beats finite-difference regression at every cell (0.060 against 0.104 percent on the logistic, 67 to 504 percent for finite differences on the damped oscillator); as the start of the solved fit it reaches the good-start error to 0.005 percentage points and removes the 97 to 100 percent failures of a 3x start on Michaelis-Menten, and buys only a factor of two in time on the other two laws; thinning the samples costs it more than the solved fit (ratio 2.74 at 600 samples, 5.18 at 25), a short record does not, and a stiff pair breaks it (88.3 and 58.0 percent on roots -1 and -10 against 1.31 and 0.30) | about five minutes | none |
| 18a_block_basis_synthetic | Where the block basis is the right one, on synthetic series: parity with Legendre on a jump at equal coefficient count and a loss on a smooth term, window edges on known epochs buy the jump columns back, the segment basis against block-on-epochs, a block Gram that stays usable on bursty records where the Legendre Gram goes singular (and loses where it does not), alignment to unknown epochs against an oracle, the equal-areas fit of per-window totals against the midpoint reading, and the storage budget on an irregular grid | about a minute | none |
| 18b_block_basis_real_data | The block basis on real archives: over 120 NGL stations with window edges on the step database's epochs the image fit lands 0.032 standard errors (median, 0.099 at the 90th percentile) from the pointwise shifts at 8 windows a year, where the Legendre image at the same K leaves 0.393 and uniform windows 0.435, with a block Gram under cond 130 against a Legendre one up to 9.8e18, and the advantage is gone at 4 windows a year; with the epochs unknown, a self-scaled detector cuts detections from 9.19 and 47.96 a station to 1.49 and 1.43 but finds only 18 and 32 percent of the 92 large steps, so the aligned velocity sits at 0.142 and 0.100 mm/yr from MIDAS between the database-epoch fit's 0.088 and the no-step fit's 0.180; on 149 ISD station-years known only as 3 h and 6 h means the equal-areas fit returns the diurnal amplitude at 1.0001 and 1.0048 of the hourly fit's, the midpoint reading 0.9775 and 0.9058; and the applicability map, 16 rows read from the tables of 18a and 18b (5 ahead, 4 needed, 4 parity, 3 behind) | about 13 minutes with the archives; the map section runs from tracked tables | the NGL and ISD archives, skipped when absent |
| 19_adaptations_in_trial | Five adaptations that were never promoted, each against the route a caller already has, under one rule (a median ratio below 0.8 with separated 90 percent bootstrap intervals over 30 seeds keeps it): the alternate spectral basis is at parity with the shipped Fourier basis on a sine (0.991) and broken on a decay (144), staged boosting trails `auto_forecast` on the series built for it (1.56) and wins only one hand-picked real series where both trail a random walk, the joint shared-parameter fit loses a well-identified construction and its pooled weak-identifiability ratio (0.792) sits inside overlapping intervals, so those three modules are deleted and their sections read the tables of the measurement; information-form fusion agrees with a single pass to 7e-15 and with the merged-image fit to 8e-9 and stays, and of the shipped bases Fourier reaches a 1e-6 reconstruction of a cycle at 9 coefficients against Legendre's 26 while Laguerre's 9 against 10 on a decay is parity | a few seconds | none |

### technology/

| Notebook | Claim | Runtime | Data |
|---|---|---|---|
| 21_filter_form | The window image filter against a parameter-space EKF on four synthetic plants, eight seeds: with `min_window` held equal the block and Legendre images enter the 10 percent band 1.5 to 1.9 steps after the first measurement, so convergence speed is the window floor and not the basis; the process noise decides accuracy, a tuning matched to static parameters leading the EKF on the damped oscillator (0.60 against 1.07 percent) and the first-order rise in mean, median and most seeds, at parity on the accelerating trajectory and a sixth behind on the AC sinusoid (0.38 against 0.33), while the inherited tracking tuning loses everywhere it is scored; a sliding-window refit is the worst estimator that recovers parameters (44 percent) and a wrong model diverges or scores several times worse; robustness is the window's reweighting, not the basis or the tuning; a dropout is a thinner window; the adaptive window re-detects an amplitude jump in 0.5 s against the fixed window's 5.8 s; a fused chi-square test over three streams flags a synchronized fault in 1 step against 19 for per-axis detectors and never for a Kalman bank; 108 to 132 microseconds a step against the EKF's 12.8 | about two minutes | none |
| 22_scale | The image as one additive statistic at scale: eight chunk images merged in either order reproduce the whole-data fit to 4.92e-14 relative in the coefficients, and a fit missing one of eight shards moves the parameters by 0.01 and 0.02 of a standard error; the float64 reduce stays at 1.9e-15 at every chunk count on a high-dynamic-range integrand where naive float32 sits at 1.3e-06 and a Kahan float32 sum reaches 5.2e-08 at 1024 chunks; `ImageStream` peaks at 80.1 MiB on a million-sample 32-channel panel against 335.7 MiB resident and costs more than resident on the short record; `fit_many` on 4 processes is 3.5x to 3.8x a serial loop and on 4 threads 1.11x slower; a cupy projection with the host transfer counted is slower than one BLAS thread at every width (1.14x to 1.95x) and 0.20x warm at the widest; a polynomial surrogate of any degree tried extrapolates 3.3x to 9431x worse than the structured fit while matching it in sample at 4 of 6 degrees | under a minute | none; the GPU columns need cupy |
| 23_stochastic_image | The second-order image of the stochastic tier: a record imaged whole and imaged in blocks then merged agree to floating-point noise once the scale budget is pinned, mean reversion and volatility persistence recover at parity with OLS AR(1) and the GARCH quasi-MLE while the two Hurst read-outs trail R/S and DFA, the regime router ties its classical twin at 96.4 percent over 56 cases, the forecast wins clearly on trend plus cycle, ties on mean reversion and loses on the AR(2) cycle and on GARCH(1,1), a finite-order AR process never routes as long memory, the streaming filter flags a persistence jump and a volatility switch and holds its state under a fixed cap, the generator round-trips every regime, and the cycle gate's floor is calibrated for white noise only | under a minute | none |
| 24_embedded_footprint | The streaming filter's deployable state: a fixed struct that does not grow with the stream, the struct model against the rig's flashed 152 bytes with the 32-byte difference named field by field, a step cost of the same order as a sliding-window refit's mean and a worst step 2 to 40 times shorter, Kalman-CA about 9x cheaper per axis, state linear in the window and faster than linear in the parameter count, and a three-axis tracker at W=15 under a third of a 2 KB part with a compute estimate that, scaled by its 3.5x error on the timed chip, clears a 10 Hz epoch on every 32-bit part and not on the 8-bit one | under 10 seconds | none |

### realdata/

| Notebook | Claim | Runtime | Data |
|---|---|---|---|
| 31_reference_datasets | Records with a documented right answer: on the ten NIST StRD nonlinear-regression sets from both NIST starts the Legendre image's error falls toward the certified parameters as its order rises wherever the record holds enough observations, the default order guards on the two sets where curve_fit diverges from the far start and a fixed order recovers both under 1 percent, the block image does not keep pace with Legendre on short non-uniform records, a fit on the first 65 percent forecasts the held-out tail within about a factor of two of NLLS for Legendre and not for block, the Puromycin replicate design lands both bases within 0.6 percent of NLLS, and three methods agree on the growth rate of two real series with no certified answer | under 10 seconds | none |
| 32_forecasting | Twelve series, eight measured and four generated from a known equation: fitting the declared structure beats the blind `auto_forecast` search by 5.9x to 25.6x on the four generated waveforms and sits close to it on the measured ones (wins four, ties two, loses two), a wrong structure is worse and flagged by its in-sample R2 on all three controls, random-walk persistence beats the declared structure on USD/UAH and the blind route on four series, an inline trend-plus-seasonal heuristic that calls nothing from dtfit is 3.2x to 5.0x off the published LTSF numbers and worse than repeating the last value, and on twelve hourly traffic sensors the image fit, its polished version and a pointwise fit land within 0.05 percent of each other | under a minute | none |
| 33_gps_simulation | A simulated maneuvering target tracked from GPS fixes and a 9-DOF IMU against a Kalman-CA and a coordinated-turn EKF: the full-IMU Legendre tracker smooths at or below the EKF and loses the 10-step forecast to it, gyro-aided trackers hold through a gap, the GPS-only Legendre tracker dead-reckons through it level with or below Kalman-CA and below its own extrapolated cubic, whole-track scores under scattered dropouts read parity at 10 percent and a loss at 20, reweighting rejects multipath spikes, the fused detector needs the gyro channel to lead, and the GPS-only Legendre tracker is at parity with Kalman-CA over 24 random flights | about two minutes | none |
| 34_gps_benchmark | Two literature trajectories with closed-form truth (a coordinated turn, a figure-8) over 30 seeds and two GSDC 2022 trips with RTK truth: through a GPS gap the IMM wins the turn and Kalman-CA the figure-8, with the dead-reckoned dtfit cubic at parity with Kalman-CA on the turn and ahead of its own extrapolated cubic on both, clean smoothing is near parity, a Huber-hardened Kalman leads the reweighted Legendre fit under a multipath glitch and the two are the only routes that recover the urban-canyon trip, and a clip-threshold sweep puts the reweighted route's window floor at twice the parameter count | about five minutes | GSDC trips, skipped per trip when absent |
| 35_archive_showcase | Two public archives reduced to images on a workstation and refitted on a Raspberry Pi 5 that never holds a raw sample: a fit from the stored image agrees with the raw least squares to 1e-8 on every row the gate passes (66708 NGL components, 11711 ISD station-years), NOAA's hourly archive stores at 180x and NGL's daily one at 1.18x with a fifth of its stations' images larger than their raw file, the ratio set by the order policy and falling below 1 above twenty years of span, resident memory flat from 200 to 2000 files, the Pi refits 12677 station-years from images in under five minutes, cupy loses to numpy at every archive width with the transfer counted, and the wire carries the image bit for bit | under 10 seconds; the rerun section needs the archives | tracked showcase tables; the NGL and ISD archives for the rerun, skipped when absent |
| 36_stochastic_real | The stochastic router on seven real records: every detected regime matches the literature's reading (GDP, two FX levels and the T-bill as random walks, CO2 as trend plus a seasonal cycle, sunspots as a cycle in the 8 to 14 year band, the Nile as trend), the held-out forecast never trails the random walk and beats it by more than half on CO2 and GDP while ETS, AR(1) and ARIMA each win a series outright, USD/UAH is a random-walk level with long memory in its absolute returns that four estimators agree on, the Nile's 19.7-year cycle is the 1899 level shift read through a line and the gate's significance test is what rejects it, and the Nile's memory reads H 0.81 to 0.88 on the train split | under 10 seconds | none; `statsmodels` supplies five of the seven series |
| rig (`packages/dtfit-hardware/experiments/rig.ipynb`) | Three recorded GPS+IMU logs and the filter frozen into C on a Nano 33 BLE Sense: the adaptive window beats the fixed one on both driving logs and loses the walk at the 10-step horizon and its two longer dropouts, CT-EKF and Kalman-CA win the 1 Hz drive and the shortest horizons elsewhere and lose the long horizons on the 5 Hz drive and the walk; with every method's configuration chosen on the other two logs the tracker goes from 12.33 m to 5.85 m at h=2 on the 1 Hz drive and still trails Kalman-CA (4.22 m) and CT-EKF (3.19 m) up to h=5, leads Kalman-CA at every horizon on the 5 Hz drive and from h=3 on the walk, and is level with CT-EKF at h=3 and ahead at h=10 on both; its forecast is not blind to the clock's zero, a shift of 10000 s taking the 5 Hz drive from 1.73 m to 8.91 m; carried in the time of its newest fix (`LocalTimeFilter`, in trial) it no longer sees the shift and goes to 4.14 m on the 1 Hz drive, level with Kalman-CA at h=2 and ahead from h=3, while it is 1 to 7 percent behind the absolute-clock tracker on the 5 Hz drive and a fifth behind it on the walk; a two-axis on-chip update averages 182 us in 304 bytes of state, and the chip's float32 track holds to 6.2 mm of a float64 replay over the whole drive | about eight minutes | recorded drives and the replay capture, private; those cells skip without them |

## Writing a notebook

1. Title, then the claims it establishes in plain words and what each would
   look like if it were false.
2. One setup cell: imports, seeds, sizes, and
   `QUICK = os.environ.get("DTFIT_QUICK") == "1"`. `QUICK` only shrinks
   counts (sample sizes, replicate counts, coefficient budgets); the
   committed notebook is always the full run, `DTFIT_QUICK` unset.
3. Sections: a question, the code that answers it (calling `dtfit` and
   `dtfit_experimental` directly, not through a wrapper that hides the
   experiment), the table or figure, the reading, then a check cell that
   asserts the ordering or bound the section establishes. A rerun that stops
   supporting the claim fails there instead of printing a different number.
4. A real-data section says where the data comes from and skips with a
   visible message when it is absent. Use `study.notebook.data_file(...)`
   and guard the cells that need it with `if PATH:` rather than letting a
   missing file raise: `nbclient` fails the whole notebook on an uncaught
   exception, and a skip is a value here, never an exception.
5. Tables that a paper or the dissertation restyles are also written to
   `experiments/results/<notebook>/<table>.csv` through
   `study.notebook.save_table`, small and tracked, so the figure builders in
   the thesis repository plot from the notebook's own numbers instead of
   recomputing them. This is an export of data, not a report.
6. A closing section: findings, the limits that were not tested, and a
   provenance line from `study.notebook.provenance(started)`.
7. Plain ASCII in every markdown cell and printed label; the two bases are
   called Legendre and block, never a retired name.
8. A saved output carries no path of the machine it ran on. A notebook whose
   fits or baselines warn calls `study.notebook.plain_warnings()` in its
   setup cell, which shows each distinct warning once as `Category: message`,
   without the source file and line.

Only reusable or heavy machinery moves to `study/`: baselines, simulators,
loaders, metrics, plot helpers, the Monte-Carlo scaffolding
(`study.montecarlo`) and the notebook helpers themselves
(`study.notebook`). A module earns its place there when two notebooks use
it, or it is an estimator, simulator or loader that deserves its own tests,
or it runs for hours over an archive. Everything else - the actual
construction of data, the calls into the method, the reading of a result -
stays in the notebook's own cells.

## What a notebook may import

At execution time, a notebook may import only what the CI research job
installs: `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, `statsmodels`,
`dtfit`, `dtfit_experimental`, `dtfit_legacy`. `numpy`, `scipy`, `sympy`
and `dtfit` are core dependencies of this package; `matplotlib`, `pandas`,
`statsmodels`, `dtfit-legacy` and the notebook harness (`nbclient`,
`ipykernel`) come with its `bench` extra, so
`pip install -e 'packages/dtfit-experimental[bench]'` is what a notebook
needs. Anything heavier -
`torch`, `cupy` - goes through `study.notebook.optional_import(name)`, and
the section that needs it skips the same way a missing data file does.
`dtfit_legacy` is imported only where a comparison of historical stages is
the point (the source forms, the families, the evolution matrix), never as
a stand-in for the current method.

## Figures

Inline only; there is no `figures/` directory here, the thesis repository
plots from the exported CSVs. `figure.dpi = 110` in the setup cell, at most
two panels to a figure, and a committed notebook stays under 1 MB.

## Running the suite

```bash
python -m dtfit_experimental.study.download_data      # fetch datasets once

# open and rerun interactively
jupyter lab experiments/method/13_two_bases.ipynb

# or execute headless, full run, outputs written in place
jupyter nbconvert --to notebook --execute --inplace experiments/method/13_two_bases.ipynb
```

A notebook is written and edited in Jupyter like any other. Before it is
committed it is rerun from a fresh kernel with `DTFIT_QUICK` unset -
Restart and Run All, or the `nbconvert` line above - so that the committed
file carries the outputs of the full run and the CSV exports on disk match
them.

`tests/test_notebooks.py` executes every notebook here under
`DTFIT_QUICK=1` with a per-cell and a whole-notebook time budget; that is
the check CI runs, never a substitute for opening the committed, full-run
notebook.
