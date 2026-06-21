# The baselines -- what they are and why they were chosen

To claim a method is good you must compare it against what people **actually use**
today. The experiment suite therefore measures `dtfit` against a broad set of
*established* baselines -- the methods a working practitioner in each domain would
reach for. This page explains each one in plain terms: **what it does**, **how it
works**, and **why it earns a place in the comparison**.

Source: [`experiments/common/baselines.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/common/baselines.py).
Deep-learning and statsmodels backends are imported lazily and skipped (not
faked) when not installed, so the suite still runs on a core install.

A guiding principle runs through the selection: each baseline is the **fair,
same-job counterpart** to a dtfit method. A black-box learner that fits a curve
but recovers *no physical parameters* is the right foil for showing dtfit
recovers parameters; a robust NLLS is the right foil for dtfit's outlier
robustness; an EKF is the right foil for the streaming filters; and so on.

---

## 1. Classical curve fitting (the parameter-estimation foils)

### SciPy `curve_fit` -- Levenberg-Marquardt nonlinear least squares
**What:** the standard tool for fitting a known nonlinear model. **How:** guesses
parameters, measures the point-by-point squared error, and iteratively nudges the
parameters downhill (Levenberg-Marquardt / trust-region). **Why chosen:** it is
the **gold standard** for parameter recovery -- the number every other estimator
is judged against. dtfit's goal is to *match* it on clean data and *beat* it under
noise/outliers/streaming while being more robust. It is the most important
baseline in the suite.

### Robust NLLS (`least_squares` with a robust loss)
**What:** `curve_fit`'s outlier-resistant sibling. **How:** the same nonlinear
least squares, but large residuals are **down-weighted** by a robust loss
(soft-L1 / Huber / Cauchy) so a few wild points don't dominate. **Why chosen:** it
is the *established* way to fit a known model when outliers are present -- the
direct, fair comparison for EDA's robust-loss and the ensemble adaptation.

### `numpy.polyfit` -- polynomial least squares
**What:** fit a plain polynomial. **How:** linear least squares on powers of `x`.
**Why chosen:** the simplest *linear-in-parameters surrogate*. It can trace a
curve but returns **opaque polynomial coefficients**, not the physical parameters
-- so it illustrates exactly what dtfit provides that a generic curve fit doesn't.

### Gaussian-process regression (`gp_curve`)
**What:** a flexible nonparametric Bayesian smoother. **How:** models the curve as
a draw from a distribution over smooth functions and conditions on the data,
giving a fitted curve *and* uncertainty. **Why chosen:** the standard
**nonparametric** smoother -- fits any smooth shape beautifully but recovers **no
physical parameters**. It is the nonparametric counterpart to dtfit's *structured*
fit: a fair way to ask "is structure worth it?"

### MLP curve fit (`mlp_curve`)
**What:** a small neural network used as a black-box `f(x)>y`. **How:** a
multilayer perceptron trained to map input to output. **Why chosen:** the
black-box-learner foil -- high flexibility, zero interpretability -- to contrast
with dtfit recovering the actual model parameters.

---

## 2. Forecasting baselines (the time-series toolkit)

These train on a series and predict `horizon` steps ahead. The set spans the
*entire* spread a forecasting practitioner uses, from trivial benchmarks to
competition winners and deep nets -- so a win is meaningful.

### Naive random walk (`random_walk_forecast`)
**What:** "tomorrow = today." **How:** persist the last observed value. **Why
chosen:** the famously **hard-to-beat benchmark**, especially for financial
series. The suite reports honestly that on near-random-walk data *nothing*
(including dtfit) beats it one step out -- using it keeps the claims grounded.

### Seasonal-naive (`seasonal_naive_forecast`)
**What:** "this season repeats." **How:** copy the values from one period ago.
**Why chosen:** the standard **seasonal** benchmark -- the minimum bar for any
method claiming to handle seasonality.

### Drift method (`drift_forecast`)
**What:** random walk **with** a trend. **How:** extrapolate the average per-step
change (Hyndman's drift method). **Why chosen:** the standard trend benchmark; a
fair, trivial competitor for dtfit's linear/structured trend forecasts.

### Polynomial extrapolation (`poly_extrap_forecast`)
**What:** fit a global polynomial and continue it. **How:** `polyfit` then evaluate
past the end. **Why chosen:** the **surrogate-fit** baseline -- extrapolates by raw
curvature with no parametric structure, so it shows the value of dtfit's
structured extrapolation (and the danger of unstructured extrapolation).

### Holt-Winters / ETS (`ets_forecast`)
**What:** exponential smoothing with level + optional trend + season. **How:**
recursively updates a smoothed level/trend/seasonal state (statsmodels
`ExponentialSmoothing`). **Why chosen:** the **workhorse** classical forecaster --
ubiquitous in practice, the realistic bar to clear.

### Theta method (`theta_forecast`)
**What:** a robust decomposition forecaster. **How:** combines a long-run trend
line with a short-run smoothed component (statsmodels `ThetaModel`). **Why
chosen:** the **M3-competition winner** -- a famously strong, simple method, so
beating it is a strong result.

### (S)ARIMA (`arima_forecast` / `sarima_forecast`)
**What:** the classic statistical time-series model. **How:** models the series as
auto-regressive + moving-average terms on differenced (and optionally seasonal)
data. **Why chosen:** the **standard statistical** forecaster taught and deployed
everywhere; the canonical comparison for any new forecaster.

### MLP and LSTM forecasters (`mlp_forecast`, `torch_mlp_forecast`, `lstm_forecast`)
**What:** neural-network sequence forecasters (a feed-forward net on a lookback
window, and a recurrent LSTM). **How:** trained on sliding windows to predict the
next value, then rolled forward recursively. **Why chosen:** the **modern
machine-learning** foils. The LSTM in particular represents deep sequence models.
(Cutting-edge deep forecasters -- DLinear/TimesNet/Time-LLM -- are **not**
re-implemented; instead the suite compares against their *published* benchmark
numbers in `experiments/06_benchmark_ltsf`, which is the fair way to cite them.)

---

## 3. Online / streaming baselines (the real-time foils)

These ingest one sample at a time -- the fair competitors for `EDAFilter` /
`LSIFilter`, which also run online.

### Extended Kalman Filter for parameters (`EKFParam`)
**What:** the textbook method for **online nonlinear parameter estimation**.
**How:** treats the parameters as a slowly-drifting state, takes each sample's
*pointwise* value as the measurement, and linearizes the model about the current
estimate via its parameter-Jacobian (compiled once with SymPy). **Why chosen:**
the **same-job** baseline for the dtfit filters -- both track a known model's
parameters online; the *only* difference is the measurement (a pointwise value
here vs an integrated **area/spectrum** for dtfit). That makes it the cleanest
possible test of dtfit's "integrate, don't sample" thesis in the streaming
setting.

### Recursive Least Squares (`RLSPredictor`)
**What:** the classic online adaptive filter. **How:** tracks the coefficients of
a linear auto-regressive predictor online with a forgetting factor so it adapts to
drift. **Why chosen:** the standard online **system-identification / adaptive-
filtering** algorithm -- but a *black box* that yields one-step predictions and **no
physical parameters**. The streaming counterpart to the MLP foil.

### Constant-acceleration Kalman filter (`KalmanCA`)
**What:** the trajectory-tracking gold standard. **How:** a per-axis Kalman filter
with a position/velocity/acceleration state and position measurements. **Why
chosen:** the established method for **tracking and short-horizon trajectory
prediction** -- the right baseline for the GPS-trajectory and embedded-control
domains. It is wired to expose the *same* self-calibrating re-arming hook
(`inflate`) and innovation statistics as the dtfit filters, so the maneuver-
detection comparison is driven by identical machinery -- a fair fight.

### Incremental MLP (`mlp_forecast(..., incremental=True)`)
**What:** a neural net trained by `partial_fit` over mini-batches. **How:** the
sklearn MLP updated incrementally rather than in one batch. **Why chosen:** the
**streaming-friendly neural** baseline used by the big-data domain -- the online
black-box learner to contrast with dtfit's online structured filter.

---

## Why this particular set

Three deliberate choices shape the list:

1. **Same-job pairing.** Every dtfit method has a matched, established competitor
   doing the *same* task with the *same* information: gold-standard NLLS for batch
   recovery, robust NLLS for outliers, GP/MLP for "structure vs black box," EKF/
   Kalman/RLS for streaming. This isolates *what dtfit changes* (the integral
   measurement and the fingerprint formulation) rather than confounding it with a
   different problem setup.
2. **Real runnable code, honestly skipped.** Baselines are actually executed on the
   same data, not quoted from memory -- except modern deep LTSF models, which are
   compared via their *published* numbers (re-implementing them faithfully would be
   its own research project, so citing their benchmark is the fair move).
3. **Including the ones dtfit can't beat.** The random walk on FX, the Lorentzian
   for NLLS -- these stay in precisely so the comparison is honest. A method that
   only ever shows its wins isn't validated; it's advertised.

See the per-domain reports (`domains/*/report.md`) for the full tables, and
[README.md](Experimental) for the suite overview.
