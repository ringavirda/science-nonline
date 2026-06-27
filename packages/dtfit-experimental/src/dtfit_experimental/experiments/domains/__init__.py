"""Domain validation suite for the dtfit methods.

Where ``experiments/`` evaluates each EAC/LSI adaptation *in isolation* (one
optimization / structural idea per folder, scored on the promotion matrix), this
suite asks a different question: **for each real application domain, what is the
best combination of the methods, and does the merged pipeline actually work in a
realistic setting?**

Each domain is its own folder holding a ``backend.py`` (the single source of
truth for its simulation / estimation / data infra) and a Jupyter notebook that
imports it and produces the report (tables, figures, narrative) into
``figures/``:

* ``forecasting`` -- structured fit-then-extrapolate, merging the trend (LSI) and
  seasonal (Fourier-basis / boosting) levers into one auto-composed forecaster.
* ``parameter_estimation`` -- physical-parameter recovery, merging batch LSI/EAC,
  adaptive-window EAC (transients) and joint multi-channel fitting behind a
  scenario-aware selector, against the NLLS gold standard.
* ``big_data`` -- batch *and* streaming, merging the promoted map-reduce
  estimators (PartitionedLSI/EAC), the fused multi-channel ``PartitionedBatchLSI``
  and the whole-array GEMM projection behind a regime dispatcher.
* ``embedded_control`` -- real-time online identification, merging the streaming
  filters, the multi-stream ``FilterBank`` and the fused drift detector, with the
  deployable embedded-footprint accounting, against a Kalman baseline.
* ``realtime_gps`` -- streaming LSI/EAC (with external regressors) and a full-IMU
  strapdown fused inside the LSI filter, vs a constant-accel Kalman and a
  gyro-aided coordinated-turn EKF, on a simulated 9-DOF rig.
* ``stochastic_series`` -- can the deterministic fitters touch *random* series
  (economic / financial data)? Fits dtfit to the **deterministic functionals**
  of a stochastic process (autocovariance, spectrum, aggregated variance,
  trend/cycle) and scores which parameter-recovery routes are viable (Hurst /
  long memory, mean reversion, volatility persistence, stochastic cycle) against
  known-truth simulators.

Open and run a notebook directly (``jupyter lab forecasting/forecasting.ipynb``)
or headless via ``jupyter nbconvert --execute``. ``DOMAINS.md`` indexes them.
Reuses ``experiments/common`` (metrics, baselines, datasets, plotting) so the
backends are consistent with the case experiments in ``experiments/cases``.
"""
