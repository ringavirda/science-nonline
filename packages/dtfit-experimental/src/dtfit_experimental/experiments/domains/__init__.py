"""Domain validation suite for the dtfit methods.

Where ``experiments/`` evaluates each EAC/LSI adaptation *in isolation* (one
optimization / structural idea per folder, scored on the promotion matrix), this
suite asks a different question: **for each real application domain, what is the
best combination of the methods, and does the merged pipeline actually work in a
realistic setting?**

Four domains, each its own folder with a ``run.py`` (-> ``report.md`` +
``figures/``):

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

Driven by ``python -m experiments.domains.run_domains [--quick]`` which also
regenerates ``experiments/domains/DOMAINS.md``. Reuses ``experiments/common``
(ReportWriter, metrics, baselines, plotting) so the reports are consistent with
the case experiments in ``experiments/cases``.
"""
