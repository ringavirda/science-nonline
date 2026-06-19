"""dtfit experiment suite.

Each subpackage (``control_systems``, ``big_data_streaming``,
``noise_robustness``, ``realworld_forecasting``, ``gps_trajectory``,
``benchmark_ltsf``, ``parallel_scaling``) is a self-contained experiment with a
``run.py`` module, a generated ``report.md`` and a ``figures/`` directory. Run
one with ``python -m experiments.<name>.run`` or all via
``python -m dtfit_experimental.experiments.cases.run_suite``.
"""
