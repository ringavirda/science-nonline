"""Shared utilities for the dtfit experiment suite.

The experiments are now self-contained Jupyter notebooks, each over a sibling
``backend.py`` (the single source of truth for its simulation/estimation infra).
This package keeps the **pure-compute** helpers the backends share — metrics,
baselines, datasets — plus a few plotting helpers the *notebooks* import (e.g.
``common.plotting.fit_overlay``), and the ``EXPERIMENTS_DIR`` anchor used to
locate bundled data. The old report-writer / parallel-runner harness was removed
when the notebooks replaced ``run.py``.
"""

from pathlib import Path

from .metrics import metrics, mse, mae, timed, md_table, fmt
from . import plotting
from . import baselines
from . import datasets

# Repository anchor: ``.../dtfit_experimental/experiments`` (parent of ``common``),
# under which bundled real-data CSVs live in ``data/``.
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent

__all__ = [
    "metrics", "mse", "mae", "timed", "md_table", "fmt",
    "EXPERIMENTS_DIR",
    "plotting", "baselines", "datasets",
]
