"""Batch-scale execution backends for the dtfit methods.

The fitting math lives in :mod:`dtfit.methods`; this package holds the
alternative *execution backends* that run those methods at scale, all built on
the same additive integral projection:

- :mod:`dtfit.scale._partitioned` -- exact one-pass / distributed map-reduce
  estimators (:class:`PartitionedLSI`, :class:`PartitionedEAC`) and the fused
  multi-channel :class:`PartitionedBatchLSI`.
- :mod:`dtfit.scale._batched` -- the GEMM-batched, backend-pluggable projection
  (:func:`fit_lsi_batched`, :func:`project_spectra`).
- :mod:`dtfit.scale._parallel` -- process/thread fan-out of independent fits
  (:func:`fit_many`).

The online (streaming) counterparts live separately in :mod:`dtfit.streaming`.
"""

from ._partitioned import PartitionedLSI, PartitionedEAC, PartitionedBatchLSI
from ._batched import fit_lsi_batched, project_spectra
from ._parallel import fit_many, FittingProblem

__all__ = [
    "PartitionedLSI",
    "PartitionedEAC",
    "PartitionedBatchLSI",
    "fit_lsi_batched",
    "project_spectra",
    "fit_many",
    "FittingProblem",
]
