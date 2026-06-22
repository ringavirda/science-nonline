"""dtfit-experimental -- experimental structural adaptations of EAC / LSI.

These are new ways to *compose* the differential-transformation fitting methods
of :mod:`dtfit`, grounded in the methods' own math (linearity of integration;
orthogonal-basis projection; additive areas). They are prototyped here, evaluated
across the experiment suite (:mod:`dtfit_experimental.experiments`), and the ones
that prove effective across a range of applications are **promoted into the stable
``dtfit`` API** -- where they then live physically, not re-imported from here.

    from dtfit_experimental import (
        fit_lsi_basis,        # #2 pluggable orthogonal basis (Fourier/...)
        fit_joint,            # #4 joint shared-parameter multi-channel fit
        boosted_fit,          # #5 stage-wise residual boosting
    )

These APIs are experimental and may change until promoted. Several adaptations
have already cleared the promotion gate and now live in :mod:`dtfit`:

* #1 ``PartitionedLSI`` / ``PartitionedEAC`` -- one-pass / distributed map-reduce
  (``from dtfit import PartitionedLSI, PartitionedEAC``);
* the GEMM-batched projection ``fit_lsi_batched`` / ``project_spectra`` and the
  fused multi-channel ``PartitionedBatchLSI`` (``from dtfit import ...``);
* #6 adaptive-window EAC ``fit_eac_adaptive`` (``from dtfit import fit_eac_adaptive``);
* the LSI **oscillatory recipe** is now built into ``dtfit.fit_lsi`` via
  ``oscillatory=True`` / ``freq_param=`` (plus ``dtfit.fft_frequency_seed``);
* the fused multi-axis ``FusedChiSquareDetector`` for a streaming ``FilterBank``;
* #3 the overlapping-window **ensemble** ``ensemble_fit`` / ``EnsembleResult``
  (``from dtfit import ensemble_fit``) -- robust to outliers on contaminated data.

The shared spectral/backend machinery (``dtfit._core._spectral`` / ``dtfit._core._backend``)
moved to stable with the map-reduce estimators and is reused by the adaptations
that remain here.
"""

from dtfit._core._backend import available_backends, resolve_backend, Backend
from .basis_lsi import fit_lsi_basis
from .joint import fit_joint, JointResult
from .boosting import boosted_fit, BoostedModel

__all__ = [
    "fit_lsi_basis",
    "available_backends",
    "resolve_backend",
    "Backend",
    "fit_joint",
    "JointResult",
    "boosted_fit",
    "BoostedModel",
]
