"""dtfit-experimental -- experimental structural adaptations of EAC / LSI.

This distribution has **two tiers**, kept deliberately distinct:

1. **The library** -- *this* top-level package (``import dtfit_experimental``): a
   small, importable surface of experimental adaptations (``fit_lsi_basis``,
   ``fit_joint``, ``boosted_fit``) plus the backend helpers. This is the
   promotion staging area -- code that may graduate into stable ``dtfit``.
2. **The study** -- :mod:`dtfit_experimental.experiments`: the per-case /
   per-domain **validation suite** (notebooks + backends + the established
   baselines each method is measured against). It is a research/dev tree, *not*
   part of the library's API contract -- it is exempt from the mypy gate and
   ruff-relaxed, and is driven via ``python -m dtfit_experimental.experiments...``
   rather than imported as a library. Nothing in tier 1 imports from it.

The rest of this docstring describes tier 1 (the adaptations).

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
* #6 adaptive-window EAC, now folded into stable ``dtfit.fit_eac`` as
  ``window_mode="curvature"``;
* the LSI **oscillatory recipe** is now built into ``dtfit.fit_lsi`` via
  ``oscillatory=True`` / ``freq_param=`` (plus ``dtfit.fft_frequency_seed``);
* the fused multi-axis ``FusedChiSquareDetector`` for a streaming ``FilterBank``;
* #3 the overlapping-window **ensemble** ``ensemble_fit`` / ``EnsembleResult``
  (``from dtfit import ensemble_fit``) -- robust to outliers on contaminated data;
* the **stochastic-series** solution -- fit dtfit's deterministic fitters to the
  deterministic functionals of a random process to characterize / forecast /
  generate it (``dtfit.stochastic.fit_stochastic`` / ``StochasticModel`` /
  ``dtfit.Stochastic``) and track it online (``dtfit.stochastic.StochasticFilter``).

The three adaptations below remain here on purpose. #2 ``fit_lsi_basis`` was
evaluated as a *vocabulary* (it makes periodic/decay models expressible) but **not
a source of predictive power** -- it did not improve accuracy and lost on the LTSF
benchmark, so it stays experimental rather than being folded into ``fit_lsi``. #5
``boosted_fit`` is a genuine win but on **one domain only** (additive trend+season,
e.g. CO2); it needs a confirming second domain to clear the ``>=2 domains`` gate.
#4 ``fit_joint`` is the substantial new solver still under evaluation. See the
``experiments/cases/analysis`` notes for the measured verdicts.

The shared spectral/backend machinery (``dtfit._core._spectral`` / ``dtfit._core._backend``)
moved to stable with the map-reduce estimators and is reused by the adaptations
that remain here.
"""

from dtfit._core._backend import available_backends, resolve_backend, Backend
from .basis_lsi import fit_lsi_basis
from .joint import fit_joint, JointResult
from .boosting import boosted_fit, BoostedModel

# The stochastic-series adaptations were promoted into stable ``dtfit`` -- they now
# live physically there (``from dtfit.stochastic import fit_stochastic,
# StochasticModel, StochasticFilter, ...`` / ``from dtfit import Stochastic``) and
# are no longer re-exported from here. The domain harness consumes them from dtfit.

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
