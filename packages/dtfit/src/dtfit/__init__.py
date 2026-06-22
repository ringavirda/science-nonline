"""dtfit -- differential-transformation fitting.

Methods for fitting models that are nonlinear in their parameters
(exponential, transcendental, mixed) for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations. Developed as
part of the author's PhD dissertation.

Public interface:
    NonlineRegressor: scikit-learn compatible estimator (fit/predict/score)
        wrapping the LSI/EAC/DSB methods; composes with sklearn Pipeline and
        GridSearchCV.
    EACFilter / LSIFilter: online/streaming estimators (partial_fit) for
        real-time parameter tracking (streaming twins of fit_eac / fit_lsi).
    fit_lsi() / fit_eac() / fit_dsb(): the individual batch methods.
    ensemble_fit(): overlapping-window robust ensemble (outlier-prone data).
    find_degree(): polynomial-degree selection (DSB support).
    FittingResult: the self-describing fitted-model result type.
    enable_logging() / logger: opt-in library logging.

Submodules (imported explicitly, kept out of the top-level namespace, after
the scikit-learn convention):
    dtfit.diagnostics: fit-aware diagnostics (fit_report, residual tests) and
        ``*Display`` visualization helpers.

Core dependencies: numpy, scipy, sympy, scikit-learn.
Optional extras: matplotlib (install with `pip install 'dtfit[viz]'`).
"""

from dtfit import diagnostics
from dtfit.types import FittingResult
from dtfit.log import enable_logging, logger
from dtfit.methods import (
    fit_lsi,
    fft_frequency_seed,
    fit_eac,
    fit_eac_adaptive,
    fit_dsb,
    ensemble_fit,
    EnsembleResult,
    find_degree,
)
from dtfit.estimators import NonlineRegressor
from dtfit.streaming import (
    EACFilter,
    LSIFilter,
    FilterBank,
    FusedChiSquareDetector,
)
from dtfit.scale._parallel import fit_many, FittingProblem, BatchFittingResult
# Promoted after the experiment suite validated them across the big-data and
# parallel workloads: the exact one-pass / distributed (map-reduce) estimators
# and the GEMM-batched multi-channel projection. The remaining adaptations stay
# experimental in the separate `dtfit-experimental` package.
from dtfit.scale._partitioned import PartitionedLSI, PartitionedEAC, PartitionedBatchLSI
from dtfit.scale._batched import fit_lsi_batched, project_spectra
# High-level "just fit it" entry points distilled from the domain merged
# pipelines (shape-routed estimation; structured fit-then-extrapolate forecast).
from dtfit.auto import auto_estimate, auto_forecast
# Model framework: a catalog of self-seeding model families + a recommender,
# so users pick structure (not sympy strings) and can infer the right model.
from dtfit import models
from dtfit.models import Model, suggest_models

__all__ = [
    "NonlineRegressor",
    "auto_estimate",
    "auto_forecast",
    "models",
    "Model",
    "suggest_models",
    "EACFilter",
    "LSIFilter",
    "FilterBank",
    "FusedChiSquareDetector",
    "PartitionedLSI",
    "PartitionedEAC",
    "PartitionedBatchLSI",
    "fit_lsi_batched",
    "project_spectra",
    "fit_lsi",
    "fft_frequency_seed",
    "fit_eac",
    "fit_eac_adaptive",
    "fit_dsb",
    "ensemble_fit",
    "EnsembleResult",
    "find_degree",
    "fit_many",
    "FittingProblem",
    "BatchFittingResult",
    "FittingResult",
    "enable_logging",
    "logger",
    "diagnostics",
]
