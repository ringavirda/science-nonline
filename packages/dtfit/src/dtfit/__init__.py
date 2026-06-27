"""dtfit -- differential-transformation fitting.

Methods for fitting models that are nonlinear in their parameters
(exponential, transcendental, mixed) for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations. Developed as
part of the author's PhD dissertation.

The public interface is layered -- pick the tier that matches how much you want
to drive:

  1. Methods (you choose the engine):
       fit_lsi() / fit_eac() / fit_dsb(): the batch fitters. ``fit_eac`` places
           windows uniformly or curvature-adaptively (``window_mode=``) and has a
           robust ``loss=`` for outlier-prone windows.
       ensemble_fit(): overlapping-window robust ensemble -- the heavier-duty
           complement to fit_eac's robust loss for densely contaminated data.
       find_degree(): polynomial-degree selection (DSB support).
  2. Estimator (sklearn-compatible):
       NonlineRegressor: fit/predict/score over LSI/EAC/DSB; composes with
           sklearn Pipeline and GridSearchCV.
  3. High-level (it chooses for you):
       auto_estimate() / auto_forecast(): shape-routed parameter estimation and
           structured forecasting -- the function-style counterparts for callers
           who do not want the model framework.
       models / Model / suggest_models(): a catalog of self-seeding model
           families and an AIC recommender (pick structure, not sympy strings).

Streaming (online, partial_fit):
    EACFilter / LSIFilter: real-time parameter trackers (streaming twins of
        fit_eac / fit_lsi); start from the ``.tracking()`` / ``.robust()``
        presets. FilterBank / FusedChiSquareDetector drive many streams at once.

Scale (same methods, run big):
    PartitionedLSI / PartitionedEAC / PartitionedBatchLSI (one-pass / distributed
        map-reduce), fit_lsi_batched (GEMM-batched multi-channel), fit_many
        (process/thread fan-out). The low-level project_spectra primitive lives
        under ``dtfit.scale``.

Result type:
    FittingResult: the self-describing fitted-model result (also returned by
        fit_many; BatchFittingResult is a back-compat alias of it).
    enable_logging() / logger: opt-in library logging.

Submodules (imported explicitly, kept out of the top-level namespace, after
the scikit-learn convention):
    dtfit.diagnostics: fit-aware diagnostics (fit_report, residual tests) and
        ``*Display`` visualization helpers.

Core dependencies: numpy, scipy, sympy, scikit-learn.
Optional extras: matplotlib (install with `pip install 'dtfit[viz]'`).
"""

from dtfit.__about__ import __version__
from dtfit import diagnostics
from dtfit.types import FittingResult
from dtfit.log import enable_logging, logger
from dtfit.methods import (
    fit_lsi,
    fft_frequency_seed,
    fit_eac,
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
# ``project_spectra`` is the low-level empirical-spectrum primitive behind
# ``fit_lsi_batched``; it stays reachable as ``dtfit.scale.project_spectra`` but
# is kept off the headline top-level namespace.
from dtfit.scale._batched import fit_lsi_batched
# High-level "just fit it" entry points distilled from the domain merged
# pipelines (shape-routed estimation; structured fit-then-extrapolate forecast).
from dtfit.auto import auto_estimate, auto_forecast
# Model framework: a catalog of self-seeding model families + a recommender,
# so users pick structure (not sympy strings) and can infer the right model.
from dtfit import models
from dtfit.models import Model, Stochastic, suggest_models
# Stochastic-series: fit the deterministic functionals of a random process to
# characterize / forecast / generate it (fit_stochastic) and track it online
# (StochasticFilter); the Stochastic model wraps it in the .fit() convention.
from dtfit import stochastic
from dtfit.stochastic import fit_stochastic, StochasticModel, StochasticFilter

__all__ = [
    "NonlineRegressor",
    "auto_estimate",
    "auto_forecast",
    "models",
    "Model",
    "suggest_models",
    "stochastic",
    "Stochastic",
    "fit_stochastic",
    "StochasticModel",
    "StochasticFilter",
    "EACFilter",
    "LSIFilter",
    "FilterBank",
    "FusedChiSquareDetector",
    "PartitionedLSI",
    "PartitionedEAC",
    "PartitionedBatchLSI",
    "fit_lsi_batched",
    "fit_lsi",
    "fft_frequency_seed",
    "fit_eac",
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
    "__version__",
]
