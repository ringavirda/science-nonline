"""Internal package with all the stuff necessary for fitting. It is separated
from everything else to allow better focus. Here are the quick descriptions of
some classes from the framework:

Model:
    Generic interface to train, predict values. Can be changed and fitted.

ModelLite:
    DataClass with trained lambda some values and coeffs.

Spectrum:
    Dedicated interface for symbolic operations to deal with differential discretes.

PolySpectrum:
    Specific case for automatic generation of expression.

Metrics:
    Contains most useful methods to calculate statistical values for models.
"""

from .options import FittingModes, FittingOptions
from .models import Model, ModelLite
from .utils import (
    ModelMetrics,
    Metrics,
    get_metrics,
    get_metrics_m,
    scale_data,
    collapse,
    apply_noise,
    NoiseConfig,
)
from .discretes import Spectrum, PolySpectrum, Discretes
from .extra import (
    nonline_fit_,
    poly_fit_,
    poly_fit_lite,
    find_poly_rank,
    nonline_lsm,
    numeric_optimize,
)

__all__ = [
    "Model",
    "ModelLite",
    "ModelMetrics",
    "FittingOptions",
    "FittingModes",
    "Spectrum",
    "PolySpectrum",
    "Discretes",
    "nonline_fit_",
    "poly_fit_",
    "poly_fit_lite",
    "find_poly_rank",
    "nonline_lsm",
    "numeric_optimize",
    "Metrics",
    "get_metrics",
    "get_metrics_m",
    "scale_data",
    "collapse",
    "apply_noise",
    "NoiseConfig",
]
