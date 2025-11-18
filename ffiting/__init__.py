"""Fallen Fitting Library

This package provides a variety of methods for fitting different functions and
models to data. The main focus here are nonline processes that require something
more than a regular polynomial. Library contains implementations of methods for
fitting nonline models developed using non-taylor transformations, which a part
of authors PHD dissertation. Use with care.

Interface:
    nonline_fit(): generic function that organizes lover level actions.
    Provides most customization in terms of fitting options.
    poly_fit(): performs classical polynomial fitting for the given data.
    ModelLite: a dataclass encapsulating original data, fitted model data,
    as well as a callable version of it.
    Model: comprehensive class that not only contains modelling data but
    also provides a variety of analytics options for the data.

Required packages:
    numpy, sympy, scipy
"""

# Main exports
from .framework import *
from .prefabs import Generators, Models
from .facade import poly_fit, nonline_fit
from .visualization import PlotRequest, multi_plot, sep_plot

__all__ = [
    "poly_fit",
    "nonline_fit",
    "Generators",
    "Models",
    "PlotRequest",
    "multi_plot",
    "sep_plot",
]
