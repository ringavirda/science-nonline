"""Fit diagnostics and visualization for dtfit.

These tools are **specific to evaluating a fitted dtfit model** -- they take a
:class:`dtfit.FittingResult` and report parameter uncertainty, information
criteria for model comparison, and residual-structure tests. For plain scalar
metrics on ``(y_true, y_pred)`` arrays use ``sklearn.metrics`` / ``scipy.stats``
directly; dtfit no longer ships a clone of those.

    from dtfit.diagnostics import fit_report, residual_diagnostics, FitDisplay

Visualization (``*Display``) requires matplotlib (``pip install 'dtfit[viz]'``).
"""

from ._report import fit_report, residual_diagnostics
from ._plot import FitDisplay, ResidualsDisplay

__all__ = [
    "fit_report",
    "residual_diagnostics",
    "FitDisplay",
    "ResidualsDisplay",
]
