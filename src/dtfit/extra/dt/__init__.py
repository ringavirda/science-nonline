"""Differential-transformation fitting methods.

- ``fit_dsb``: differential spectra balance (symbolic, analytical reference).
  Now built on generic Maclaurin coefficients (:mod:`dtfit.extra.dt.taylor`),
  so it supports any differentiable model expression.
- ``fit_lsi``: least-squares integral (numeric; successor to DSBI).
- ``fit_eda``: equal differential areas / equal areas (numeric; successor to
  DSBE).
"""

from .taylor import model_params, taylor_coeffs
from .solve import solve_nonline, solve_numeric
from .dsb import fit_dsb
from .lsi import fit_lsi
from .eda import fit_eda

__all__ = [
    "model_params",
    "taylor_coeffs",
    "fit_dsb",
    "fit_lsi",
    "fit_eda",
    "solve_nonline",
    "solve_numeric",
]
