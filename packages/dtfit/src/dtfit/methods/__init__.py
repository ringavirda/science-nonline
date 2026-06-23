"""dtfit fitting methods.

The differential-transformation batch fitters (``fit_lsi``, ``fit_eac`` -- with
uniform or curvature-adaptive window placement -- and ``fit_dsb``), the
``find_degree`` polynomial support primitive the DSB scheme builds on, and the
shared symbolic helpers (``model_params``, ``taylor_coeffs``).
"""

from ._common import model_params, taylor_coeffs, find_degree
from ._dsb import fit_dsb
from ._lsi import fit_lsi, fft_frequency_seed
from ._eac import fit_eac
from ._ensemble import ensemble_fit, EnsembleResult

__all__ = [
    "fit_lsi",
    "fit_eac",
    "fit_dsb",
    "ensemble_fit",
    "EnsembleResult",
    "fft_frequency_seed",
    "find_degree",
    "model_params",
    "taylor_coeffs",
]
