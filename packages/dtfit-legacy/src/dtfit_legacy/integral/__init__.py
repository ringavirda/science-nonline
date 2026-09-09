"""The integral criteria before the image: ``fit_lsi`` on the Legendre
basis with BIC order selection and optional pointwise refinement, and
``fit_eac`` with direct numerical areas, adaptive windows and the soft-L1
loss. Restored from dtfit at commit 17f43aa (2026-09-06)."""

from ._common import model_params, taylor_coeffs
from ._modelinput import ModelSpec, resolve_model, result_kwargs
from ._lsi import fft_frequency_seed, fit_lsi
from ._eac import fit_eac

__all__ = [
    "fit_lsi", "fit_eac", "fft_frequency_seed", "model_params",
    "taylor_coeffs", "resolve_model", "ModelSpec", "result_kwargs",
]
