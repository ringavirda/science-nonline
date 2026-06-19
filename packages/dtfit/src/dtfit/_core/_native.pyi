"""Type stub for the optional compiled extension ``dtfit._native``.

The kernels are implemented in ``_native.c`` and built by ``build_native.py``;
this stub lets type checkers resolve the symbols even when the binary is not
present. See ``dtfit/_kernels.py`` for the runtime wrappers and fallbacks.
"""

import numpy as np

def simpson_windows(
    y: np.ndarray, x: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray: ...
def simpson_windows_rows(
    Y: np.ndarray, x: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray: ...
def legendre_project(
    fv: np.ndarray, qw: np.ndarray, legvander: np.ndarray, norm: np.ndarray
) -> np.ndarray: ...
