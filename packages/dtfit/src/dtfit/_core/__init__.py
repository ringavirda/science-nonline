"""dtfit._core -- private numeric core (not part of the public API).

The compute layer the higher-level methods are built on, kept together rather
than scattered across the package root:

- :mod:`._backend`: the array-backend registry (NumPy / optional CuPy).
- :mod:`._kernels`: thin wrappers over the optional compiled C kernels
  (:mod:`dtfit._core._native`), with pure SciPy/NumPy fallbacks.
- :mod:`._spectral`: orthogonal-basis construction and spectral NLLS solving.

The compiled extension ``dtfit._core._native`` (built by ``build_native.py``)
lives here too, alongside the ``_kernels`` wrapper that loads it.

Nothing here is re-exported from the top-level ``dtfit`` namespace; import
paths under ``dtfit._core`` are internal and may change without notice.
"""
