"""Optional compiled-kernel build for dtfit.

All package metadata lives in ``pyproject.toml``; this file exists only to declare
the **optional C extension** that the declarative ``[tool.setuptools]`` table
cannot express. The extension (``dtfit._core._native`` -- the compiled Simpson /
Legendre kernels) is marked ``optional=True``, so ``pip install dtfit`` *attempts*
to build the fast native kernels when a C compiler and the NumPy headers are
present and **silently falls back** to the pure NumPy/SciPy path
(``dtfit._core._kernels``) when they are not -- the package always installs, just
slower without the kernels.

``build_native.py`` remains as a standalone developer builder (incremental clang
rebuilds without a full reinstall, plus the VS Code IntelliSense config); the
extension declared here is what makes ``pip install`` build the kernels
automatically rather than requiring that manual step.
"""

from setuptools import Extension, setup

try:  # NumPy headers are needed to compile against the C-API
    import numpy

    _include_dirs = [numpy.get_include()]
except Exception:  # NumPy absent at build time -> the optional ext simply skips
    _include_dirs = []

setup(
    ext_modules=[
        Extension(
            name="dtfit._core._native",
            sources=["src/dtfit/_core/_native.c"],
            include_dirs=_include_dirs,
            optional=True,  # a missing compiler / headers is non-fatal
        )
    ]
)
