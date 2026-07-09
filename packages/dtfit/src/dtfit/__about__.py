"""Single source of truth for the package version.

Read by ``pyproject.toml`` (``[tool.setuptools.dynamic]``) at build time and
re-exported as ``dtfit.__version__`` at runtime, so the version is defined once.
"""

__version__ = "0.4.0"
