"""Default locations of the showcase's inputs and outputs.

Every function in the domain takes its paths as arguments; these are the
defaults only, read from the environment so that no machine's directory
layout is written into the package. ``data_root``, ``ngl_dir``, ``isd_dir``
and ``normals_dir`` are forwards to ``dtfit_experimental.study.paths``,
which owns the archive locations; ``images_dir``, ``domain_dir``,
``results_dir`` and ``figures_dir`` are repository-relative and stay here.
"""

from __future__ import annotations

from pathlib import Path

from dtfit_experimental.study.paths import isd_dir, normals_dir, ngl_dir
from dtfit_experimental.study.paths import showcase_root as data_root

__all__ = [
    "data_root", "ngl_dir", "isd_dir", "normals_dir",
    "images_dir", "domain_dir", "results_dir", "figures_dir",
]


def images_dir(root: Path | None = None) -> Path:
    """Where the reduced images are written: ``<root>/images``, with
    ``ngl/`` and ``isd/<year>/`` beneath it."""
    return (data_root() if root is None else Path(root)) / "images"


def domain_dir() -> Path:
    """This package's directory inside the repository."""
    return Path(__file__).resolve().parent


def results_dir() -> Path:
    """The tracked CSV output directory, created if missing."""
    p = domain_dir() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    """The tracked figure output directory, created if missing."""
    p = domain_dir() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
