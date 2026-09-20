"""Default locations of the showcase's inputs and outputs.

Every function in the domain takes its paths as arguments; these are the
defaults only, read from the environment so that no machine's directory
layout is written into the package. ``data_root``, ``ngl_dir``, ``isd_dir``
and ``normals_dir`` live in ``dtfit_experimental.study.paths``, which owns
the archive locations; ``images_dir`` and ``results_dir`` stay here.
"""

from __future__ import annotations

from pathlib import Path

from dtfit_experimental.study.paths import experiments_dir, isd_dir, ngl_dir, normals_dir
from dtfit_experimental.study.paths import showcase_root as data_root

__all__ = [
    "data_root", "ngl_dir", "isd_dir", "normals_dir",
    "images_dir", "results_dir",
]


def images_dir(root: Path | None = None) -> Path:
    """Where the reduced images are written: ``<root>/images``, with
    ``ngl/`` and ``isd/<year>/`` beneath it."""
    return (data_root() if root is None else Path(root)) / "images"


def results_dir() -> Path:
    """The tracked CSV output directory, created if missing.

    Returns ``<experiments>/results/35_archive_showcase``. Raises
    ``FileNotFoundError`` naming the path when the ``experiments/`` tree does
    not exist, which is the case in an installed wheel: the showcase CLI
    needs a source checkout to write its tables.
    """
    p = experiments_dir() / "results" / "35_archive_showcase"
    p.mkdir(parents=True, exist_ok=True)
    return p
