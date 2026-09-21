"""Filesystem anchors the study tier's notebooks and downloaded data use."""

from __future__ import annotations

import os
from pathlib import Path

import dtfit_experimental


def experiments_dir() -> Path:
    """Root of the ``experiments/`` tree: notebooks, their results and data.

    Returns ``packages/dtfit-experimental/experiments``, beside the package's
    source rather than inside it. Raises ``FileNotFoundError`` naming the path
    when that directory does not exist; an installed wheel carries no
    experiments tree.
    """
    root = Path(dtfit_experimental.__file__).resolve().parents[2] / "experiments"
    if not root.is_dir():
        raise FileNotFoundError(f"no experiments tree at {root}")
    return root


def data_dir() -> Path:
    """Directory holding the bundled and downloaded CSV datasets.

    ``$DTFIT_DATA`` when set, else ``experiments/data`` beside the package
    source. Unlike :func:`experiments_dir`, does not require the directory
    to exist: ``study.download_data`` creates it on first use.
    """
    env = os.environ.get("DTFIT_DATA")
    if env:
        return Path(env)
    return Path(dtfit_experimental.__file__).resolve().parents[2] / "experiments" / "data"


DEFAULT_SHOWCASE = "~/data/showcase"


def showcase_root() -> Path:
    """Root of the showcase archives.

    Returns:
        ``$SHOWCASE_DATA`` when set, else ``~/data/showcase``, with ``~``
        expanded. The directory is neither created nor required to exist.
    """
    return Path(os.environ.get("SHOWCASE_DATA", DEFAULT_SHOWCASE)).expanduser()


def ngl_dir(root: Path | None = None) -> Path:
    """Directory of the NGL copy: ``tenv3/``, ``steps.txt``,
    ``midas.IGS20.txt``, ``DataHoldings.txt``.

    Args:
        root: Archive root used instead of :func:`showcase_root`; ``None``
            reads ``$SHOWCASE_DATA`` (default ``~/data/showcase``).

    Returns:
        ``<root>/ngl``; not required to exist.
    """
    return (showcase_root() if root is None else Path(root)) / "ngl"


def isd_dir(year: int | None = None) -> Path:
    """Directory of the NOAA Global Hourly copy.

    ``$SHOWCASE_ISD`` outranks ``$SHOWCASE_DATA``: when it is set the copy
    is read from there whatever the archive root is.

    Args:
        year: Calendar year of the wanted subdirectory, one CSV per
            station; ``None`` for the directory holding the years.

    Returns:
        ``$SHOWCASE_ISD`` when set (``~`` expanded), else
        ``<showcase_root()>/isd``, with ``/<year>`` appended when ``year``
        is given; not required to exist.
    """
    env = os.environ.get("SHOWCASE_ISD")
    base = Path(env).expanduser() if env else showcase_root() / "isd"
    return base if year is None else base / str(year)


def normals_dir(root: Path | None = None) -> Path:
    """Directory of the reduced NOAA hourly normals.

    Args:
        root: Archive root used instead of :func:`showcase_root`; ``None``
            reads ``$SHOWCASE_DATA`` (default ``~/data/showcase``).

    Returns:
        ``<root>/normals``; not required to exist.
    """
    return (showcase_root() if root is None else Path(root)) / "normals"
