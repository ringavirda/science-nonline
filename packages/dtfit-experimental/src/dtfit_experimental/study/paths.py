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

    ``$DTFIT_DATA`` when set; else the pre-migration
    ``dtfit_experimental/experiments/data`` while that directory still exists
    on disk; else ``experiments/data`` beside the package source, the current
    location. Unlike :func:`experiments_dir`, does not require the directory
    to exist: ``study.download_data`` creates it on first use.
    """
    env = os.environ.get("DTFIT_DATA")
    if env:
        return Path(env)
    legacy = Path(__file__).resolve().parent.parent / "experiments" / "data"
    if legacy.is_dir():
        return legacy
    return Path(dtfit_experimental.__file__).resolve().parents[2] / "experiments" / "data"


DEFAULT_SHOWCASE = "~/data/showcase"


def showcase_root() -> Path:
    """The showcase archive root from ``$SHOWCASE_DATA`` (default
    ``~/data/showcase``); not created here."""
    return Path(os.environ.get("SHOWCASE_DATA", DEFAULT_SHOWCASE)).expanduser()


def ngl_dir(root: Path | None = None) -> Path:
    """The NGL copy ``<root>/ngl``: ``tenv3/``, ``steps.txt``,
    ``midas.IGS20.txt``, ``DataHoldings.txt``."""
    return (showcase_root() if root is None else Path(root)) / "ngl"


def isd_dir(year: int | None = None) -> Path:
    """The NOAA copy from ``$SHOWCASE_ISD``, or ``<showcase_root()>/isd``; with
    ``year``, that year's directory of one CSV per station."""
    env = os.environ.get("SHOWCASE_ISD")
    base = Path(env).expanduser() if env else showcase_root() / "isd"
    return base if year is None else base / str(year)


def normals_dir(root: Path | None = None) -> Path:
    """Where the reduced NOAA hourly normals are stored."""
    return (showcase_root() if root is None else Path(root)) / "normals"
