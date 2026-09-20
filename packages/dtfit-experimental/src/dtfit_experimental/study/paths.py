"""Filesystem anchors the study tier's notebooks and downloaded data use."""

from __future__ import annotations

import os
from pathlib import Path

import dtfit_experimental


def experiments_dir() -> Path:
    """Root of the ``experiments/`` tree: notebooks, their results and data.

    Returns ``packages/dtfit-experimental/experiments``, beside the package's
    source rather than inside it. Raises ``FileNotFoundError`` naming the path
    when ``dtfit_experimental`` is not an editable checkout: the notebooks and
    their results only exist there, never in an installed wheel.
    """
    root = Path(dtfit_experimental.__file__).resolve().parents[2] / "experiments"
    if not root.is_dir():
        raise FileNotFoundError(f"not an editable checkout, no experiments tree: {root}")
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
