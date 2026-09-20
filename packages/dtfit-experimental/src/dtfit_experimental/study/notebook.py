"""Notebook-facing helpers: result export, optional data and imports, and
a provenance line for a closing cell. No experiment logic lives here.
"""

from __future__ import annotations

import importlib
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from . import paths

if TYPE_CHECKING:
    import pandas as pd


def results_dir(notebook: str, *, root: Path | None = None) -> Path:
    """Directory a notebook's exported tables land in.

    Creates ``<root>/results/<notebook>`` if it does not already exist, so
    a fresh checkout can save a table on the first run.

    Args:
        notebook: Notebook identifier, the file stem (for example
            ``"13_two_bases"``).
        root: Root of the experiments tree; ``study.paths.experiments_dir()``
            when ``None``. A caller outside the tree (the rig notebook in
            ``dtfit-hardware``) passes its own root.

    Returns:
        The directory, guaranteed to exist.

    Raises:
        FileNotFoundError: propagated from ``study.paths.experiments_dir()``
            when ``root`` is ``None`` and the package is not an editable
            checkout.
    """
    if root is None:
        root = paths.experiments_dir()
    out = root / "results" / notebook
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_table(
    notebook: str,
    table: str,
    frame: "pd.DataFrame",
    *,
    root: Path | None = None,
) -> "pd.DataFrame":
    """Write a notebook table to ``experiments/results/<notebook>/<table>.csv``
    and return it unchanged.

    The index is written only when it is named (a deliberate label such as
    an id column); a positional ``RangeIndex`` is dropped. Numbers are
    formatted ``%.6g`` and lines end with ``\\n`` regardless of platform, so
    a rerun that reproduces the same numbers reproduces the same bytes.

    Args:
        notebook: Notebook identifier, as for :func:`results_dir`.
        table: Table name, the file stem the ``.csv`` suffix is added to.
        frame: Table to write.
        root: Root of the experiments tree; ``study.paths.experiments_dir()``
            when ``None``.

    Returns:
        ``frame``, unchanged, so the caller's last line displays it.

    Raises:
        FileNotFoundError: propagated from ``study.paths.experiments_dir()``
            when ``root`` is ``None`` and the package is not an editable
            checkout.
    """
    out = results_dir(notebook, root=root) / f"{table}.csv"
    frame.to_csv(
        out,
        index=frame.index.name is not None,
        float_format="%.6g",
        lineterminator="\n",
    )
    return frame


def data_file(*parts: str, root: Path | None = None) -> Path | None:
    """Locate a data file under the study data directory, or skip.

    Args:
        *parts: Path components under the data directory (for example
            ``"usd_uah_2014_2015.csv"``).
        root: Data directory; ``study.paths.data_dir()`` when ``None``.

    Returns:
        The path when it exists on disk. ``None`` when it does not, after
        printing one line naming it, so a notebook cell can skip its
        section with ``if PATH:`` instead of raising: nbclient fails the
        whole notebook on an uncaught exception.
    """
    if root is None:
        root = paths.data_dir()
    path = root.joinpath(*parts)
    if not path.exists():
        print(f"data file missing, skipping: {path}")
        return None
    return path


def optional_import(name: str) -> ModuleType | None:
    """Import a module the research CI job does not install, or skip.

    Args:
        name: Module name to import (for example ``"statsmodels"``,
            ``"torch"`` or ``"cupy"``).

    Returns:
        The imported module. ``None`` when it is not installed, after
        printing one line naming it, so a notebook cell can skip its
        section with ``if mod:`` instead of raising.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"{name} not installed, skipping")
        return None


def provenance(started: float) -> str:
    """One line for a notebook's closing cell: when and at what commit and
    package versions it ran, and how long it took.

    Args:
        started: ``time.time()`` reading taken at the top of the notebook.

    Returns:
        A single line with the UTC date, the short commit hash (``unknown``
        outside a git checkout), the installed ``dtfit`` and
        ``dtfit-experimental`` versions, and the wall time since ``started``
        in seconds.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    dtfit_version = metadata.version("dtfit")
    exp_version = metadata.version("dtfit-experimental")
    elapsed = time.time() - started
    return (
        f"{date} commit {commit} dtfit {dtfit_version} "
        f"dtfit-experimental {exp_version} {elapsed:.1f}s"
    )
