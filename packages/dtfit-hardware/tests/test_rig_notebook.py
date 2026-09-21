"""Executes the rig notebook in quick mode. Its own recipe, root and limits:
dtfit-hardware is not dtfit-experimental's package and carries no test-suite
dependency on it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

import dtfit_hardware  # noqa: E402

CELL_TIMEOUT = 120
NOTEBOOK_BUDGET = 300


def discover_notebooks() -> list[Path]:
    """Every ``.ipynb`` directly under ``experiments/``, sorted, skipping any
    that live under a ``.ipynb_checkpoints`` directory."""
    root = Path(dtfit_hardware.__file__).resolve().parents[2] / "experiments"
    return sorted(
        nb for nb in root.glob("*.ipynb")
        if ".ipynb_checkpoints" not in nb.parts
    )


def _notebook_id(nb: Path) -> str:
    root = Path(dtfit_hardware.__file__).resolve().parents[2] / "experiments"
    return str(nb.relative_to(root))


@pytest.mark.parametrize("notebook", discover_notebooks(), ids=_notebook_id)
def test_notebook_runs_quick(notebook, monkeypatch):
    monkeypatch.setenv("DTFIT_QUICK", "1")
    nb = nbformat.read(notebook, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    started = time.time()
    client.execute()
    elapsed = time.time() - started
    assert elapsed <= NOTEBOOK_BUDGET, (
        f"{notebook} took {elapsed:.1f}s, over the {NOTEBOOK_BUDGET}s budget"
    )


def test_discovery_root_is_hw_experiments():
    root = Path(dtfit_hardware.__file__).resolve().parents[2] / "experiments"
    # fails if the glob pointed at the package source tree instead
    assert root == Path(__file__).resolve().parents[1] / "experiments"
    for nb in discover_notebooks():
        assert Path(dtfit_hardware.__file__).resolve().parent not in nb.parents


def test_package_resolves_inside_this_checkout():
    pkg_root = Path(dtfit_hardware.__file__).resolve().parents[2]
    this_root = Path(__file__).resolve().parents[1]
    # fails if the worktree's tests ran against the main tree's package
    assert pkg_root == this_root
