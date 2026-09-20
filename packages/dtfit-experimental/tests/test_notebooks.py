"""Executes every notebook under the new experiments tree in quick mode.

One case a notebook, with a per-cell and a whole-notebook time budget. Two
guard tests catch the harness itself pointing at the wrong tree.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

from dtfit_experimental.study import paths  # noqa: E402

CELL_TIMEOUT = 120
NOTEBOOK_BUDGET = 300


def discover_notebooks() -> list[Path]:
    """Every ``.ipynb`` under ``experiments/{method,technology,realdata}``,
    sorted, skipping any that live under a ``.ipynb_checkpoints`` directory.
    """
    root = paths.experiments_dir()
    found = []
    for sub in ("method", "technology", "realdata"):
        for nb in sorted((root / sub).glob("*.ipynb")):
            if ".ipynb_checkpoints" in nb.parts:
                continue
            found.append(nb)
    return found


def _notebook_id(nb: Path) -> str:
    return str(nb.relative_to(paths.experiments_dir()))


@pytest.mark.parametrize(
    "notebook", discover_notebooks(), ids=_notebook_id
)
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


def test_discovery_root_is_the_new_experiments_tree():
    root = paths.experiments_dir()
    # fails if the glob pointed at the old src/.../experiments tree
    assert root == Path(__file__).resolve().parents[1] / "experiments"
    for nb in discover_notebooks():
        assert "src" not in nb.parts


def test_package_resolves_inside_this_checkout():
    import dtfit_experimental

    pkg_root = Path(dtfit_experimental.__file__).resolve().parents[2]
    this_root = Path(__file__).resolve().parents[1]
    # fails if the worktree's tests ran against the main tree's package
    assert pkg_root == this_root
