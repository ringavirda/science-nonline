"""Notebook helpers: result export, optional data and imports, provenance.

Each test names the mutation of notebook.py it fails under.
"""

from __future__ import annotations

import re
import time
import warnings

import pandas as pd
import pytest

from dtfit_experimental.study.notebook import (
    data_file,
    optional_import,
    plain_warnings,
    provenance,
    results_dir,
    save_table,
)


def test_results_dir_creates_the_directory(tmp_path):
    out = results_dir("13_two_bases", root=tmp_path)
    # fails if results_dir returned a path without creating it
    assert out.is_dir()
    assert out == tmp_path / "results" / "13_two_bases"


def test_save_table_drops_an_unnamed_index(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    save_table("nb", "t", frame, root=tmp_path)
    text = (tmp_path / "results" / "nb" / "t.csv").read_text()
    # fails if save_table wrote index=True unconditionally
    assert "Unnamed" not in text
    assert text.splitlines()[0] == "a,b"


def test_save_table_keeps_a_named_index(tmp_path):
    frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="k"))
    save_table("nb", "t", frame, root=tmp_path)
    back = pd.read_csv(tmp_path / "results" / "nb" / "t.csv", index_col="k")
    # fails if save_table always wrote index=False, dropping the label
    assert list(back.index) == ["x", "y"]


def test_save_table_returns_the_frame_unchanged(tmp_path):
    frame = pd.DataFrame({"a": [1]})
    out = save_table("nb", "t", frame, root=tmp_path)
    # fails if save_table returns None
    assert out is frame


def test_save_table_round_trip_reproduces_the_numbers(tmp_path):
    frame = pd.DataFrame({"a": [1.0 / 3.0, 2.0]})
    save_table("nb", "t", frame, root=tmp_path)
    back = pd.read_csv(tmp_path / "results" / "nb" / "t.csv")
    # fails under a float_format that truncates past 6 significant digits
    assert back["a"][0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_data_file_missing_prints_and_returns_none(tmp_path, capsys):
    out = data_file("no_such.csv", root=tmp_path)
    # fails if data_file raises instead of returning None
    assert out is None
    # fails if the missing path were not named on the printed line
    assert "no_such.csv" in capsys.readouterr().out


def test_data_file_present_returns_the_path(tmp_path):
    (tmp_path / "present.csv").write_text("a,b\n1,2\n")
    out = data_file("present.csv", root=tmp_path)
    assert out == tmp_path / "present.csv"


def test_optional_import_missing_prints_and_returns_none(capsys):
    out = optional_import("no_such_package_xyz")
    # fails if optional_import lets ImportError propagate
    assert out is None
    assert "no_such_package_xyz" in capsys.readouterr().out


def test_optional_import_present_returns_the_module():
    out = optional_import("os")
    # fails if optional_import always returned None regardless of import
    assert out is not None
    assert out.__name__ == "os"


def test_provenance_carries_a_commit_and_wall_time():
    started = time.time() - 0.05
    line = provenance(started)
    # fails if the commit hash were dropped or replaced by a constant
    assert re.search(r"commit [0-9a-f]{7}|commit unknown", line)
    match = re.search(r"(-?[0-9.]+)s$", line)
    assert match is not None
    # fails if elapsed were computed from the wrong end of the interval
    # (started - now instead of now - started, a negative reading)
    assert float(match.group(1)) >= 0.05


def test_plain_warnings_shows_the_category_and_message_only(capsys):
    """Fails if ``plain_warnings`` leaves the default display in place."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        plain_warnings()
        warnings.warn("the solve restarted", UserWarning, stacklevel=1)
    assert capsys.readouterr().err == "UserWarning: the solve restarted\n"
