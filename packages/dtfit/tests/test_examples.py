"""Execute every example script so the guides stay in step with the API.

The examples/ scripts are the runnable documentation; running them here turns a
public-API rename or signature change into a failing test instead of a stale doc.
Each script is headless (prints only) and fast.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("0*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(script):
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed:\n{proc.stdout}\n{proc.stderr}"
    )
