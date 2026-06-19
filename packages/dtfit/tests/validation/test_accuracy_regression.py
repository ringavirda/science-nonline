"""Phase-4 golden-regression guard.

Pins the recovery accuracy of every scenario to a checked-in snapshot
(``accuracy/golden_baseline.json``). A change that silently degrades any
catalogue family -- the drift that let docs surprises slip in unnoticed -- fails
here. Intended accuracy improvements are adopted by regenerating the snapshot
(``python -m accuracy.make_golden``) and reviewing the diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accuracy.scenarios import SCENARIOS, NOISE_LEVELS
from accuracy.harness import metrics_for

_GOLDEN = json.loads(
    (Path(__file__).parents[1] / "accuracy" / "golden_baseline.json").read_text()
)

# Allowed drift from the snapshot before it counts as a regression.
_PERR_SLACK = 0.03   # absolute relative-error worsening
_R2_SLACK = 0.01     # absolute R^2 worsening

_CASES = [(s, noise) for s in SCENARIOS for noise in NOISE_LEVELS]
_IDS = [f"{s.name}@{noise:g}" for s, noise in _CASES]


def test_golden_covers_every_case():
    """The snapshot must cover the whole corpus -- no scenario silently unguarded."""
    expected = {f"{s.name}@{noise:g}" for s in SCENARIOS for noise in NOISE_LEVELS}
    assert set(_GOLDEN) == expected, (
        "golden_baseline.json is stale; regenerate with `python -m accuracy.make_golden`")


@pytest.mark.parametrize("scn,noise", _CASES, ids=_IDS)
def test_no_accuracy_regression(scn, noise):
    key = f"{scn.name}@{noise:g}"
    base = _GOLDEN[key]
    now = metrics_for(scn, noise, seed=0)
    if base["metric"] == "params":
        assert now["perr"] <= base["perr"] + _PERR_SLACK, (
            f"{key}: param error regressed {base['perr']:.3f} -> {now['perr']:.3f}")
    else:
        assert now["r2"] >= base["r2"] - _R2_SLACK, (
            f"{key}: R2 regressed {base['r2']:.4f} -> {now['r2']:.4f}")
