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

# Allowed drift from the snapshot before it counts as a regression. A pure
# absolute slack is nearly toothless here: most baseline param-errors are far
# below 0.03 (many ~1e-3, or ~1e-15 at noise=0), so a method could regress 5-10x
# and still pass. Guard RELATIVELY (allow ~50% worsening) with a small absolute
# floor so a near-zero baseline is not held to machine precision.
_PERR_REL = 1.5      # allow up to +50% relative param-error worsening ...
_PERR_ABS = 0.01     # ... or +0.01 absolute, whichever is larger
_R2_REL = 1.5        # allow the residual (1 - R^2) to grow up to 50% ...
_R2_ABS = 1e-4       # ... plus a small absolute floor

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
        limit = max(base["perr"] * _PERR_REL, base["perr"] + _PERR_ABS)
        assert now["perr"] <= limit, (
            f"{key}: param error regressed {base['perr']:.4g} -> {now['perr']:.4g} "
            f"(limit {limit:.4g})")
    else:
        # guard the residual (1 - R^2) relatively so a 0.9999 -> 0.999 drop trips
        limit = _R2_REL * (1.0 - base["r2"]) + _R2_ABS
        assert (1.0 - now["r2"]) <= limit, (
            f"{key}: R2 regressed {base['r2']:.5f} -> {now['r2']:.5f} "
            f"(residual limit {limit:.4g})")
