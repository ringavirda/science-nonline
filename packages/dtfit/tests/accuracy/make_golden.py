"""Regenerate the golden accuracy baseline.

Snapshots the recovery metrics of the realistic ``Model.fit`` path for every
scenario x noise level into ``golden_baseline.json``. Run deliberately, after an
*intended* accuracy change, and review the diff::

    python -m accuracy.make_golden        # from packages/dtfit/tests

The Phase-4 guard (:mod:`test_accuracy_regression`) fails if any scenario later
drifts worse than this snapshot by more than its tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

from .scenarios import SCENARIOS, NOISE_LEVELS
from .harness import metrics_for

GOLDEN_PATH = Path(__file__).with_name("golden_baseline.json")


def build() -> dict:
    data: dict[str, dict] = {}
    for scn in SCENARIOS:
        for noise in NOISE_LEVELS:
            data[f"{scn.name}@{noise:g}"] = metrics_for(scn, noise, seed=0)
    return data


def main() -> None:
    data = build()
    GOLDEN_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(data)} entries to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
