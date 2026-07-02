"""Smoke tests for dtfit-hardware.

The package is host-glue driving real silicon, so most of it can only run with a
board attached. These checks just confirm the package imports and ships its
firmware, without needing any hardware.
"""
from __future__ import annotations

import importlib.resources as resources
from pathlib import Path


def test_package_imports() -> None:
    import dtfit_hardware

    assert dtfit_hardware.__doc__


def test_firmware_sketches_present() -> None:
    root = resources.files("dtfit_hardware")
    firmware = Path(str(root)) / "firmware"
    assert firmware.is_dir(), "firmware/ should ship with the package"
    # The rig firmware and the BLE-telemetry sketch the phone app talks to.
    for sketch in ("nano_lsi_log", "nano_ble_telemetry"):
        assert (firmware / sketch).is_dir(), f"missing firmware/{sketch}"
