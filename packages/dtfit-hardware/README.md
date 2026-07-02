# dtfit-hardware — the real-silicon rig (hardware twin of `realtime_gps`)

The simulated [`realtime_gps`](../dtfit-experimental/src/dtfit_experimental/experiments/domains/realtime_gps/realtime_gps.ipynb)
domain (in `dtfit-experimental`) validates the streaming LSI/EAC trackers on a
**simulated** 9-DOF GPS/inertial rig. This package runs the same story on **real
hardware** — an Arduino Nano 33 BLE Sense + NEO-M8N GPS — and scores the on-MCU
filter against captured real-data logs and public RTK/INS datasets. It is the
engineering counterpart of the planned embedded paper.

It encapsulates everything device-specific: the Arduino **firmware**, the host
**telemetry link** (`backend.py`, USB + BLE), the real-log **comparison harness**
(`compare_real.py`), and a phone-side **BLE monitor app** (`mobile/`).

Depends on `dtfit-experimental` (to score against the sim's `realtime_gps.backend`)
→ `dtfit`. The dependency is one-directional: `dtfit-hardware` → `dtfit-experimental` → `dtfit`.

- **Report:** [`src/dtfit_hardware/realtime_gps_hw.ipynb`](src/dtfit_hardware/realtime_gps_hw.ipynb) — reproduces the sim's **E1/E2/E3/E5 on real-data logs** (forecast, dropout-coasting, glitch robustness, on-MCU cost + float32 bit-faithfulness), with rig status and BOM. A living report, expanded as more runs land.
- **Parts + wiring:** [`papers/embedded_hardware_bom.md`](../../papers/embedded_hardware_bom.md)
- **Beginner build guide:** [`papers/embedded_nano_build_guide.md`](../../papers/embedded_nano_build_guide.md)
- **Phone monitor:** [`mobile/dtfit-monitor/`](mobile/dtfit-monitor/) — its own README covers the React Native build.

## Layout

```
dtfit-hardware/
├── pyproject.toml
├── mobile/dtfit-monitor/       # phone-side live BLE view (React Native / Expo) -- see its README
└── src/dtfit_hardware/
    ├── realtime_gps_hw.ipynb   # THE REPORT: real-data E1/E2/E3/E5 + rig status (figures/tables)
    ├── backend.py              # host control + telemetry (find/flash/capture/log, USB + BLE)
    ├── compare_real.py         # real-log comparison: dtfit trackers vs Kalman/CT-EKF, matched
    │                           #   S=0 control, gyro-gated IMU fusion, glitch + float32 checks
    ├── firmware/               # Arduino sketches flashed to the board
    │   ├── nano_diagnostic/        # LED + IMU + I2C scan, no wiring
    │   ├── nano_lsi_log/           # THE rig firmware: on-MCU LSI + 9-DOF IMU + mag -> SD & BLE
    │   ├── nano_sd_dump/           # stream riglog.csv off the SD card over USB (no card reader)
    │   └── ...                     # gps_passthrough, lsi_onboard, rig_check, ble_telemetry
    ├── tools/                  # host tools (embed_lsi.py -> C tables, test_lsi.cpp)
    └── data/                   # captured telemetry logs (CSV; git-ignored, local-only)
```

The same `dtfit-gps` GATT service is consumed three ways: `backend.py` (PC, via
`bleak`) and [`mobile/dtfit-monitor/`](mobile/dtfit-monitor/) — a small React
Native app that shows GPS fix / sats / speed / IMU / on-MCU LSI live on an Android
phone, so you can tell the rig is working untethered in the field.

`backend.py` wraps the Arduino CLI (bundled with the Arduino IDE — no separate
install) and `pyserial`, so the notebook/tests can locate the board, flash a
sketch and capture its serial stream. There is **no NumPy simulation here**: this
package's "compute" is driving real silicon and logging what it returns.

## Setup

```bash
pip install -e packages/dtfit                       # core (dependency)
pip install -e packages/dtfit-experimental          # sim backend (dependency)
pip install -e 'packages/dtfit-hardware[bench]'     # this package + report plotting
```

`pyserial` + `bleak` (the USB and BLE links) install as core dependencies here.
The Arduino toolchain (board core + libraries) installs on first use, or:

```bash
arduino-cli core install arduino:mbed_nano
arduino-cli lib install Arduino_BMI270_BMM150 TinyGPSPlus
```

## Use

With the Nano 33 BLE on USB (close the Arduino IDE's Serial Monitor so it isn't
holding the port), from `src/dtfit_hardware/`:

```bash
python backend.py find          # detect the board + COM port
python backend.py selftest      # flash the diagnostic, confirm the IMU stream
python backend.py flash nano_gps_passthrough
python backend.py capture 10    # print 10 s of serial
python backend.py log data/run1.csv 60
python backend.py ble 20        # receive 20 s of BLE telemetry
```

From Python / the notebook:

```python
from dtfit_hardware import backend as rig   # or `import backend as rig` when run in-tree
rig.find_board()                       # -> Board(port='COM3', ...)
print(rig.self_test())                 # flash diagnostic + verify telemetry
rig.flash("nano_gps_passthrough")
for line in rig.capture(seconds=10):   # raw NMEA once the GPS is wired
    print(line)
```

## Bring-up stages (mirrors the build guide)

| Stage | Firmware | Proves | Wiring |
|---|---|---|---|
| 0 | `nano_diagnostic` | upload path, USB serial, onboard IMU, I2C bus | none |
| 2 | `nano_gps_passthrough` | NEO-M8N UART link (raw NMEA) | GPS↔Nano UART + 5 V |
| 5 | `nano_ble_telemetry` | untethered GPS+IMU logging over BLE | battery + boost |

> **Stage 4 (INA226 energy-per-estimate) is dropped.** Both INA226 modules are dead — the
> chip never ACKs even on an I2C bus proven good by the GPS magnetometer at `0x0E`, with VCC
> at 3.2 V and SDA↔A4 / SCL↔A5 continuity verified. With no working power meter, the
> energy-per-estimate experiment is cut; the on-MCU **cost story rests on µs/update + RAM**
> instead (measured on the M4F via `nano_lsi_onboard`: ~267 µs/update, sub-kB state).

Each hardware stage reproduces a cell (E1–E7) of the `realtime_gps` simulation on
real silicon; the notebook compares the captured logs to the simulated baselines
(constant-accel Kalman, CT-EKF) and to public RTK/INS ground truth.
