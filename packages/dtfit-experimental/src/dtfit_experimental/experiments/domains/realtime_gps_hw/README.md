# realtime_gps_hw — the hardware rig (real-silicon twin of `realtime_gps`)

The [`realtime_gps/`](../realtime_gps/realtime_gps.ipynb) domain validates the
streaming LSI/EAC trackers on a **simulated** 9-DOF GPS/inertial rig. This domain
runs the same story on **real hardware** and scores the on-MCU filter against
captured real-data logs and public RTK/INS datasets. It is the engineering
counterpart of the planned embedded paper.

- **Report (this domain's notebook):** [`realtime_gps_hw.ipynb`](realtime_gps_hw.ipynb) — reproduces the sim's **E1/E2/E3/E5 on real-data logs** (forecast, dropout-coasting, glitch robustness, on-MCU cost + float32 bit-faithfulness), with the rig status and BOM. A living report, expanded as more runs land.
- **Parts + wiring:** [`papers/embedded_hardware_bom.md`](../../../../../../../papers/embedded_hardware_bom.md)
- **Beginner build guide:** [`papers/embedded_nano_build_guide.md`](../../../../../../../papers/embedded_nano_build_guide.md)

## Layout

```
realtime_gps_hw/
├── realtime_gps_hw.ipynb  # THE REPORT: real-data E1/E2/E3/E5 + rig status (figures/tables)
├── backend.py     # host control + telemetry (find/flash/capture/log, USB + BLE)
├── compare_real.py# real-log comparison harness: dtfit trackers vs Kalman/CT-EKF, the
│                  #   matched S=0 control, gyro-gated IMU fusion, glitch + float32 checks
├── firmware/      # Arduino sketches flashed to the board
│   ├── nano_diagnostic/        # LED + IMU + I2C scan, no wiring
│   ├── nano_lsi_log/           # THE rig firmware: on-MCU LSI + 9-DOF IMU + mag -> SD & BLE
│   ├── nano_sd_dump/           # stream riglog.csv off the SD card over USB (no card reader)
│   └── ...                     # gps_passthrough, lsi_onboard, rig_check, ble_telemetry
├── data/          # captured telemetry logs (CSV; sd_s2.csv = the clean session used here)
└── figures/       # report figures (written by the notebook)
```

`backend.py` wraps the Arduino CLI (bundled with the Arduino IDE — no separate
install) and `pyserial`, so the notebook/tests can locate the board, flash a
sketch and capture its serial stream. There is **no NumPy simulation here**: this
domain's "compute" is driving real silicon and logging what it returns.

## Setup

```bash
pip install -e '.[bench,rig]'     # rig = pyserial; bench = plotting for the report
```

The Arduino toolchain (board core + libraries) installs on first use, or:

```bash
arduino-cli core install arduino:mbed_nano
arduino-cli lib install Arduino_BMI270_BMM150 TinyGPSPlus
```

## Use

With the Nano 33 BLE on USB (close the Arduino IDE's Serial Monitor so it isn't
holding the port):

```bash
python backend.py find          # detect the board + COM port
python backend.py selftest      # flash the diagnostic, confirm the IMU stream
python backend.py flash nano_gps_passthrough
python backend.py capture 10    # print 10 s of serial
python backend.py log data/run1.csv 60
```

From Python / the notebook:

```python
import backend as rig
rig.find_board()                       # -> Board(port='COM3', ...)
print(rig.self_test())                 # flash diagnostic + verify telemetry
rig.flash("nano_gps_passthrough")
for line in rig.capture(seconds=10):   # raw NMEA once the GPS is wired
    print(line)
rig.parse_imu("IMU a=0.01,-0.02,1.00  g=0.1,0.0,-0.2  m=...")
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
