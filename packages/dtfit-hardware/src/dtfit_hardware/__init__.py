"""dtfit-hardware -- the real-silicon rig (hardware twin of ``realtime_gps``).

Where the ``realtime_gps`` domain (in ``dtfit-experimental``) simulates a 9-DOF
GPS/inertial rig in NumPy, this package *runs it on hardware*: an Arduino Nano 33
BLE Sense (onboard IMU + BLE) reading a NEO-M8N GPS. It encapsulates everything
device-specific:

* ``backend.py``   -- host control + telemetry: locate the board, flash a sketch
  from ``firmware/``, capture the USB/BLE stream. The single source of truth for
  *driving the board* (no NumPy simulation here).
* ``compare_real.py`` -- scores captured real-data logs against the sim's
  ``realtime_gps.backend`` baselines (Kalman/CT-EKF, IMU fusion, glitch/float32).
* ``firmware/``    -- the Arduino sketches (``nano_lsi_log`` is the rig firmware).
* ``tools/``       -- host tools (``embed_lsi.py`` bakes LSI tables into C).
* ``mobile/``      -- a React Native phone app reading the ``dtfit-gps`` BLE
  telemetry live (built with yarn; see ``mobile/dtfit-monitor/README.md``).

The report notebook (``realtime_gps_hw.ipynb``) reproduces the simulation's
E1-E7 story on real silicon. Build details: ``papers/embedded_hardware_bom.md``
(parts + wiring) and ``papers/embedded_nano_build_guide.md`` (bring-up guide).
"""
