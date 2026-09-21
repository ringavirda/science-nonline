"""dtfit-hardware: the real-silicon rig, hardware twin of the GPS simulation.

``dtfit_experimental.study.gps`` simulates a 9-DOF GPS/inertial rig in
NumPy. This package runs the same study on an Arduino
Nano 33 BLE Sense (onboard IMU and BLE) reading a NEO-M8N GPS.

* ``backend.py`` drives the board from the host: locate it, flash a sketch
  from ``firmware/``, capture the USB or BLE stream.
* ``compare_real.py`` scores captured logs against the simulation's
  baselines in ``dtfit_experimental.study.gps``: Kalman/CT-EKF, IMU fusion,
  glitch and float32.
* ``firmware/`` holds the Arduino sketches. ``nano_lsi_log`` is the one the
  rig actually runs.
* ``tools/`` holds the host tools; ``embed_lsi.py`` bakes the LSI tables
  into C. The firmware carries the window (LSI) image only; the block (EAC)
  image is assembled host-side.
* ``mobile/`` is a React Native app reading the ``dtfit-gps`` BLE telemetry
  live (built with yarn; see ``mobile/dtfit-monitor/README.md``).

``experiments/rig.ipynb`` is the report: the recorded drives against the
simulation's baselines and the on-chip filter's cost, memory and float32
agreement. ``realtime_gps_hw.ipynb`` holds the rig status and the BOM.
"""
