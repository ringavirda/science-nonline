"""Hardware twin of the ``realtime_gps`` domain -- the real-silicon rig.

Where ``realtime_gps`` simulates a 9-DOF GPS/inertial rig in NumPy, this domain
*runs it on hardware*: an Arduino Nano 33 BLE Sense (onboard IMU + BLE) reading a
NEO-M8N GPS, with an INA226 measuring energy-per-estimate. ``backend.py`` is the
single source of truth for *driving the board* (locate it, flash a firmware
sketch from ``firmware/``, capture the telemetry stream); the notebook scores the
captured real-data logs against the simulation's E1-E7 story.

Build details live with the paper notes:
``papers/embedded_hardware_bom.md`` (parts + wiring tables) and
``papers/embedded_nano_build_guide.md`` (beginner step-by-step bring-up).
"""
