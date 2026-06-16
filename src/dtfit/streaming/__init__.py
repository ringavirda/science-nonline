"""Online / real-time estimators for streaming data.

These estimators ingest one sample at a time with bounded per-update cost,
suitable for control loops and Big-Data streams. Symbolic work (model and
Jacobian) is done once at construction; updates are pure NumPy/SciPy.
"""

from .equal_areas import EqualAreasFilter

__all__ = ["EqualAreasFilter"]
