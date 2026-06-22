"""Online / real-time estimators for streaming data.

These estimators ingest one sample at a time with bounded per-update cost,
suitable for control loops and Big-Data streams. Symbolic work (model and
Jacobian) is done once at construction; updates are pure NumPy/SciPy.

Each filter is the **streaming twin of a batch method** and is named for that
method, so the same method is discoverable across execution modes:

- :class:`EACFilter` -- streaming twin of :func:`dtfit.fit_eac` /
  :class:`dtfit.PartitionedEAC`; its measurement is the integrated **area**
  innovation over a sliding window.
- :class:`LSIFilter` -- streaming twin of :func:`dtfit.fit_lsi` /
  :class:`dtfit.PartitionedLSI`; its measurement is the window's **Legendre
  spectrum** (a richer measurement that captures oscillations the area criterion
  partly cancels).
"""

from ._eac import EACFilter
from ._lsi import LSIFilter
from ._bank import FilterBank, FusedChiSquareDetector

__all__ = ["EACFilter", "LSIFilter", "FilterBank", "FusedChiSquareDetector"]
