"""The recursive (filter) forms of the integral criteria: parameters as a
Kalman state, the integral residual on a sliding window as the measurement.
Restored from dtfit at commit 17f43aa (2026-09-06)."""

from ._eac import EACFilter
from ._lsi import LSIFilter

__all__ = ["EACFilter", "LSIFilter"]
