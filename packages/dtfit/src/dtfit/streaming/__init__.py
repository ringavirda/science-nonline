"""Online / real-time parameter tracking on the window image.

:class:`ImageFilter` ingests one sample at a time, images the sliding
window in a basis and updates the parameter estimate in information form,
with bounded per-update cost, in the Legendre or the block basis.
:class:`DriftDetector` is the shared change-detection logic
on a whitened innovation, usable on its own or as a building block for a
filter.
"""

from .filter import ImageFilter
from .detect import DriftDetector

__all__ = [
    "ImageFilter",
    "DriftDetector",
]
