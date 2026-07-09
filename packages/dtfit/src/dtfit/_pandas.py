"""Optional-pandas interop helpers (guarded).

pandas is an **optional** dependency of dtfit: it must never be imported at
module top level in a way that would make dtfit require it, and every ndarray
path must work (and stay bit-identical) with pandas absent. This module owns the
one guarded import and exposes small duck-typing helpers so the fitters and the
result type can accept pandas ``Series`` / single-column ``DataFrame`` inputs and
return index-aligned ``Series`` outputs ("pandas in -> pandas out") without any
other module importing pandas directly.

Every helper is safe when pandas is missing: :data:`HAS_PANDAS` is then
``False``, the ``is_*`` predicates return ``False``, :func:`to_1d_array` still
coerces plain array-likes, and the index helpers degrade to ``None`` / the plain
ndarray.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pandas is an optional dependency (not in the core install)
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - exercised only in a pandas-free env
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False


# One consistent message for the 1-D boundary, shared across the entry points:
# dtfit's integral criteria are one-dimensional, so multivariate X is a hard
# scope limit rather than a passing validation error. The pointer to
# additively-separable composition is the supported escape hatch.
_MULTIVARIATE_MSG = (
    "{name} must be 1-D; got {what}. dtfit's integral criteria (LSI/EAC) are "
    "one-dimensional, so multivariate X (several predictors) is not supported. "
    "If instead you have a 1-D signal that is a sum of components along one axis "
    "(e.g. trend + cycle), compose 1-D models with `+` "
    "(models.linear() + models.sine()); see the 'Multivariate data' note in the "
    "docs."
)


def is_series(obj: Any) -> bool:
    """True if ``obj`` is a pandas ``Series`` (always ``False`` without pandas)."""
    return HAS_PANDAS and isinstance(obj, pd.Series)


def is_dataframe(obj: Any) -> bool:
    """True if ``obj`` is a pandas ``DataFrame`` (always ``False`` without pandas)."""
    return HAS_PANDAS and isinstance(obj, pd.DataFrame)


def to_1d_array(obj: Any, name: str = "x") -> np.ndarray:
    """Coerce ``obj`` to a 1-D float ``ndarray``.

    A pandas ``Series`` becomes its float values; a **single-column**
    ``DataFrame`` becomes that one column's float values (so a one-column frame
    no longer trips the historical "must be 1-D" error); a genuinely
    **multivariate** input (a multi-column ``DataFrame`` or a 2-D array with more
    than one column) raises a :class:`ValueError` -- dtfit is one-dimensional.
    A 1-D ndarray / list is ``np.asarray(obj, float).reshape(-1)``, unchanged and
    **bit-identical**; a column vector ``(n, 1)`` is squeezed to ``(n,)``.
    """
    if is_series(obj):
        return np.asarray(obj.to_numpy(dtype=float)).reshape(-1)
    if is_dataframe(obj):
        ncol = obj.shape[1]
        if ncol != 1:
            raise ValueError(_MULTIVARIATE_MSG.format(name=name, what=f"a DataFrame with {ncol} columns"))
        return np.asarray(obj.iloc[:, 0].to_numpy(dtype=float)).reshape(-1)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim >= 2 and int(np.prod(arr.shape[1:])) != 1:
        # A real multivariate array -- do NOT silently flatten it (that would
        # turn an nD mistake into a wrong 1-D fit). A single trailing column is a
        # column vector and is squeezed below.
        raise ValueError(_MULTIVARIATE_MSG.format(name=name, what=f"an array of shape {arr.shape}"))
    return arr.reshape(-1)


def capture_index(obj: Any) -> Any:
    """The pandas index of a ``Series`` / ``DataFrame``, else ``None``."""
    if is_series(obj) or is_dataframe(obj):
        return obj.index
    return None


def extend_index(index: Any, horizon: int) -> Any:
    """The length-``horizon`` FUTURE index continuing ``index``.

    * ``DatetimeIndex`` -- continued at its frequency (``index.freq`` when set,
      else inferred via :func:`pandas.infer_freq`); ``None`` when the frequency
      cannot be inferred.
    * ``RangeIndex`` or any **integer-typed** index -- continued by its constant
      step (the range step, or the difference of the last two labels).
    * anything else -- ``None``.

    Returns ``None`` when pandas is absent, ``index`` is ``None`` / empty, or
    ``horizon <= 0``.
    """
    if not HAS_PANDAS or index is None or horizon <= 0 or len(index) == 0:
        return None
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or pd.infer_freq(index)
        if freq is None:
            return None
        return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]
    if isinstance(index, pd.RangeIndex):
        step = index.step
        start = int(index[-1]) + step
        return pd.RangeIndex(start, start + step * horizon, step)
    if pd.api.types.is_integer_dtype(getattr(index, "dtype", None)):
        if len(index) < 2:
            return None
        step = int(index[-1]) - int(index[-2])
        last = int(index[-1])
        return pd.Index([last + step * (i + 1) for i in range(horizon)])
    return None


def as_series(values: np.ndarray, index: Any) -> Any:
    """A pandas ``Series`` of ``values`` aligned to ``index`` when possible.

    Returns the plain ``ndarray`` unchanged when pandas is absent or ``index`` is
    ``None`` -- so a non-pandas caller is never handed a ``Series``.
    """
    arr = np.asarray(values)
    if HAS_PANDAS and index is not None:
        return pd.Series(arr, index=index)
    return arr
