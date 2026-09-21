"""The window image filter for a polynomial in time, carried in the time of
its newest sample.

:class:`dtfit.ImageFilter` evaluates its model at the absolute time stamps
it buffers, so the parameters of ``c0 + c1*t + c2*t**2`` are the position,
velocity and half the acceleration extrapolated back to ``t = 0``, the
process noise is added to those, and the estimate depends on where the
clock's zero sits. A polynomial is closed under a shift of its argument:
moving the origin by ``d`` maps the coefficients through the matrix
``T(d)`` with ``T[i, j] = C(j, i) * d**(j - i)``, exactly. The filter here
moves the origin onto every new sample before it measures, ``p <- T p`` and
``P <- T P T^T + q_rate * d``, and shifts the buffered times with it, so
the model is only ever evaluated on ``[-span, 0]``, the coefficients are the
position, velocity and half the acceleration at the newest sample, and a
shift of the whole clock changes nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from dtfit import ImageFilter

__all__ = ["LocalTimeFilter", "shift_matrix"]

_REFUSED = ("q_diag", "regressors", "stream", "param_names")


def shift_matrix(degree: int, d: float) -> np.ndarray:
    """The map of a polynomial's coefficients under a shift of its origin.

    For ``f(t) = sum_j c[j] * t**j`` and ``g(s) = f(s + d)`` it returns the
    ``(degree + 1, degree + 1)`` upper-triangular ``T`` with ``g``'s
    coefficients equal to ``T @ c``: ``T[i, j] = C(j, i) * d**(j - i)``.

    Args:
        degree: The polynomial degree, at least 0.
        d: The shift of the origin, in the units of ``t``; any finite value,
            ``T(0)`` is the identity and ``T(-d)`` inverts ``T(d)``.
    """
    n = degree + 1
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            T[i, j] = math.comb(j, i) * d ** (j - i)
    return T


class LocalTimeFilter(ImageFilter):
    """:class:`dtfit.ImageFilter` on a polynomial trend whose coefficients
    refer to the newest sample's time.

    Args:
        degree: Degree of the polynomial, 1 to 9. The parameters are
            ``c0 .. c<degree>``: the value, the first derivative, and
            ``1 / j!`` of the ``j``-th derivative at the newest sample.
        q_rate: Process-noise variance added to each coefficient per unit of
            time between samples (units of ``y`` squared per unit of ``t``
            for ``c0``); one non-negative value or one per coefficient,
            default 0.1. A gap of ``d`` adds ``q_rate * d``, so one setting
            carries across sampling rates.
        p0: The coefficients at the first sample, default ones.
        **filter_options: Any other keyword of :class:`dtfit.ImageFilter`
            (``basis``, ``order``, ``window_size``, ``min_window``,
            ``adaptive_window``, ``noise_var``, ``robust``, ``huber_c``,
            ``drift_reset``, ``drift_inflation``, ``alpha``, ``cusum_k``,
            ``cusum_h``).

    Attributes:
        t_ref_: Time of the newest ingested sample, the origin ``p`` and
            ``P`` refer to; ``None`` before the first sample.

    :meth:`predict`, :meth:`predict_cov`, :meth:`coast` and
    :meth:`coast_cov` take absolute times. :meth:`result` reports the
    coefficients at ``t_ref_``, in the local time the window is held in.

    Raises:
        ValueError: ``degree`` outside 1 to 9; a negative or wrongly sized
            ``q_rate``; one of ``q_diag``, ``regressors``, ``stream`` or
            ``param_names`` passed; whatever :class:`dtfit.ImageFilter`
            raises for the rest.
    """

    def __init__(
        self,
        degree: int = 2,
        *,
        q_rate: float | Sequence[float] = 0.1,
        p0: Sequence[float] | None = None,
        **filter_options: Any,
    ) -> None:
        if not 1 <= int(degree) <= 9:
            raise ValueError(f"degree must be 1 to 9, got {degree}")
        refused = [k for k in _REFUSED if k in filter_options]
        if refused:
            raise ValueError(
                f"{', '.join(refused)} cannot be set on a LocalTimeFilter"
            )
        self.degree = int(degree)
        n = self.degree + 1
        rate = np.atleast_1d(np.asarray(q_rate, dtype=float))
        if rate.size == 1:
            rate = np.full(n, float(rate[0]))
        if rate.shape != (n,) or not np.all(rate >= 0.0):
            raise ValueError(
                f"q_rate must be one non-negative value or {n} of them"
            )
        self.q_rate: np.ndarray = rate
        terms = ["c0", "c1*tau"] + [f"c{j}*tau**{j}" for j in range(2, n)]
        super().__init__(
            " + ".join(terms), "tau", p0=p0, q_diag=[0.0] * n,
            **filter_options,
        )
        self.t_ref_: float | None = None

    def partial_fit(
        self, t_new: Any, y_new: Any, regressors: Any = None
    ) -> "LocalTimeFilter":
        """Move the origin onto ``t_new``, then ingest ``(t_new, y_new)``.

        The state is transported over the gap since the previous sample and
        gains ``q_rate`` times that gap whether or not the window then
        measures. A non-finite ``t_new`` or ``y_new`` is skipped with a
        ``RuntimeWarning`` and moves nothing.

        Raises:
            ValueError: ``t_new`` earlier than the previous sample;
                ``regressors`` given.
        """
        if regressors is not None:
            raise ValueError("a LocalTimeFilter takes no regressors")
        t_val = float(t_new)
        if not (np.isfinite(t_val) and np.isfinite(float(y_new))):
            super().partial_fit(t_new, y_new)
            return self
        if self.t_ref_ is not None:
            d = t_val - self.t_ref_
            if d < 0.0:
                raise ValueError(
                    f"samples must arrive in time order: {t_val} follows "
                    f"{self.t_ref_}"
                )
            T = shift_matrix(self.degree, d)
            self.p = T @ self.p
            P = T @ self.P @ T.T + np.diag(self.q_rate * d)
            self.P = 0.5 * (P + P.T)
            self._t = [tau - d for tau in self._t]
        self.t_ref_ = t_val
        super().partial_fit(0.0, y_new)
        return self

    update = partial_fit

    def _local(self, x: Any) -> np.ndarray:
        if self.t_ref_ is None:
            raise RuntimeError("no sample ingested yet")
        return np.asarray(x, dtype=float) - self.t_ref_

    def predict(self, x: Any, regressors: Any = None) -> np.ndarray:
        """The polynomial at the current estimate on absolute times ``x``.

        Raises:
            RuntimeError: before the first sample.
        """
        return super().predict(self._local(x))

    def predict_cov(self, x: Any, regressors: Any = None) -> np.ndarray:
        """The output variance ``P`` implies at absolute times ``x``.

        Raises:
            RuntimeError: before the first sample.
        """
        return super().predict_cov(self._local(x))

    def coast(
        self, x: Any, *, order: int = 1, regressors: Any = None
    ) -> np.ndarray:
        """Dead-reckon from the newest sample to absolute times ``x``; see
        :func:`dtfit.streaming.coast.coast`.

        Raises:
            RuntimeError: before the first sample.
        """
        return super().coast(self._local(x), order=order)

    def coast_cov(self, x: Any, *, order: int = 1) -> np.ndarray:
        """The variance of :meth:`coast` at absolute times ``x``.

        Raises:
            RuntimeError: before the first sample.
        """
        return super().coast_cov(self._local(x), order=order)
