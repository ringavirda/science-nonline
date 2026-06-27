"""Shared plumbing for the streaming filters (EACFilter / LSIFilter).

Both are Kalman-style recursive estimators that differ only in the *measurement*
(an integrated **area** vs a **Legendre spectrum**) and the measurement-specific
hot path (``partial_fit`` and the drift step). Everything else -- the parameter /
uncertainty read-out, the external-regressor handling, prediction at the current
estimate, and the covariance re-arm hook -- is identical and lives here, so the
two filters share one implementation. Each subclass sets the attributes these
methods read (``p``, ``P``, ``params``, ``regressors``, ``_f``, ``_has_reg``,
``drift_inflation``) in its own ``__init__``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np


class _RecursiveFilter:
    """Mixin base: the measurement-agnostic surface of a streaming filter."""

    # Set by each subclass's ``__init__`` (declared here for the type checkers).
    params: list                 # the model's sympy parameter symbols
    p: np.ndarray                # current parameter estimate
    P: np.ndarray                # parameter (Kalman state) covariance
    regressors: list[str]        # external-regressor channel names ([] if none)
    drift_inflation: float
    _f: Callable[..., Any]       # compiled model callable f(t[, regressors], *p)
    _has_reg: bool

    @property
    def params_(self) -> dict[str, float]:
        """Current parameter estimate as a ``{name: value}`` mapping."""
        return {str(s): float(v) for s, v in zip(self.params, self.p)}

    @property
    def param_cov_(self) -> np.ndarray:
        """Current parameter covariance ``P`` (the running Kalman state
        covariance), shape ``(n_params, n_params)`` -- the streaming analogue of
        :attr:`dtfit.FittingResult.cov`. Its diagonal's square roots are the
        standard errors (:attr:`stderr_`). Large early on, it contracts as the
        parameters become identified and re-inflates on a detected drift."""
        return self.P

    @property
    def stderr_(self) -> dict[str, float]:
        """Per-parameter running standard errors -- ``sqrt`` of the
        :attr:`param_cov_` diagonal -- as a ``{name: value}`` mapping. The online
        twin of :meth:`dtfit.FittingResult.stderr`, giving an uncertainty band on
        the streamed estimate (embedded control, fault detection)."""
        se = np.sqrt(np.clip(np.diag(self.P), 0.0, None))
        return {str(s): float(v) for s, v in zip(self.params, se)}

    def _reg_tuple(self, regressors) -> tuple:
        """Coerce one regressor sample to a tuple ordered like ``self.regressors``."""
        if regressors is None:
            raise ValueError(
                "this model declares external regressors; pass them to partial_fit"
            )
        if isinstance(regressors, Mapping):
            return tuple(float(regressors[r_]) for r_ in self.regressors)
        vals = np.atleast_1d(np.asarray(regressors, dtype=float))
        if vals.size != len(self.regressors):
            raise ValueError(
                f"expected {len(self.regressors)} regressors, got {vals.size}"
            )
        return tuple(float(v) for v in vals)

    def _predict_cols(self, xa: np.ndarray, regressors) -> list[np.ndarray]:
        """Regressor columns broadcast to ``xa`` for :meth:`predict`."""
        if regressors is None:
            raise ValueError("predict() needs regressor values for this model")
        if isinstance(regressors, Mapping):
            return [np.broadcast_to(np.asarray(regressors[r_], float), xa.shape)
                    for r_ in self.regressors]
        arr = np.asarray(regressors, float)
        if arr.ndim == 2 and arr.shape[1] == len(self.regressors):
            return [arr[:, c] for c in range(arr.shape[1])]
        return [np.broadcast_to(arr.reshape(-1)[c], xa.shape)
                for c in range(len(self.regressors))]

    def inflate(self, factor: float | None = None) -> None:
        """Inflate the parameter covariance so new data dominates -- a public
        hook for an *external* maneuver/change detector to re-arm the filter for
        fast re-adaptation without discarding the current estimate.

        Args:
            factor: Covariance multiplier; defaults to ``drift_inflation``.
        """
        self.P = self.P * (self.drift_inflation if factor is None else float(factor))

    def predict(self, x: np.ndarray, regressors=None) -> np.ndarray:
        """Evaluate the model at the current parameter estimate.

        With external regressors, ``regressors`` supplies their value(s) at ``x``
        (a ``{name: array-or-scalar}`` mapping broadcast to ``x``'s shape, or an
        ``(len(x), n_reg)`` array)."""
        xa = np.asarray(x, dtype=float)
        if not self._has_reg:
            return self._f(xa, *self.p)
        cols = self._predict_cols(xa, regressors)
        return self._f(xa, *cols, *self.p)
