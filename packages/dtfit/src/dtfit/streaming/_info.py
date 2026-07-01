"""Information-form (inverse-covariance) recursive estimator.

The covariance-form Kalman update the streaming filters run maintains ``P`` and,
per step, inverts the innovation covariance ``S`` (dimension = measurement size).
The **information form** maintains the inverse ``Y = P^-1`` (the *information
matrix*) and ``yv = P^-1 p`` (the *information vector*) instead, which flips three
properties that matter for the embedded / sensor-fusion target:

* **the measurement update is purely additive** -- ``Y += Hᵀ R⁻¹ H`` and
  ``yv += Hᵀ R⁻¹ z`` -- so no matrix inverse is needed to *absorb* a measurement,
  and independent estimators **fuse by adding information** (associative,
  order-independent, bit-faithful);
* **you invert the smaller matrix**: a readout solves the ``n_params × n_params``
  system ``Y θ = yv`` once, rather than inverting an ``m × m`` innovation
  covariance every step (a win whenever the measurement dimension ``m`` exceeds
  the state dimension ``n`` -- the common LSI case, order+1 vs 2-3 params);
* it is **fixed-point friendlier** -- adding information never suffers the
  covariance-collapse conditioning that ``(I - K H) P`` does.

This module provides the linear-Gaussian primitive (recursive least squares in
information form, with an optional forgetting factor) -- the "on-MCU info-form"
building block. The nonlinear dtfit filters linearize to exactly this update each
step; this class is the standalone, fusion-oriented core.
"""

from __future__ import annotations

import numpy as np


class InformationFilter:
    """Recursive linear-Gaussian estimator in information (inverse-covariance) form.

    Estimates ``theta`` in the linear measurement model ``z = h . theta + noise``
    (noise variance ``r``) by accumulating the information matrix ``Y`` and vector
    ``yv``. Absorbing a measurement is an addition (no inverse); the estimate is
    read out by one small solve.

    Usage::

        f = InformationFilter(n_params=2)
        for h, z in stream:                 # h: (2,) row, z: scalar
            f.partial_fit(h, z, r=0.04)
        theta = f.theta_                    # current estimate
        cov = f.cov_                        # its covariance (P = Y^-1)

    Fuse two independent estimators (e.g. per-sensor or per-partition) by adding
    their information -- exactly, in any order::

        fused = a.fuse(b)                   # associative & commutative

    Args:
        n_params: State dimension.
        prior_precision: Diagonal of the initial information matrix ``Y0 =
            prior_precision * I`` (a weak prior; ``0`` is an uninformative start
            but leaves ``Y`` singular until enough measurements arrive).
        forgetting: Exponential forgetting factor in ``(0, 1]`` (``1`` = no
            forgetting). Each step down-weights the accumulated information by this
            factor before adding the new measurement, so the estimator tracks
            slowly-varying parameters.
    """

    def __init__(
        self,
        n_params: int,
        *,
        prior_precision: float = 1e-6,
        forgetting: float = 1.0,
    ) -> None:
        self.n = int(n_params)
        if self.n < 1:
            raise ValueError("n_params must be >= 1")
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("forgetting must be in (0, 1]")
        self.forgetting = float(forgetting)
        self._prior = float(prior_precision)
        self.Y = np.eye(self.n) * self._prior   # information matrix (P^-1)
        self.yv = np.zeros(self.n)              # information vector (P^-1 p)
        self.n_updates = 0

    def partial_fit(self, h, z, r: float = 1.0) -> "InformationFilter":
        """Absorb one measurement ``z = h . theta + noise`` (noise variance ``r``).

        ``h`` is a length-``n`` row for a scalar ``z``, or an ``(m, n)`` matrix for
        a vector measurement ``z`` of length ``m`` (then ``r`` may be a scalar or a
        length-``m`` per-component variance). The update is additive:
        ``Y += Hᵀ R⁻¹ H``, ``yv += Hᵀ R⁻¹ z`` -- no inverse.
        """
        H = np.atleast_2d(np.asarray(h, dtype=float))
        zz = np.atleast_1d(np.asarray(z, dtype=float))
        if H.shape[1] != self.n:
            raise ValueError(
                f"h has {H.shape[1]} columns but the state has {self.n} params."
            )
        if H.shape[0] != zz.size:
            raise ValueError(
                f"got {H.shape[0]} measurement rows but {zz.size} values."
            )
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        if rr.size == 1:
            rr = np.full(zz.size, float(rr.reshape(-1)[0]))
        r_inv = 1.0 / rr
        if self.forgetting < 1.0:
            self.Y *= self.forgetting
            self.yv *= self.forgetting
        # Hᵀ R⁻¹ H and Hᵀ R⁻¹ z, no inverse of an m x m innovation covariance.
        HtRinv = H.T * r_inv                     # (n, m)
        self.Y += HtRinv @ H
        self.yv += HtRinv @ zz
        self.n_updates += 1
        return self

    @property
    def cov_(self) -> np.ndarray:
        """Parameter covariance ``P = Y^-1`` (inverts the small ``n x n`` matrix)."""
        return np.linalg.inv(self.Y)

    @property
    def theta_(self) -> np.ndarray:
        """Current estimate, from the single small solve ``Y theta = yv``."""
        return np.linalg.solve(self.Y, self.yv)

    @property
    def p(self) -> np.ndarray:
        """Alias of :attr:`theta_` (covariance-filter naming)."""
        return self.theta_

    @property
    def P(self) -> np.ndarray:
        """Alias of :attr:`cov_` (covariance-filter naming)."""
        return self.cov_

    def fuse(self, other: "InformationFilter") -> "InformationFilter":
        """Combine another estimator's information into this one (in place).

        Information is **additive**, so fusing independent estimators is an exact,
        associative, commutative reduce -- the property that makes the information
        form the natural sensor-fusion / map-reduce state. The shared prior is
        subtracted once so it is not double-counted. Returns ``self``.
        """
        if other.n != self.n:
            raise ValueError("cannot fuse filters with different state sizes")
        self.Y = self.Y + other.Y - np.eye(self.n) * self._prior
        self.yv = self.yv + other.yv
        self.n_updates += other.n_updates
        return self
