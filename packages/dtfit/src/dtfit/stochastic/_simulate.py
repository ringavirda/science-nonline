"""Generative residual simulators for :meth:`StochasticModel.simulate` -- each
draws a fresh path of a detected second-order regime (AR(1), long memory, GARCH)
from the recovered parameter, so re-fitting the simulated path round-trips to the
same regime. Pure numpy."""

from __future__ import annotations

import numpy as np


def _sim_ar1(n: int, phi: float, sigma: float, rng: np.random.Generator,
             *, burn: int = 100) -> np.ndarray:
    """Stationary AR(1) residual ``x_t = phi x_{t-1} + sigma * eps`` (burned in,
    started from the stationary distribution so there is no transient)."""
    phi = float(np.clip(phi, -0.999, 0.999))
    e = rng.standard_normal(n + burn) * float(sigma)
    x = np.empty(n + burn)
    x[0] = e[0] / np.sqrt(max(1e-9, 1.0 - phi ** 2))
    for tt in range(1, n + burn):
        x[tt] = phi * x[tt - 1] + e[tt]
    return x[burn:]


def _sim_long_memory(n: int, hurst: float, sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Long-memory residual as ARFIMA(0, d, 0) with ``d = H - 1/2`` -- the SAME
    process family the long-memory route models, so re-fitting the simulated path
    recovers the long-memory regime (a generic fGn synthesis does not round-trip:
    the detector's AR(1) whitening absorbs it and lands on mean reversion). Built
    from the truncated MA(inf) expansion of ``(1 - B)^{-d}`` driven by innovations
    of std ``sigma``."""
    d = float(np.clip(hurst - 0.5, 0.0, 0.49))
    ntrunc = int(min(2000, max(500, n)))
    psi = np.empty(ntrunc)
    psi[0] = 1.0
    for j in range(1, ntrunc):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    e = rng.standard_normal(n + ntrunc) * float(sigma)
    return np.asarray(np.convolve(e, psi)[ntrunc:ntrunc + n], dtype=float)


def _sim_garch(n: int, persistence: float, sigma: float,
               rng: np.random.Generator, *, burn: int = 200) -> np.ndarray:
    """GARCH(1,1) innovations with the given ``alpha + beta`` persistence and
    unconditional std ``sigma`` (a conventional ``alpha`` / ``beta`` split, since
    only their sum -- the persistence -- is identified by the ACF route)."""
    persistence = float(np.clip(persistence, 0.0, 0.999))
    alpha = max(0.02, 0.1 * persistence)
    beta = max(0.0, persistence - alpha)
    var = float(sigma) ** 2
    omega = var * max(1e-6, 1.0 - persistence)
    z = rng.standard_normal(n + burn)
    s2 = np.empty(n + burn)
    r = np.empty(n + burn)
    s2[0] = var
    r[0] = np.sqrt(s2[0]) * z[0]
    for tt in range(1, n + burn):
        s2[tt] = omega + alpha * r[tt - 1] ** 2 + beta * s2[tt - 1]
        r[tt] = np.sqrt(s2[tt]) * z[tt]
    return r[burn:]
