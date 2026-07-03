"""Generative residual simulators for :meth:`StochasticModel.simulate` -- each
draws a fresh path of a detected second-order regime (AR(1), long memory, GARCH)
from the recovered parameter, so re-fitting the simulated path round-trips to the
same regime. Pure numpy."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import lfilter

# A standardized (unit-variance) innovation draw: ``noise(rng, size) -> ndarray``.
Noise = Callable[[np.random.Generator, int], np.ndarray]


def make_innovations(dist: str = "normal", df: float = 7.0) -> Noise:
    """Return a standardized innovation sampler ``noise(rng, size)``.

    ``dist="normal"`` draws unit-variance Gaussian innovations; ``dist="t"`` draws
    Student-t with ``df`` degrees of freedom, **rescaled to unit variance** (a
    heavy-tailed innovation for fat-tailed series -- financial returns -- so a
    simulated path / forecast interval reflects the real tail risk instead of
    understating it with a Gaussian). ``df`` must exceed 2 for finite variance.
    """
    if dist == "normal":
        return lambda rng, size: rng.standard_normal(size)
    if dist == "t":
        scale = float(np.sqrt((df - 2.0) / df)) if df > 2.0 else 1.0
        return lambda rng, size: rng.standard_t(df, size) * scale
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def _sim_ar1(n: int, phi: float, sigma: float, rng: np.random.Generator,
             *, burn: int = 100, noise: Noise | None = None) -> np.ndarray:
    """Stationary AR(1) residual ``x_t = phi x_{t-1} + sigma * eps`` (burned in,
    started from the stationary distribution so there is no transient)."""
    if noise is None:
        noise = make_innovations()
    phi = float(np.clip(phi, -0.999, 0.999))
    e = noise(rng, n + burn) * float(sigma)
    # AR(1) is a linear recurrence x_t = phi x_{t-1} + e_t, so a single IIR pass
    # (scipy.signal.lfilter) replaces the Python loop. The initial filter state is
    # chosen so x_0 = e_0 / sqrt(1 - phi^2) -- the stationary start (no transient)
    # the loop used -- making the vectorized path bit-identical to it.
    denom = np.sqrt(max(1e-9, 1.0 - phi ** 2))
    zi = [float(e[0]) * (1.0 / denom - 1.0)]
    x, _ = lfilter([1.0], [1.0, -phi], e, zi=zi)
    return x[burn:]


def _sim_long_memory(n: int, hurst: float, sigma: float,
                     rng: np.random.Generator,
                     *, noise: Noise | None = None) -> np.ndarray:
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
    # Normalize the MA(inf) weights to unit L2 norm so the *realized* marginal
    # std of the output is exactly ``sigma`` (Var[conv(e, psi)] = sigma^2 *
    # sum(psi^2)). Without this the output std is sigma * sqrt(sum psi^2) > sigma
    # for d>0 and drifts with the truncation length, so the round-trip recovers
    # an inflated sigma. Scaling all psi by a constant preserves the long-memory
    # ACF shape (hence the regime), only fixing the amplitude.
    psi /= np.sqrt(float(np.sum(psi ** 2)))
    if noise is None:
        noise = make_innovations()
    e = noise(rng, n + ntrunc) * float(sigma)
    return np.asarray(np.convolve(e, psi)[ntrunc:ntrunc + n], dtype=float)


def _sim_garch(n: int, persistence: float, sigma: float,
               rng: np.random.Generator, *, burn: int = 200,
               noise: Noise | None = None) -> np.ndarray:
    """GARCH(1,1) innovations with the given ``alpha + beta`` persistence and
    unconditional std ``sigma`` (a conventional ``alpha`` / ``beta`` split, since
    only their sum -- the persistence -- is identified by the ACF route)."""
    if noise is None:
        noise = make_innovations()
    persistence = float(np.clip(persistence, 0.0, 0.999))
    alpha = max(0.02, 0.1 * persistence)
    beta = max(0.0, persistence - alpha)
    var = float(sigma) ** 2
    omega = var * max(1e-6, 1.0 - persistence)
    z = noise(rng, n + burn)
    s2 = np.empty(n + burn)
    r = np.empty(n + burn)
    s2[0] = var
    r[0] = np.sqrt(s2[0]) * z[0]
    for tt in range(1, n + burn):
        s2[tt] = omega + alpha * r[tt - 1] ** 2 + beta * s2[tt - 1]
        r[tt] = np.sqrt(s2[tt]) * z[tt]
    return r[burn:]
