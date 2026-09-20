"""Monte-Carlo scaffolding shared by the experiment suite: seeded draws,
synthetic grids and contamination, the image's restriction efficiency,
replicate summaries, and a process pool for the sweeps that need one.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import threadpoolctl
from dtfit.image.bases import make_basis, u_of

# Held by _pool_worker_init so the BLAS thread cap it applies outlives the
# initializer call for the life of the worker process.
_POOL_THREAD_LIMIT = None


def generators(seed: int, count: int) -> list[np.random.Generator]:
    """``count`` independent random generators derived from one seed.

    Uses ``numpy.random.SeedSequence(seed).spawn(count)``, so replicate ``i``
    is reproducible on its own and independent of how many other replicates
    are drawn alongside it.

    Args:
        seed: Integer seed for the whole batch.
        count: Number of generators to spawn, at least 0.

    Returns:
        A list of ``count`` ``numpy.random.Generator`` instances.
    """
    seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in seq.spawn(count)]


def grid(
    kind: str,
    n: int,
    *,
    span: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A sample grid on ``[0, span]``, the three shapes of
    ``experiments/evolution.py``.

    Args:
        kind: ``"uniform"`` (evenly spaced), ``"clustered"`` (half the
            points drawn uniformly in the first fifth of the span, the rest
            in the remaining four fifths) or ``"random"`` (all points drawn
            uniformly over the span).
        n: Number of points, at least 2.
        span: Length of the interval in the grid's own units, positive.
        rng: Generator for the clustered and random draws; a fresh default
            generator when ``None``. Unused for ``"uniform"``.

    Returns:
        A sorted array of ``n`` floats with both endpoints pinned to
        ``0.0`` and ``span``.

    Raises:
        ValueError: ``kind`` is none of ``"uniform"``, ``"clustered"`` or
            ``"random"``.
    """
    if kind == "uniform":
        return np.linspace(0.0, span, n)
    if rng is None:
        rng = np.random.default_rng()
    if kind == "clustered":
        a = rng.uniform(0.0, 0.2 * span, n // 2)
        b = rng.uniform(0.2 * span, span, n - n // 2)
        x = np.sort(np.concatenate([a, b]))
    elif kind == "random":
        x = np.sort(rng.uniform(0.0, span, n))
    else:
        raise ValueError(f"unknown grid kind {kind!r}")
    x[0], x[-1] = 0.0, span
    return x


def noisy(
    clean: np.ndarray, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Additive Gaussian noise scaled to the signal's own range.

    ``sigma = noise * numpy.ptp(clean)``, the convention of
    ``experiments/evolution.py``; a flat ``clean`` (``ptp == 0``) falls back
    to ``sigma = noise`` so the noise level never collapses to zero.

    Args:
        clean: Noise-free signal, any shape.
        noise: Relative noise level, dimensionless, in ``[0, inf)``.
        rng: Generator drawing the noise.

    Returns:
        ``(y, sigma)``: the noisy signal, same shape as ``clean``, and the
        noise standard deviation actually used.
    """
    clean = np.asarray(clean, dtype=float)
    amp = float(np.ptp(clean)) or 1.0
    sigma = noise * amp
    y = clean + sigma * rng.standard_normal(clean.shape)
    return y, sigma


def contaminate(
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    fraction: float,
    magnitude: float = 10.0,
    sigma: float | None = None,
    burst: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a fraction of ``y`` with outliers in contiguous runs.

    ``y`` is split into ``ceil(len(y) / burst)`` contiguous, non-overlapping
    blocks of ``burst`` samples (the last block shorter when ``burst`` does
    not divide ``len(y)``); whole blocks are drawn without replacement until
    at least ``round(fraction * len(y))`` samples are covered, then trimmed
    back to exactly that count. Each replaced sample is displaced by
    ``magnitude * sigma`` standard-normal noise.

    Args:
        y: Signal to contaminate, 1-D; not modified in place.
        rng: Generator choosing the blocks and the displacement.
        fraction: Fraction of samples to replace, in ``[0, 1]``.
        magnitude: Multiple of ``sigma`` the outlier displacement carries,
            dimensionless.
        sigma: Displacement scale; the sample standard deviation of ``y``
            when ``None``.
        burst: Run length of a contiguous outlier block, at least 1.

    Returns:
        ``(y, idx)``: a contaminated copy of ``y`` and the sorted integer
        indices that were replaced (empty when ``fraction`` rounds to 0).
    """
    y = np.array(y, dtype=float, copy=True)
    n = y.shape[0]
    count = int(round(fraction * n))
    if count <= 0:
        return y, np.array([], dtype=int)
    if sigma is None:
        sigma = float(np.std(y))
    n_blocks = -(-n // burst)
    n_runs = -(-count // burst)
    block_ids = rng.choice(n_blocks, size=n_runs, replace=False)
    idx = np.concatenate(
        [np.arange(b * burst, min(b * burst + burst, n)) for b in block_ids]
    )
    idx = np.sort(idx)
    if idx.size > count:
        idx = np.sort(rng.choice(idx, size=count, replace=False))
    y[idx] = y[idx] + magnitude * sigma * rng.standard_normal(idx.size)
    return y, idx


def numeric_jacobian(
    f: Callable[..., np.ndarray],
    x: np.ndarray,
    params: Sequence[float],
    *,
    step: float = 1e-6,
) -> np.ndarray:
    """Central-difference Jacobian of ``f(x, *params)`` with respect to
    ``params``.

    Args:
        f: Model called as ``f(x, *params) -> array of shape x.shape``.
        x: Sample positions, any shape.
        params: Parameter values at which to differentiate.
        step: Relative finite-difference step, positive; the absolute step
            per parameter is ``step * max(1, abs(params[i]))``.

    Returns:
        Array of shape ``(x.size, len(params))``, column ``i`` the
        derivative of ``f`` with respect to ``params[i]``.
    """
    params = np.asarray(params, dtype=float)
    cols = []
    for i in range(params.size):
        d = np.zeros_like(params)
        d[i] = step * max(1.0, abs(params[i]))
        hi = f(x, *(params + d))
        lo = f(x, *(params - d))
        cols.append((np.asarray(hi, dtype=float) -
                      np.asarray(lo, dtype=float)) / (2.0 * d[i]))
    return np.stack(cols, axis=1)


@dataclass(frozen=True)
class Efficiency:
    """Efficiency of an estimator restricted to a basis image against
    unrestricted least squares, on the same Jacobian.

    Attributes:
        ratio: Per-parameter efficiency, ``diag((J^T J)^-1) /
            diag(pinv(J^T P J))``; 1.0 where the basis spans the samples
            exactly, below 1.0 where the restriction costs precision.
        cond: Condition number of the basis's Gram matrix over the samples,
            at least 1.
        rank: Numerical rank of the basis on the sample grid, at most
            ``n_coef``; a deficient restriction hides in ``pinv`` without
            this.
        n_coef: Number of basis coefficients retained after dropping
            all-zero columns.
    """

    ratio: np.ndarray
    cond: float
    rank: int
    n_coef: int


def image_efficiency(
    jac: np.ndarray,
    x: np.ndarray,
    *,
    basis: str,
    order: int,
    domain: tuple[float, float] | None = None,
) -> Efficiency:
    """Efficiency of the image-restricted estimator, exact and noise-free.

    For a model with Jacobian ``jac`` sampled at ``x``, the estimator
    restricted to the span of ``basis`` has asymptotic covariance
    ``s2 * (J^T P J)^-1`` with ``P`` the projector onto the basis's span
    over the samples, against ``s2 * (J^T J)^-1`` for pointwise least
    squares; ``ratio`` is the diagonal quotient.

    Args:
        jac: Jacobian of the model at the samples, shape
            ``(len(x), n_params)``.
        x: Sample positions matching ``jac``'s rows.
        basis: Name accepted by ``dtfit.image.bases.make_basis`` (for
            example ``"legendre"`` or ``"block"``).
        order: Basis order: polynomial degree for Legendre, window count
            for block.
        domain: ``(x0, x1)`` the basis variable is mapped from;
            ``(min(x), max(x))`` when ``None``.

    Returns:
        An :class:`Efficiency`.

    Raises:
        ValueError: ``domain`` is degenerate (``x0 == x1``), or ``order``
            is not a valid order for ``basis``.
    """
    if domain is None:
        domain = (float(np.min(x)), float(np.max(x)))
    phi = make_basis(basis, order).evaluate(u_of(x, *domain))
    keep = phi.any(axis=0)
    phi = phi[:, keep]
    q, r = np.linalg.qr(phi)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-10 * np.abs(r).max()))
    cond = float(np.linalg.cond(phi.T @ phi))
    restricted_jac = q @ (q.T @ jac)
    full = np.linalg.inv(jac.T @ jac)
    restr = np.linalg.pinv(jac.T @ restricted_jac)
    ratio = np.diag(full) / np.diag(restr)
    return Efficiency(ratio=ratio, cond=cond, rank=rank, n_coef=phi.shape[1])


@dataclass(frozen=True)
class Summary:
    """Bias, RMSE and percentiles of a Monte-Carlo batch of estimates.

    Attributes:
        bias: Per-parameter mean of ``estimate - truth`` over the finite
            replicates.
        rmse: Per-parameter root-mean-square of ``estimate - truth`` over
            the finite replicates.
        percentiles: Array of shape ``(len(percentiles), n_params)``, the
            requested percentiles of the finite replicates.
        n_nonfinite: Count of replicates carrying a non-finite (NaN or
            infinite) value in any parameter, excluded from the statistics
            above rather than silently dropped.
    """

    bias: np.ndarray
    rmse: np.ndarray
    percentiles: np.ndarray
    n_nonfinite: int


def summarize(
    estimates: np.ndarray,
    truth: np.ndarray,
    *,
    percentiles: Sequence[float] = (5, 50, 95),
) -> Summary:
    """Summarize a batch of Monte-Carlo parameter estimates against the
    truth that generated them.

    Args:
        estimates: Array of shape ``(n_replicates, n_params)``.
        truth: Array of shape ``(n_params,)``.
        percentiles: Percentile ranks in ``[0, 100]`` computed over the
            finite replicates.

    Returns:
        A :class:`Summary`.
    """
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    finite = np.all(np.isfinite(estimates), axis=1)
    n_nonfinite = int(np.sum(~finite))
    clean = estimates[finite]
    err = clean - truth
    bias = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    pct = np.percentile(clean, percentiles, axis=0)
    return Summary(bias=bias, rmse=rmse, percentiles=pct,
                    n_nonfinite=n_nonfinite)


def _pool_worker_init() -> None:
    global _POOL_THREAD_LIMIT
    _POOL_THREAD_LIMIT = threadpoolctl.threadpool_limits(
        limits=1, user_api="blas"
    )


def pool_map(
    fn: Callable[[object], object],
    items: Iterable[object],
    *,
    workers: int | None = None,
) -> list[object]:
    """Run ``fn`` over ``items`` in a process pool, one BLAS thread a
    worker.

    Each worker process calls ``threadpoolctl.threadpool_limits(limits=1,
    user_api="blas")`` once at start-up and keeps the limit for its whole
    life, so a pool of ``workers`` processes never oversubscribes the BLAS
    threads the caller's own process also uses.

    Args:
        fn: Picklable callable of one argument.
        items: Picklable arguments, consumed once.
        workers: Worker process count; ``ProcessPoolExecutor``'s default
            (the machine's CPU count) when ``None``.

    Returns:
        Results in the same order as ``items``.
    """
    items = list(items)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_pool_worker_init
    ) as ex:
        return list(ex.map(fn, items))
