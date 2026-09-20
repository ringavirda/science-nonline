"""Monte-Carlo scaffolding shared by the experiment suite: seeded draws,
synthetic grids and contamination, the image's restriction efficiency,
replicate summaries, and a process pool for the sweeps that need one.
"""

from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import threadpoolctl
from dtfit.image.bases import Basis, make_basis, u_of

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
    not divide ``len(y)``); whole blocks are drawn without replacement, in a
    random order, until their sizes sum to at least
    ``round(fraction * len(y))``, then the last block drawn is trimmed from
    its tail down to exactly that count, so every earlier run stays whole.
    Two blocks drawn adjacent to each other still merge into one longer run.
    Each replaced sample is displaced by ``magnitude * sigma`` standard-normal
    noise.

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
    order = rng.permutation(n_blocks)
    block_idx = []
    covered = 0
    for b in order:
        run = np.arange(b * burst, min(b * burst + burst, n))
        block_idx.append(run)
        covered += run.size
        if covered >= count:
            break
    overage = covered - count
    if overage > 0:
        # the last block drawn is the one short of covering count exactly;
        # trim its tail so every earlier run stays whole and contiguous.
        block_idx[-1] = block_idx[-1][: block_idx[-1].size - overage]
    idx = np.sort(np.concatenate(block_idx))
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
            Meaningful only when ``rank`` is at least the Jacobian's column
            count: a basis image with fewer independent directions than
            parameters leaves the restricted model unidentified, and
            ``pinv`` then reports it as arbitrarily more precise than
            pointwise least squares, so ``ratio`` can exceed 1. Compare
            ``rank`` against ``jac.shape[1]`` before trusting a value
            above 1.
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
    basis: str | Basis,
    order: int | None = None,
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
            example ``"legendre"`` or ``"block"``), or a :class:`Basis`
            instance evaluated as it is, ``EdgeBlockBasis`` and
            ``SegmentBasis`` included.
        order: Basis order: polynomial degree for Legendre, window count
            for block. Must be ``None`` when ``basis`` is an instance.
        domain: ``(x0, x1)`` the basis variable is mapped from;
            ``(min(x), max(x))`` when ``None``.

    Returns:
        An :class:`Efficiency`.

    Raises:
        ValueError: ``domain`` is degenerate (``x0 == x1``); ``basis`` is
            a name and ``order`` is ``None`` or not a valid order for it;
            or ``basis`` is a :class:`Basis` instance and ``order`` is not
            ``None``.
    """
    if isinstance(basis, Basis) and order is not None:
        raise ValueError(
            f"order must be None when basis is a Basis instance, got "
            f"basis={basis!r} order={order!r}"
        )
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
        A :class:`Summary`. When every replicate is non-finite, ``bias``,
        ``rmse`` and ``percentiles`` are all-NaN of the usual shape and
        ``n_nonfinite`` equals ``len(estimates)``, rather than raising: a
        sweep where a fitter diverges everywhere is the case this function
        exists to report, not to fail on.
    """
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    finite = np.all(np.isfinite(estimates), axis=1)
    n_nonfinite = int(np.sum(~finite))
    clean = estimates[finite]
    n_params = truth.shape[0]
    if clean.shape[0] == 0:
        nan_row = np.full(n_params, np.nan)
        pct = np.full((len(percentiles), n_params), np.nan)
        return Summary(bias=nan_row, rmse=nan_row.copy(), percentiles=pct,
                        n_nonfinite=n_nonfinite)
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
        fn: Picklable callable of one argument, importable by name from a
            module. Under the ``forkserver`` and ``spawn`` start methods a
            worker process imports it, which it cannot do for a function
            defined in a notebook cell, in ``python -c`` or on stdin.
        items: Picklable arguments, consumed once.
        workers: Worker process count; ``ProcessPoolExecutor``'s default
            (the machine's CPU count) when ``None``.

    Returns:
        Results in the same order as ``items``.

    Raises:
        ValueError: ``fn`` is defined in the ``__main__`` of a session with
            no file behind it and the start method is not ``fork``.
    """
    main = sys.modules.get("__main__")
    method = multiprocessing.get_start_method()
    if (
        getattr(fn, "__module__", None) == "__main__"
        and getattr(main, "__file__", None) is None
        and method != "fork"
    ):
        raise ValueError(
            f"{getattr(fn, '__qualname__', fn)!s} is defined in an "
            f"interactive __main__ (a notebook cell, -c or stdin), which a "
            f"worker process cannot import under the {method!r} start "
            f"method; move it into a module"
        )
    items = list(items)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_pool_worker_init
    ) as ex:
        return list(ex.map(fn, items))
