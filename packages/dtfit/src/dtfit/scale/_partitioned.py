"""Map-reduce / partitioned LSI & EAC (promoted from the experiment suite).

The empirical LSI spectrum coefficient is an integral ``∫ y·φ_j dx`` and an EAC
window area is an integral ``∫ y dx``. Integrals are **additive over a partition
of the domain**, so the data-side sufficient statistic can be accumulated chunk
by chunk and summed -- an associative reduce. This turns the batch methods into
**exact one-pass, distributed estimators**: a stream of arbitrary length is
processed in fixed memory (O(order) state), and a partitioned dataset is fitted
by reducing per-partition partial statistics with no re-pass over the data.

* :class:`PartitionedLSI` accumulates the basis-projection integrals per chunk
  (and merges across workers), then solves the usual LSI spectral match once.
* :class:`PartitionedEAC` accumulates per-window areas the same way.

Both require the **global domain fixed up front** (so every chunk projects onto
the same basis); pass it to the constructor.

This adaptation was validated across the big-data and parallel workloads of the
experiment suite and **promoted to the stable API** -- it is re-exported from
``dtfit`` and is the supported way to do one-pass / distributed (map-reduce)
fitting.

* :class:`PartitionedBatchLSI` is the fused, GEMM-batched **multi-channel**
  variant: it combines the volume partition of :class:`PartitionedLSI` with the
  channel batch of :func:`dtfit.project_spectra` into a single one-pass
  estimator (flat ``O(channels x order)`` memory over volume *and* one matmul
  over channels). Promoted after the big-data domain study confirmed the
  GB-scale flat-memory result.
"""

from __future__ import annotations

import numpy as np

from dtfit.types import FittingResult, InitialGuess
from dtfit._core._backend import Backend, resolve_backend
from dtfit._core._spectral import make_basis, solve_spectral


class PartitionedLSI:
    """Streaming / distributed LSI via an additive basis-projection reduce.

    Usage::

        acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
        for x_chunk, y_chunk in stream:      # one pass, fixed memory
            acc.update(x_chunk, y_chunk)
        result = acc.fit(p0=[1.0, 1.0])      # FittingResult

    Workers can each build their own accumulator and be combined with
    :meth:`merge` (the reduce step).
    """

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        domain: tuple[float, float],
        order: int = 6,
        basis: str = "legendre",
    ) -> None:
        self.expr = expr
        self.var = var
        self.basis = make_basis(basis, order, domain)
        self._s = np.zeros(self.basis.n_coef)  # accumulated ∫ y·φ_j integrals
        self._last: tuple[float, float] | None = None  # carried boundary sample
        self.n_samples = 0

    def update(self, x_chunk: np.ndarray, y_chunk: np.ndarray) -> "PartitionedLSI":
        """Fold one chunk's partial projection integrals into the accumulator.

        Consecutive ``update`` calls are made **exactly** additive (equal to a
        single whole-domain projection) by carrying the previous chunk's last
        sample into the next, so the interval connecting two disjoint chunks is
        not dropped by the trapezoid rule. Feed chunks in domain order.
        """
        x = np.asarray(x_chunk, dtype=float)
        y = np.asarray(y_chunk, dtype=float)
        if self._last is not None and x.size:
            x = np.concatenate([[self._last[0]], x])
            y = np.concatenate([[self._last[1]], y])
        if x.size >= 2:
            self._s += self.basis.project_integral(x, y)
            self.n_samples += x_chunk.shape[0] if np.ndim(x_chunk) else 0
            self._last = (float(x[-1]), float(y[-1]))
        return self

    def merge(self, other: "PartitionedLSI") -> "PartitionedLSI":
        """Associative reduce: combine another accumulator's partial sums.

        Exact when the partitions **share boundary samples** (each partition
        includes the sample where the next begins), so every connecting interval
        belongs to exactly one partition; otherwise additive up to one trapezoid
        interval per partition boundary.
        """
        self._s += other._s
        self.n_samples += other.n_samples
        return self

    def spectrum(self) -> np.ndarray:
        """The reduced empirical spectrum (whole-domain coefficients)."""
        return self.basis.integral_to_spectrum(self._s)

    def fit(self, *, p0: InitialGuess = None) -> FittingResult:
        """Solve the LSI spectral match against the accumulated spectrum."""
        guess = None if p0 is None else np.asarray(p0, float)
        return solve_spectral(self.expr, self.var, self.basis, self.spectrum(), p0=guess)


class PartitionedEAC:
    """Streaming / distributed EAC via additive per-window area accumulation.

    The domain is split into ``n_windows`` fixed area windows; each chunk adds
    its contribution to whichever windows it overlaps. The model is then matched
    to the reduced data areas with the same overdetermined least-squares solve
    as batch EAC.
    """

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        domain: tuple[float, float],
        n_windows: int = 8,
    ) -> None:
        self.expr = expr
        self.var = var
        self.x0, self.xn = float(domain[0]), float(domain[1])
        self.m = int(n_windows)
        self.edges = np.linspace(self.x0, self.xn, self.m + 1)
        self._areas = np.zeros(self.m)
        self._first: tuple[float, float] | None = None  # accumulator's first sample
        self._last: tuple[float, float] | None = None    # ...and its last sample
        self.n_samples = 0

    def _add_interval(self, xa: float, ya: float, xb: float, yb: float) -> None:
        """Add the single trapezoid interval ``[xa, xb]`` into the window(s) that
        contain it, matching the per-window masking of :meth:`update`."""
        if xb <= xa:
            return
        cx = np.array([xa, xb])
        cy = np.array([ya, yb])
        for k in range(self.m):
            lo, hi = self.edges[k], self.edges[k + 1]
            if (cx >= lo).all() and (cx <= hi).all():
                self._areas[k] += float(np.trapezoid(cy, cx))
                return

    def update(self, x_chunk: np.ndarray, y_chunk: np.ndarray) -> "PartitionedEAC":
        x = np.asarray(x_chunk, dtype=float)
        y = np.asarray(y_chunk, dtype=float)
        n_orig = x.size
        if self._first is None and x.size:
            self._first = (float(x[0]), float(y[0]))
        if self._last is not None and x.size:
            x = np.concatenate([[self._last[0]], x])
            y = np.concatenate([[self._last[1]], y])
        if x.size:
            self._last = (float(x[-1]), float(y[-1]))
        if x.size < 2:
            return self
        # bin each window's area by trapezoid over the samples falling in it
        for k in range(self.m):
            lo, hi = self.edges[k], self.edges[k + 1]
            mask = (x >= lo) & (x <= hi)
            if mask.sum() >= 2:
                self._areas[k] += float(np.trapezoid(y[mask], x[mask]))
        self.n_samples += n_orig
        return self

    def merge(self, other: "PartitionedEAC") -> "PartitionedEAC":
        """Associative reduce, made **exact** by stitching the partition boundary.

        Summing ``_areas`` alone drops the trapezoid interval connecting one
        partition's last sample to the next's first sample (that interval was
        never integrated by either accumulator) -- a partition-boundary error
        that makes the reduce order-dependent. Here the connecting interval is
        added explicitly (the reduce twin of :meth:`update`'s boundary carry), so
        merging disjoint domain-ordered partitions equals processing them in one
        pass. Partitions must be disjoint; order between the two is inferred.
        """
        left, right = self, other
        if (self._first is not None and other._last is not None
                and other._last[0] <= self._first[0]):
            left, right = other, self  # `other` lies before `self` on the domain
        if left._last is not None and right._first is not None:
            self._add_interval(left._last[0], left._last[1],
                               right._first[0], right._first[1])
        self._areas += other._areas
        self.n_samples += other.n_samples
        # extend this accumulator's boundary span to cover both partitions
        self._first = left._first if left._first is not None else self._first
        self._last = right._last if right._last is not None else self._last
        return self

    def fit(self, *, p0: InitialGuess = None) -> FittingResult:
        """Match the model's window areas to the reduced data areas."""
        import sympy as sp
        from scipy.optimize import least_squares
        from typing import cast

        from dtfit.methods._common import model_params
        from dtfit.methods._common import _covariance

        t = sp.Symbol(self.var)
        f_sym = cast(sp.Expr, sp.sympify(self.expr))
        params = model_params(f_sym, t)
        n = len(params)
        f_func = sp.lambdify((t, *params), f_sym, "numpy")
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        widths = np.diff(self.edges)

        def model_areas(c: np.ndarray) -> np.ndarray:
            # midpoint-rule model area per window (cheap, consistent across calls)
            fv = np.asarray(f_func(centers, *c), dtype=float)
            if fv.ndim == 0:
                fv = np.full_like(centers, float(fv))
            return fv * widths

        def residual(c: np.ndarray) -> np.ndarray:
            return model_areas(c) - self._areas

        guess = np.ones(n) if p0 is None else np.asarray(p0, float)
        sol = least_squares(residual, guess, method="lm")
        coeffs = np.asarray(sol.x, dtype=np.float64)
        cov = _covariance(sol.jac, residual(coeffs), n)
        return FittingResult(coeffs=coeffs, cov=cov,
                             expr=self.expr, var=self.var, names=tuple(str(p) for p in params))


class PartitionedBatchLSI:
    """Fused map-reduce + GEMM-batched LSI for **many channels** in one pass.

    Combines the two big-data levers that :class:`PartitionedLSI` and
    :func:`dtfit.project_spectra` provide *separately*:

    * the **volume** partition of :class:`PartitionedLSI` -- the empirical
      spectrum is an additive integral, so a stream of arbitrary length is
      reduced in fixed ``O(channels x order)`` memory, exact and one-pass;
    * the **channel** batch of :func:`dtfit.project_spectra` -- ``B`` channels
      sharing the sampling grid are projected in a *single* GEMM
      ``S = Dᵀ·(w⊙Y)`` per chunk, dispatched through a pluggable array backend
      (NumPy/BLAS, or cupy/torch on a GPU).

    The fusion is exact because the projection is **linear across channels** and
    **additive over the domain**: each chunk's ``(B, n_coef)`` partial integrals
    are folded into the accumulator, :meth:`merge` reduces accumulators across
    workers/partitions, and :meth:`fit` solves each channel's small spectral
    match. The result is flat memory over volume *and* one matmul (GPU-able) over
    channels.

    Usage::

        acc = PartitionedBatchLSI("a*exp(b*t)", "t", domain=(0, 10),
                                  n_channels=512, order=6, backend="auto")
        for x_chunk, Y_chunk in stream:        # Y_chunk is (n_chunk, 512)
            acc.update(x_chunk, Y_chunk)
        results = acc.fit(p0=[1.0, 1.0])       # list[FittingResult], one/channel
    """

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        domain: tuple[float, float],
        n_channels: int,
        order: int = 6,
        basis: str = "legendre",
        backend: str | Backend = "auto",
        **basis_kwargs: object,
    ) -> None:
        self.expr = expr
        self.var = var
        self.basis = make_basis(basis, order, domain, **basis_kwargs)
        self.backend = (
            backend if isinstance(backend, Backend) else resolve_backend(backend)
        )
        self.n_channels = int(n_channels)
        # accumulated raw integrals s_{c,j} = ∫ y_c·φ_j  (B, n_coef)
        self._s = np.zeros((self.n_channels, self.basis.n_coef))
        self._last_x: float | None = None       # carried boundary sample (time)
        self._last_y: np.ndarray | None = None   # carried boundary row (B,)
        self.n_samples = 0

    def update(self, x_chunk: np.ndarray, Y_chunk: np.ndarray) -> "PartitionedBatchLSI":
        """Fold one chunk's ``B``-channel partial projection into the accumulator.

        ``Y_chunk`` is ``(n_chunk, B)``. As in :class:`PartitionedLSI`, the
        previous chunk's last row is carried into this one so the connecting
        interval is integrated exactly (feed chunks in domain order).
        """
        x = np.asarray(x_chunk, dtype=float)
        Y = np.asarray(Y_chunk)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[1] != self.n_channels:
            raise ValueError(
                f"Y_chunk has {Y.shape[1]} channels but accumulator holds "
                f"{self.n_channels}."
            )
        n_orig = x.shape[0]
        if self._last_x is not None and self._last_y is not None and x.size:
            x = np.concatenate([[self._last_x], x])
            Y = np.concatenate([self._last_y[None, :], Y], axis=0)
        if x.size >= 2:
            self._s += self.basis.project_integral_batched(x, Y, self.backend)
            self.n_samples += n_orig
            self._last_x = float(x[-1])
            self._last_y = np.asarray(Y[-1], dtype=float)
        return self

    def merge(self, other: "PartitionedBatchLSI") -> "PartitionedBatchLSI":
        """Associative reduce: combine another worker's partial integrals."""
        self._s += other._s
        self.n_samples += other.n_samples
        return self

    def spectra(self) -> np.ndarray:
        """Reduced per-channel empirical spectra, shape ``(B, n_coef)``."""
        return self.basis.integral_to_spectrum(self._s)

    def fit(
        self,
        *,
        p0: InitialGuess = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> list[FittingResult]:
        """Solve each channel's LSI spectral match against its reduced spectrum."""
        specs = self.spectra()
        p0a = None if p0 is None else np.asarray(p0, float)
        return [
            solve_spectral(self.expr, self.var, self.basis, specs[i], p0=p0a, bounds=bounds)
            for i in range(specs.shape[0])
        ]
