"""FilterBank -- a parallel array of independent streaming filters.

Many real-time problems are not a single stream but a *bank* of them: the
pseudorange from each GPS satellite, the outputs of a MIMO plant, the channels
of a sensor array, the axes of a trajectory. Each is tracked by its own
:class:`~dtfit.streaming.EDAFilter` /
:class:`~dtfit.streaming.LSIFilter`, and -- because the streams are
independent -- the whole bank fans across CPU cores.

The composition is deliberately thin: a :class:`FilterBank` *holds* K filters
and routes samples to them. The parallel speed-up comes from the compiled
kernels (``dtfit._native``) releasing the GIL on their hot loops, so worker
**threads** updating different filters run their integral work concurrently
without pickling (filters carry compiled SymPy callables and are not sent to
processes). This is the streaming counterpart of :func:`dtfit.fit_many`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2

from ._eda import EDAFilter

__all__ = ["FilterBank", "FusedChiSquareDetector"]


class FilterBank:
    """A bank of K independent streaming filters, updated in lockstep.

    Construct from explicit filters or, more commonly, from a model via
    :meth:`from_model`. Sample routing: at each step every stream gets its own
    ``y`` (and, optionally, its own ``t``).
    """

    def __init__(self, filters: Sequence[Any]) -> None:
        self.filters: list[Any] = list(filters)
        if not self.filters:
            raise ValueError("FilterBank needs at least one filter.")

    @classmethod
    def from_model(
        cls,
        expr: str,
        var: str,
        n_streams: int,
        *,
        filter_cls: type = EDAFilter,
        **kwargs: Any,
    ) -> "FilterBank":
        """Build ``n_streams`` identically-configured filters for one model.

        Args:
            expr, var: Model expression and main variable (per stream).
            n_streams: Number of parallel streams (filters) in the bank.
            filter_cls: ``EDAFilter`` (default) or
                ``LSIFilter``.
            **kwargs: Forwarded to each filter's constructor.
        """
        return cls([filter_cls(expr, var, **kwargs) for _ in range(n_streams)])

    def __len__(self) -> int:
        return len(self.filters)

    def __getitem__(self, i: int) -> Any:
        return self.filters[i]

    # per-step ingestion (one sample per stream)
    def partial_fit(
        self,
        t: float | np.ndarray,
        y: np.ndarray,
        *,
        n_jobs: int = 1,
    ) -> "FilterBank":
        """Ingest one sample per stream and update every filter in place.

        Args:
            t: Shared scalar time, or a per-stream array of length K.
            y: Per-stream observations, length K.
            n_jobs: Threads to fan the K updates across (``1`` = serial). Uses a
                thread pool so the GIL-released kernels overlap; for cheap
                per-step work serial is usually fastest -- threading wins when K
                and the window are large (see :meth:`run`).
        """
        y = np.asarray(y, dtype=float)
        tv = np.full(len(self.filters), float(t)) if np.ndim(t) == 0 else np.asarray(t, float)
        if n_jobs == 1:
            for flt, ti, yi in zip(self.filters, tv, y):
                flt.partial_fit(float(ti), float(yi))
            return self
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(
                lambda k: self.filters[k].partial_fit(float(tv[k]), float(y[k])),
                range(len(self.filters)),
            ))
        return self

    # whole-stream driver (parallel across streams, no per-step barrier)
    def run(
        self,
        t_seq: np.ndarray,
        Y: np.ndarray,
        *,
        n_jobs: int = 1,
        track: bool = False,
    ) -> dict[str, Any]:
        """Drive every stream over a whole block of samples.

        Streams are independent, so each worker thread takes a disjoint subset
        of streams and runs it to completion -- no per-step synchronization,
        which is what makes the bank scale. This is the throughput primitive
        used by the parallel-scaling experiment.

        Args:
            t_seq: Time stamps, shape ``(n_steps,)`` (shared by all streams).
            Y: Observations, shape ``(n_steps, K)`` -- column k feeds filter k.
            n_jobs: Worker threads (``1`` = serial).
            track: If True, also return the per-stream, per-step prediction of
                the current sample (``(n_steps, K)``); costs O(n_steps*K) memory.

        Returns:
            ``params``: ``(K, n_params)`` final estimates; ``n_drifts``:
            per-stream drift counts; ``track`` (optional): tracking history.
        """
        t_seq = np.asarray(t_seq, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n_steps, K = Y.shape
        if K != len(self.filters):
            raise ValueError(f"Y has {K} columns but bank holds {len(self.filters)} filters.")
        track_hist = np.full((n_steps, K), np.nan) if track else None
        drifts = np.zeros(K, dtype=int)

        def drive(k: int) -> None:
            flt = self.filters[k]
            col = Y[:, k]
            for s in range(n_steps):
                flt.partial_fit(t_seq[s], col[s])
                if getattr(flt, "drift_flag_", False):
                    drifts[k] += 1
                if track_hist is not None and len(getattr(flt, "_t", [1])) > 0:
                    track_hist[s, k] = float(flt.predict(np.array([t_seq[s]]))[0])

        if n_jobs == 1:
            for k in range(K):
                drive(k)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                list(ex.map(drive, range(K)))

        out: dict[str, Any] = {
            "params": self.params_array(),
            "n_drifts": drifts,
        }
        if track_hist is not None:
            out["track"] = track_hist
        return out

    # readout
    def params_array(self) -> np.ndarray:
        """Current estimates as ``(K, n_params)``."""
        return np.array([flt.p for flt in self.filters], dtype=float)

    @property
    def params_(self) -> list[dict[str, float]]:
        """Per-stream ``{name: value}`` parameter mappings."""
        return [flt.params_ for flt in self.filters]

    @property
    def drift_flags_(self) -> np.ndarray:
        """Per-stream drift flag from the most recent update, shape ``(K,)``."""
        return np.array(
            [bool(getattr(flt, "drift_flag_", False)) for flt in self.filters]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Per-stream prediction. Returns ``(K,)`` for scalar-like ``x`` else
        ``(K, len(x))``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        preds = np.array([np.asarray(flt.predict(x), dtype=float) for flt in self.filters])
        return preds[:, 0] if x.size == 1 else preds

    def fused_detector(self, **kwargs: Any) -> "FusedChiSquareDetector":
        """A :class:`FusedChiSquareDetector` driving this bank (see its docs)."""
        return FusedChiSquareDetector(self, **kwargs)


class FusedChiSquareDetector:
    """Pool a :class:`FilterBank`'s one-step innovations into a fused fault test.

    A fault that moves **every** stream (a structural change shared across axes:
    a damping fault on all axes of an oscillator, a regime shift in a sensor
    array) leaves only a weak signature in any single stream's innovation but a
    strong one in the *sum* across streams. This detector normalises each
    filter's one-step residual (``last_residual_``) by an online EWMA estimate of
    its variance and sums the squares into a ``chi2(K)`` statistic; when it
    exceeds the ``alpha``-level threshold it flags a fault and (optionally)
    re-arms each filter via :meth:`~dtfit.EDAFilter.inflate` so the bank
    re-adapts quickly. Validated in the embedded-control domain study (a 3-axis
    damping fault flagged within a window at zero false alarms, where the pooled
    chi2(3) has far higher SNR than any single axis).

    Usage::

        bank = FilterBank.from_model(model, "t", n_axes,
                                     filter_cls=LSIFilter, ...)
        det = bank.fused_detector(alpha=1e-4, inflate=4.0)
        for i, (t, y) in enumerate(stream):     # y is length-K
            if det.update(t, y):
                handle_fault(i, det.statistic_)

    Args:
        bank: The :class:`FilterBank` to drive (its filters must expose
            ``last_residual_``, ``W`` and :meth:`inflate` -- both stock filters
            do).
        alpha: Per-step false-alarm probability; the threshold is
            ``chi2.ppf(1 - alpha, df=K)``.
        inflate: Covariance re-arm factor applied to every filter on a detection
            (``<= 1`` disables the re-arm; the flag is still raised).
        ewma: Decay for the per-stream innovation-variance estimate.
        warmup: Steps to wait before detecting (defaults to ``3 * window``, so
            the EWMA variance and the filters have settled).
        cooldown: Steps to suppress detection after a flag (defaults to one
            ``window``, so a single fault is not re-flagged every step).
    """

    def __init__(
        self,
        bank: FilterBank,
        *,
        alpha: float = 1e-4,
        inflate: float = 4.0,
        ewma: float = 0.9,
        warmup: int | None = None,
        cooldown: int | None = None,
    ) -> None:
        self.bank = bank
        self.k = len(bank.filters)
        self.threshold_ = float(chi2.ppf(1.0 - alpha, df=self.k))
        self.inflate_factor = float(inflate)
        self.ewma = float(ewma)
        w = int(getattr(bank.filters[0], "W", 1))
        self._warmup = 3 * w if warmup is None else int(warmup)
        self._cooldown_len = w if cooldown is None else int(cooldown)
        self._scale2 = np.zeros(self.k)
        self._step = -1   # raw stream index of the current sample
        self._seen = 0    # number of steps with a full (finite-residual) window
        self._cool = 0
        self.statistic_ = float("nan")
        self.flag_ = False
        self.flags_: list[int] = []

    def update(self, t: float | np.ndarray, y: np.ndarray) -> bool:
        """Ingest one sample per stream and test for a fused fault.

        Updates the bank in place, then returns ``True`` iff this step raises a
        fault flag. ``flags_`` records the (zero-based) **stream** step indices
        that fired (counting every :meth:`update` call, including the warm-up
        steps before the filters' windows fill).
        """
        self._step += 1
        self.bank.partial_fit(t, y)
        self.flag_ = False
        res = np.array(
            [getattr(f, "last_residual_", np.nan) for f in self.bank.filters]
        )
        if not np.all(np.isfinite(res)):
            return False  # windows not yet full
        idx = self._step
        self._seen += 1
        z2 = np.zeros(self.k)
        nz = self._scale2 > 0
        z2[nz] = res[nz] ** 2 / self._scale2[nz]
        self._scale2 = self.ewma * self._scale2 + (1.0 - self.ewma) * res ** 2
        self.statistic_ = float(z2.sum())
        if self._cool > 0:
            self._cool -= 1
            return False
        if self._seen < self._warmup:
            return False
        if self.statistic_ > self.threshold_:
            if self.inflate_factor > 1.0:
                for f in self.bank.filters:
                    f.inflate(self.inflate_factor)
            self.flag_ = True
            self.flags_.append(idx)
            self._scale2[:] = 0.0
            self._cool = self._cooldown_len
            return True
        return False

    @property
    def n_flags_(self) -> int:
        """Number of fault flags raised so far."""
        return len(self.flags_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Per-stream prediction from the underlying bank."""
        return self.bank.predict(x)
