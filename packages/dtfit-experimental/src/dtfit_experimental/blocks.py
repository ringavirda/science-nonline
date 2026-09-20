"""Block windows placed where the data ask for them.

Three adaptations in trial of dtfit's block (EAC) basis, which is equal
windows only:

* :class:`EdgeBlockBasis`, a block basis with explicit window edges, so a
  window boundary can be put on a known epoch (an equipment change, an
  earthquake, a tariff date) and the discontinuity there costs the fit
  nothing.
* :func:`aggregated_image` and :func:`fit_aggregated`, for data that arrive
  as one total per window and never as samples: the window image is then the
  data itself and the equal-areas fit is the correct least squares.
* :func:`fit_aligned` with :func:`detect_jumps` and :func:`coarsen`, which
  find the epochs in the residual of a fine window image and re-cut the
  windows onto them, leaving the windows that hold an epoch out of the fit.

They are experimental: measured on synthetic series, on NGL GPS records and
on NOAA hourly temperature, not promoted into ``dtfit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

import dtfit
from dtfit._input import normalize_bounds, normalize_p0, resolve_model
from dtfit.image import Grid, Image, Original
from dtfit.image.bases import Basis, BlockBasis, u_of
from dtfit.types import FittingResult

_END_TOL = 1e-12


class EdgeBlockBasis(Basis):
    """Indicators of windows with explicit edges on the unit interval.

    The block basis of :mod:`dtfit` with the equal-window constraint
    dropped: the windows are given by their edges and any of them may be
    left out of the basis. ``order`` and ``n_coef`` are the number of kept
    windows.

    Args:
        edges: Edges on the unit variable ``u``, strictly increasing, of
            length ``K + 1`` for ``K`` intervals; ``edges[0]`` must be
            ``-1`` and ``edges[-1]`` ``+1`` to an absolute tolerance of
            1e-12.
        keep: Boolean mask of length ``K`` selecting the intervals that
            become basis functions; ``None`` keeps all of them. At least
            one interval must be kept.

    Raises:
        ValueError: fewer than two edges; edges not strictly increasing or
            not ending at ``-1`` and ``+1``; ``keep`` of the wrong length;
            no interval kept.
    """

    name = "edge_block"

    def __init__(
        self, edges: Any, keep: Any = None
    ) -> None:
        e = np.asarray(edges, dtype=float).ravel()
        if e.size < 2:
            raise ValueError(
                f"an edge_block basis needs at least 2 edges, got {e.size}"
            )
        if not np.all(np.diff(e) > 0.0):
            raise ValueError("edges must be strictly increasing")
        if abs(e[0] + 1.0) > _END_TOL or abs(e[-1] - 1.0) > _END_TOL:
            raise ValueError(
                "edges on the unit variable must run from -1 to +1, got "
                f"({e[0]!r}, {e[-1]!r})"
            )
        if keep is None:
            k = np.ones(e.size - 1, dtype=bool)
        else:
            k = np.asarray(keep, dtype=bool).ravel()
            if k.size != e.size - 1:
                raise ValueError(
                    f"keep must have one entry per interval "
                    f"({e.size - 1}), got {k.size}"
                )
        if not k.any():
            raise ValueError("at least one interval must be kept")
        self.edges = e
        self.keep = k
        self.order = int(k.sum())

    @property
    def n_coef(self) -> int:
        return self.order

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Indicators of the kept windows at ``u``, ``(len(u), n_coef)``.

        The windows are half-open ``[lo, hi)`` with the last one closed at
        ``+1``; ``u`` outside ``[-1, 1]`` falls into the nearest end window
        (clamped, not extrapolated), and a ``u`` inside an interval that is
        not kept gives an all-zero row.
        """
        u = np.asarray(u, dtype=float).ravel()
        idx = np.clip(
            np.searchsorted(self.edges, u, side="right") - 1,
            0,
            self.keep.size - 1,
        )
        col = np.cumsum(self.keep) - 1
        col = np.where(self.keep, col, -1)[idx]
        phi = np.zeros((u.size, self.order))
        hit = col >= 0
        phi[np.flatnonzero(hit), col[hit]] = 1.0
        return phi

    def edges_on(self, domain: tuple[float, float]) -> np.ndarray:
        """The edges in data units on ``domain = (x0, x1)``."""
        x0, x1 = float(domain[0]), float(domain[1])
        return x0 + (self.edges + 1.0) / 2.0 * (x1 - x0)

    def windows_on(
        self, domain: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Left and right ends of the kept windows in data units."""
        e = self.edges_on(domain)
        return e[:-1][self.keep], e[1:][self.keep]

    @classmethod
    def on(
        cls,
        edges: Any,
        domain: tuple[float, float],
        keep: Any = None,
    ) -> "EdgeBlockBasis":
        """The basis from edges in data units on ``domain``.

        Args:
            edges: Edges in data units, strictly increasing; the first and
                the last must be the ends of ``domain``, to a relative
                tolerance of 1e-12 of its span.
            domain: ``(x0, x1)``, the interval the image is taken on.
            keep: As in the constructor.

        Returns:
            The basis on the unit variable of ``domain``.

        Raises:
            ValueError: ``domain`` is degenerate; the first or last edge is
                not the matching end of ``domain``; plus the constructor's
                own errors.
        """
        e = np.asarray(edges, dtype=float).ravel()
        if e.size < 2:
            raise ValueError(
                f"an edge_block basis needs at least 2 edges, got {e.size}"
            )
        x0, x1 = float(domain[0]), float(domain[1])
        tol = 1e-12 * max(abs(x1 - x0), 1.0)
        if abs(e[0] - x0) > tol or abs(e[-1] - x1) > tol:
            raise ValueError(
                f"the first and last edge ({e[0]!r}, {e[-1]!r}) must be "
                f"the ends of the domain {(x0, x1)!r}"
            )
        u = u_of(e, x0, x1)
        u[0], u[-1] = -1.0, 1.0
        return cls(u, keep)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "order": self.order,
            "edges": [float(v) for v in self.edges],
            "keep": [bool(v) for v in self.keep],
        }

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.order,
                tuple(float(v) for v in self.edges),
                tuple(bool(v) for v in self.keep),
            )
        )


def aggregated_image(
    edges: Any,
    totals: Any,
    counts: Any,
    *,
    x: Any = None,
    domain: tuple[float, float] | None = None,
) -> Image:
    """The block image of data known only as one total per window.

    ``totals[k]`` is the sum of the samples that fell in window ``k`` and
    ``counts[k]`` how many there were, so the window means are
    ``totals / counts``. The image is the exact sufficient statistic of
    those totals in the window basis, which is what makes the fit on it the
    equal-areas least squares rather than the biased reading of the model at
    the window centres.

    Args:
        edges: Window edges in data units, strictly increasing, length
            ``K + 1``.
        totals: Sum of the samples per window, length ``K``.
        counts: Number of samples per window, length ``K``, non-negative.
            Windows with a zero count carry no data and are left out of the
            basis.
        x: The sample positions, when they are known; every position must
            fall in a window with a positive count and the number of
            positions per window must equal ``counts``. They are sorted
            here. ``None`` places ``counts[k]`` positions at the midpoints
            of equal sub-intervals of window ``k``, the midpoint quadrature
            of the window mean with as many nodes as there were samples.
        domain: ``(x0, x1)`` of the image; ``None`` takes
            ``(edges[0], edges[-1])``.

    Returns:
        An :class:`~dtfit.image.Image` in an :class:`EdgeBlockBasis` over
        the windows with a positive count, with ``S`` the kept totals, ``G``
        the diagonal matrix of the kept counts, ``n`` the total sample count
        and ``sumsq`` the sum of ``totals**2 / counts`` over the kept
        windows, the scatter inside a window not being observed.

    Raises:
        ValueError: fewer than two edges, edges not strictly increasing;
            ``totals`` or ``counts`` of the wrong length; a negative count;
            every count zero; ``x`` of a length other than ``counts.sum()``,
            a position outside the domain or in a zero-count window, or a
            per-window position count differing from ``counts``.
    """
    e = np.asarray(edges, dtype=float).ravel()
    if e.size < 2:
        raise ValueError(f"need at least 2 edges, got {e.size}")
    if not np.all(np.diff(e) > 0.0):
        raise ValueError("edges must be strictly increasing")
    t = np.asarray(totals, dtype=float).ravel()
    c = np.asarray(counts, dtype=float).ravel()
    n_win = e.size - 1
    if t.size != n_win or c.size != n_win:
        raise ValueError(
            f"totals and counts need one entry per window ({n_win}), got "
            f"{t.size} and {c.size}"
        )
    if np.any(c < 0.0):
        raise ValueError("counts must be non-negative")
    keep = c > 0.0
    if not keep.any():
        raise ValueError("every window is empty: no data to image")
    dom = (
        (float(e[0]), float(e[-1])) if domain is None
        else (float(domain[0]), float(domain[1]))
    )
    n = int(round(float(c.sum())))
    if x is None:
        pos = _midpoint_grid(e, c)
    else:
        pos = np.sort(np.asarray(x, dtype=float).ravel())
        if pos.size != n:
            raise ValueError(
                f"x has {pos.size} positions but the counts sum to {n}"
            )
        if pos.size and (pos[0] < e[0] or pos[-1] > e[-1]):
            raise ValueError(
                f"every position must lie in [{e[0]!r}, {e[-1]!r}]"
            )
        idx = np.clip(
            np.searchsorted(e, pos, side="right") - 1, 0, n_win - 1
        )
        per_window = np.bincount(idx, minlength=n_win).astype(float)
        if not np.array_equal(per_window, c):
            raise ValueError(
                "the positions per window do not match counts: got "
                f"{per_window.tolist()} against {c.tolist()}"
            )
    basis = EdgeBlockBasis.on(e, dom, keep)
    return Image(
        basis,
        dom,
        t[keep].copy(),
        np.diag(c[keep]),
        n,
        float(np.sum(t[keep] ** 2 / c[keep])),
        float(t.sum()),
        float(n),
        Grid.of(pos),
    )


def _midpoint_grid(edges: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Midpoints of ``counts[k]`` equal sub-intervals of each window."""
    out = []
    for k in np.flatnonzero(counts > 0.0):
        m = int(round(float(counts[k])))
        step = (edges[k + 1] - edges[k]) / m
        out.append(edges[k] + (np.arange(m) + 0.5) * step)
    return np.concatenate(out)


def fit_aggregated(
    model: Any,
    edges: Any,
    totals: Any,
    counts: Any,
    *,
    x: Any = None,
    domain: tuple[float, float] | None = None,
    **fit_kwargs: Any,
) -> FittingResult:
    """Fit ``model`` to data known only as one total per window.

    The fit is :func:`dtfit.fit` on :func:`aggregated_image`: the window
    sums of the model on the sample grid are matched to the window sums of
    the data.

    The covariance is on the scale of the window totals, not of the samples
    the caller never saw. With ``absolute_sigma`` unset or false the
    covariance :func:`dtfit.fit` returns is scaled by ``rss / (n - p)`` with
    ``n`` the sample count; here only ``K`` window observations enter the
    fit, so it is multiplied by ``(n - p) / (K - p)``, ``K`` the number of
    windows with a positive count and ``p`` the number of parameters. With
    ``absolute_sigma=True`` it is returned unchanged. ``cov`` is ``None``
    when ``K <= p``.

    Args:
        model: A SymPy expression string, a ``sympy.Expr`` or a callable
            ``f(x, *params)``, as :func:`dtfit.fit` takes.
        edges: Window edges in data units, length ``K + 1``.
        totals: Sum of the samples per window, length ``K``.
        counts: Number of samples per window, length ``K``.
        x: Sample positions when known; see :func:`aggregated_image`.
        domain: ``(x0, x1)``; ``None`` takes ``(edges[0], edges[-1])``.
        **fit_kwargs: Passed to :func:`dtfit.fit` (``p0``, ``bounds``,
            ``var``, ``param_names``, ``absolute_sigma``, ...).

    Returns:
        The :class:`~dtfit.types.FittingResult` of the fit, with ``cov``
        rescaled to the window-level degrees of freedom; ``stderr`` and the
        prediction bands follow from it.

    Raises:
        ValueError: from :func:`aggregated_image` or :func:`dtfit.fit`.
    """
    image = aggregated_image(
        edges, totals, counts, x=x, domain=domain
    )
    result = dtfit.fit(model, image, **fit_kwargs)
    if fit_kwargs.get("absolute_sigma", False) or result.cov is None:
        return result
    p = len(result.names) or result.coeffs.size
    k_kept = image.n_coef
    if k_kept <= p:
        result.cov = None
    else:
        result.cov = result.cov * ((image.n - p) / (k_kept - p))
    return result


def detect_jumps(
    counts: Any,
    sums_x: Any,
    resid_sums: Any,
    edges: Any,
    *,
    span: int = 8,
    threshold: float = 5.0,
    n_min: int = 2,
    exclude: Sequence[int] = (),
    self_scale: bool = False,
) -> list[tuple[int, float, float, float]]:
    """Flag the windows of a fine block image that hold a level jump.

    A skip-one local-linear test: at each window a weighted straight line is
    fitted to the residual means of the ``span`` non-empty windows on each
    side, the window itself left out, and the difference of the two
    extrapolations at the window's centre of mass is the jump amplitude.
    Its z statistic is against the noise of a window mean, read from the
    median absolute successive difference of the residual means. Detections
    come out by decreasing ``|z|`` with the ``span`` windows around each one
    suppressed, so two epochs closer than that give one detection.

    Everything is read from the fine image alone, so detection costs one
    pass over the data.

    Args:
        counts: Samples per fine window, length ``K_f``.
        sums_x: Sum of the sample positions per fine window, length
            ``K_f``; the window centres of mass are ``sums_x / counts``.
        resid_sums: Sum of the residuals of the current model per fine
            window, length ``K_f``.
        edges: The fine window edges in data units, length ``K_f + 1``.
        span: Windows a side entering each local line, >= 3 for a
            detection to be possible.
        threshold: Level of ``|z|`` a detection must reach, > 0.
        n_min: Fewest samples a window must hold to take part, >= 1.
        exclude: Fine windows already flagged; they and their immediate
            neighbours cannot be detected again.
        self_scale: Divide the statistic by its own robust spread over the
            record, which absorbs noise colour at the detector's time scale
            (a unit factor on white noise). The divisor is floored at 1, so
            self-scaling never makes a detection easier.

    Returns:
        ``(window, epoch, amplitude, z)`` per detection, strongest first:
        the fine window index, the epoch in data units placed inside that
        window by its area, the jump amplitude in the units of the data and
        the statistic. Empty when nothing reaches ``threshold``, when fewer
        than three windows hold ``n_min`` samples, or when the noise scale
        reads as zero.

    Raises:
        ValueError: the four arrays do not have matching lengths.
    """
    n = np.asarray(counts, dtype=float).ravel()
    sx = np.asarray(sums_x, dtype=float).ravel()
    rs = np.asarray(resid_sums, dtype=float).ravel()
    e = np.asarray(edges, dtype=float).ravel()
    kf = n.size
    if sx.size != kf or rs.size != kf or e.size != kf + 1:
        raise ValueError(
            "counts, sums_x and resid_sums must share a length and edges "
            f"must be one longer; got {kf}, {sx.size}, {rs.size}, {e.size}"
        )
    ok = n >= n_min
    good = np.flatnonzero(ok)
    if good.size < 3:
        return []
    safe = np.maximum(n, 1.0)
    m = np.where(ok, rs / safe, np.nan)
    tc = np.where(ok, sx / safe, np.nan)
    d1 = np.diff(m[good])
    spread = 1.4826 * float(np.median(np.abs(d1 - np.median(d1))))
    s2 = spread ** 2 / 2.0 * float(np.median(n[good]))
    if not np.isfinite(s2) or s2 <= 0.0:
        return []
    z = np.zeros(kf)
    amp = np.zeros(kf)
    lvl = np.zeros((kf, 2))
    for b in good:
        lo = good[good < b][-span:]
        hi = good[good > b][:span]
        if lo.size < 3 or hi.size < 3:
            continue
        at = tc[b]
        vl, fl = _side_fit(tc[lo], m[lo], n[lo], at)
        vr, fr = _side_fit(tc[hi], m[hi], n[hi], at)
        amp[b] = vr - vl
        lvl[b] = vl, vr
        z[b] = amp[b] / np.sqrt(s2 * (fl + fr))
    if self_scale:
        nz = z[z != 0.0]
        if nz.size > 10:
            scale = 1.4826 * float(np.median(np.abs(nz - np.median(nz))))
            z = z / max(scale, 1.0)
    found: list[tuple[int, float, float, float]] = []
    zz = np.abs(z).copy()
    for b in exclude:
        zz[max(0, int(b) - 1): int(b) + 2] = 0.0
    while True:
        b = int(np.argmax(zz))
        if zz[b] < threshold:
            break
        vl, vr = lvl[b]
        frac = (
            float(np.clip((m[b] - vl) / (vr - vl), 0.0, 1.0))
            if vr != vl else 0.5
        )
        epoch = e[b] + (1.0 - frac) * (e[b + 1] - e[b])
        found.append((b, float(epoch), float(amp[b]), float(z[b])))
        zz[max(0, b - span): b + span + 1] = 0.0
    return found


def _side_fit(
    tc: np.ndarray, m: np.ndarray, w: np.ndarray, at: float
) -> tuple[float, float]:
    """Weighted line through ``(tc, m)``: value and variance factor at
    ``at``."""
    tot = float(w.sum())
    tb = float((w * tc).sum() / tot)
    sxx = float((w * (tc - tb) ** 2).sum())
    mb = float((w * m).sum() / tot)
    slope = float((w * (tc - tb) * m).sum() / sxx) if sxx > 0.0 else 0.0
    var = 1.0 / tot + ((at - tb) ** 2 / sxx if sxx > 0.0 else 0.0)
    return mb + slope * (at - tb), var


def coarsen(
    n_fine: int, n_windows: int, flagged: Sequence[int] = ()
) -> np.ndarray:
    """Merge fine windows into about ``n_windows`` coarse ones.

    A block image is additive over windows, so its windows can be re-cut
    after the pass by merging fine ones. No coarse window spans a flagged
    fine window, which is what puts a coarse edge on each detected epoch;
    the flagged windows themselves are dropped, the epoch inside them being
    known only to within a fine window.

    Args:
        n_fine: Number of fine windows, >= 1.
        n_windows: Coarse windows aimed for, >= 1; the segments between the
            flagged windows each get a share in proportion to their length,
            at least one window, so the count is approximate.
        flagged: Fine windows to drop; repeats and order do not matter.

    Returns:
        The coarse window index of each fine window, length ``n_fine``;
        ``-1`` for a flagged fine window. The labels are non-decreasing.
    """
    flags = sorted({int(b) for b in flagged})
    lab = np.full(int(n_fine), -1, dtype=int)
    cuts = [-1] + flags + [int(n_fine)]
    free = int(n_fine) - len(flags)
    nxt = 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = np.arange(a + 1, b)
        if seg.size == 0:
            continue
        m = max(1, int(round(n_windows * seg.size / free)))
        lab[seg] = nxt + np.minimum(
            (np.arange(seg.size) * m) // seg.size, m - 1
        )
        nxt += m
    return lab


@dataclass(frozen=True)
class AlignedFit:
    """The outcome of :func:`fit_aligned`.

    Attributes:
        result: The fit of the model plus one level shift per epoch, on the
            aligned window image.
        epochs: The detected epochs in data units, ascending; empty when
            nothing was detected.
        steps: The fitted shift amplitudes, in the order of ``epochs``.
        z: The detection statistic of each epoch, same order.
        basis: The :class:`EdgeBlockBasis` the final fit ran in.
        n_dropped: Samples left out by ``clip``; the samples of the flagged
            windows are left out too and are not counted here.
    """

    result: FittingResult
    epochs: np.ndarray
    steps: np.ndarray
    z: np.ndarray
    basis: EdgeBlockBasis
    n_dropped: int


def fit_aligned(
    model: Callable[..., Any],
    original: Original,
    *,
    n_windows: int,
    fine: int = 8,
    span: int = 8,
    threshold: float = 5.0,
    self_scale: bool = False,
    clip: float | None = None,
    max_jumps: int | None = None,
    max_iter: int = 3,
    p0: Any = None,
    param_names: Sequence[str] | None = None,
    bounds: Any = None,
    **fit_kwargs: Any,
) -> AlignedFit:
    """Fit a model whose level jumps at unknown epochs, aligning the
    windows to them.

    The model is ``f(x, *params)`` plus one level shift ``d_j * (x >= e_j)``
    per detected epoch. Fine windows (``fine * n_windows`` of them) carry
    the image the detector works on: the model is fitted without shifts, the
    residual window sums go to :func:`detect_jumps`, the fine windows are
    merged by :func:`coarsen` to about ``n_windows`` with the flagged ones
    left out, and the model with its shifts is refitted on the samples
    outside the flagged windows in the resulting :class:`EdgeBlockBasis`.
    Detection then repeats on the new residual until it finds nothing new.
    Leaving the flagged window out is what makes the error of the epoch
    harmless.

    Args:
        model: A callable ``f(x, *params)``; its parameter names are
            introspected unless ``param_names`` is given.
        original: The record; its ``domain`` sets the window range.
        n_windows: Coarse windows aimed for, >= 1; the fit sees this many
            numbers plus the shifts.
        fine: Fine windows per coarse window, >= 1. The epoch resolution is
            one fine window, ``span(domain) / (fine * n_windows)``.
        span: Fine windows a side in the detector's local lines, >= 3.
        threshold: Detection level of ``|z|``, > 0.
        self_scale: Scale the detection statistic by its own robust spread
            (see :func:`detect_jumps`); for coloured noise.
        clip: ``None``, or the number of robust sigma (1.4826 MAD) beyond
            which a sample is left out before anything else, measured
            against its own fine window's median. Guards the detector
            against gross outliers.
        max_jumps: Cap on the number of epochs; ``None`` is no cap.
        max_iter: Detection rounds, >= 1.
        p0: Initial guess for the model's own parameters, positional or
            name-keyed; the shifts are seeded from the detector's
            amplitudes.
        param_names: Names of the model's parameters after ``x``.
        bounds: Bounds on the model's own parameters; the shifts are
            unbounded.
        **fit_kwargs: Passed to :func:`dtfit.fit`.

    Returns:
        An :class:`AlignedFit`. Finding no jump is a normal outcome: the
        epochs are empty and the result is the plain fit on ``n_windows``
        equal windows.

    Raises:
        ValueError: ``n_windows``, ``fine`` or ``max_iter`` below 1; the
            model's parameter names cannot be introspected and were not
            given; ``clip`` leaves too few samples to image; plus the
            errors of :func:`dtfit.fit`.
    """
    if n_windows < 1 or fine < 1:
        raise ValueError(
            f"n_windows and fine must be >= 1, got {n_windows} and {fine}"
        )
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    names = list(resolve_model(model, param_names=param_names).names)
    base_p0 = normalize_p0(p0, names)
    base_p0 = np.ones(len(names)) if base_p0 is None else base_p0
    base_bounds = normalize_bounds(bounds, names)
    dom = original.domain
    kf = int(fine) * int(n_windows)
    x, y = original.x, original.y
    n_dropped = 0
    if clip is not None:
        keep_s = _clip_local(x, y, dom, kf, float(clip))
        n_dropped = int(keep_s.size - keep_s.sum())
        x, y = x[keep_s], y[keep_s]
        if x.size < kf:
            raise ValueError(
                f"clip={clip} left {x.size} samples, too few for "
                f"{kf} fine windows"
            )
    idx = _fine_index(x, dom, kf)
    counts = np.bincount(idx, minlength=kf).astype(float)
    sums_x = np.bincount(idx, weights=x, minlength=kf)
    fine_edges = np.linspace(dom[0], dom[1], kf + 1)

    def stage(epochs: list[float], flagged: list[int], coarse: bool):
        f = _stepped(model, epochs)
        all_names = names + [f"step_{i + 1}" for i in range(len(epochs))]
        guess = np.concatenate([base_p0, np.zeros(len(epochs))])
        guess[len(names):] = [amp_of[e] for e in epochs]
        b = None
        if base_bounds is not None:
            b = list(base_bounds) + [
                (-np.inf, np.inf) for _ in epochs
            ]
        if coarse:
            lab = coarsen(kf, n_windows, flagged)
            basis = _basis_of(lab, fine_edges, dom)
            sel = lab[idx] >= 0
            data = Original(x[sel], y[sel], domain=dom)
        else:
            basis = BlockBasis(kf)
            data = Original(x, y, domain=dom)
        res = dtfit.fit(
            f, data, basis=basis, p0=guess, bounds=b,
            param_names=all_names, **fit_kwargs,
        )
        return res, basis, f

    amp_of: dict[float, float] = {}
    z_of: dict[float, float] = {}
    epochs: list[float] = []
    flagged: list[int] = []
    result, basis, stepped = stage([], [], coarse=False)
    aligned = False
    for _ in range(max_iter):
        r = y - stepped(x, *result.coeffs)
        resid_sums = np.bincount(idx, weights=r, minlength=kf)
        new = detect_jumps(
            counts, sums_x, resid_sums, fine_edges, span=span,
            threshold=threshold, exclude=flagged, self_scale=self_scale,
        )
        if max_jumps is not None:
            new = new[: max(0, int(max_jumps) - len(epochs))]
        if not new:
            break
        for b, epoch, amplitude, z in new:
            flagged.append(b)
            epochs.append(epoch)
            amp_of[epoch] = amplitude
            z_of[epoch] = z
        epochs.sort()
        result, basis, stepped = stage(epochs, flagged, coarse=True)
        aligned = True
        if max_jumps is not None and len(epochs) >= int(max_jumps):
            break
    if not aligned:
        result, basis, stepped = stage([], [], coarse=True)
    return AlignedFit(
        result=result,
        epochs=np.asarray(epochs, dtype=float),
        steps=np.asarray(result.coeffs[len(names):], dtype=float),
        z=np.asarray([z_of[e] for e in epochs], dtype=float),
        basis=basis,
        n_dropped=n_dropped,
    )


def _stepped(
    model: Callable[..., Any], epochs: Sequence[float]
) -> Callable[..., Any]:
    """``model`` plus one level shift per epoch, the shifts last."""
    eps = np.asarray(epochs, dtype=float)

    def f(x: np.ndarray, *params: float) -> np.ndarray:
        k = len(params) - eps.size
        v = np.asarray(model(x, *params[:k]), dtype=float)
        for d, e in zip(params[k:], eps):
            v = v + d * (np.asarray(x, dtype=float) >= e)
        return v

    return f


def _fine_index(
    x: np.ndarray, domain: tuple[float, float], kf: int
) -> np.ndarray:
    """Fine window of each position, clamped to the end windows."""
    u = (x - domain[0]) / (domain[1] - domain[0]) * kf
    return np.clip(u.astype(int), 0, kf - 1)


def _clip_local(
    x: np.ndarray,
    y: np.ndarray,
    domain: tuple[float, float],
    kf: int,
    c: float,
) -> np.ndarray:
    """Samples within ``c`` robust sigma of their fine window's median."""
    idx = _fine_index(x, domain, kf)
    order = np.argsort(idx, kind="stable")
    cuts = np.flatnonzero(np.diff(idx[order])) + 1
    med = np.empty_like(y)
    for g in np.split(order, cuts):
        med[g] = np.median(y[g])
    r = y - med
    scale = 1.4826 * float(np.median(np.abs(r)))
    if scale <= 0.0:
        return np.ones(y.size, dtype=bool)
    return np.abs(r) <= c * scale


def _basis_of(
    lab: np.ndarray, fine_edges: np.ndarray, domain: tuple[float, float]
) -> EdgeBlockBasis:
    """The basis of the coarse windows a label array describes."""
    bounds = np.concatenate(
        [[0], np.flatnonzero(np.diff(lab)) + 1, [lab.size]]
    )
    return EdgeBlockBasis.on(
        fine_edges[bounds], domain, lab[bounds[:-1]] >= 0
    )
