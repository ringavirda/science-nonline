"""Parallel batch fitting -- fan many independent fits across CPU cores.

The batch methods (:func:`dtfit.fit_lsi`, :func:`dtfit.fit_eac`) are pure per
problem: fitting one signal never touches another. A real workload -- the
channels of a multivariate series, the cells of a noise/size sweep, the chunks
of a large stream, the axes of a trajectory -- is therefore *embarrassingly
parallel*. :func:`fit_many` maps the chosen method over a list of independent
problems with :mod:`joblib`, so an N-core machine fits ~N signals at once.

Two backends, both useful:

* ``backend="loky"`` (default) -- separate worker **processes**, true parallelism
  unaffected by the GIL. Problems carry the model as a SymPy **expression
  string** (picklable); workers rebuild and lambdify it, and return a
  :class:`dtfit.FittingResult` that drops its lambdified callable on pickling
  (rebuilt lazily on the caller side), so nothing unpicklable crosses the
  process boundary -- this matters on Windows, where workers are spawned, not
  forked.
* ``backend="threading"`` -- worker **threads** sharing memory. The compiled
  numeric kernels (``dtfit._native``) release the GIL on their hot loops, so the
  integral/projection work runs concurrently without process or pickling
  overhead; best when the per-problem arrays are large.

``joblib`` ships with scikit-learn (already a core dependency), so this adds no
new requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, cast

import numpy as np
from joblib import Parallel, delayed

from dtfit.methods import fit_lsi, fit_eac
from dtfit.types import FittingResult

# ``FittingResult`` is itself picklable (it drops its lazily-built callable on
# pickling and rebuilds it from ``expr``/``coeffs`` on first access), so it
# doubles as the batch result -- there is no separate lightweight type. Batch and
# single fits return the same :class:`dtfit.FittingResult`.

__all__ = ["FittingProblem", "fit_many"]

_FITTERS: dict[str, Callable[..., FittingResult]] = {
    "lsi": fit_lsi,
    "eac": fit_eac,
}


@dataclass
class FittingProblem:
    """One independent fit, fully described by picklable data.

    Attributes:
        x, y: Observed samples for this problem.
        expr: Model expression string, e.g. ``"a*exp(b*t)"``.
        var: Main variable name in ``expr``.
        method: ``"lsi"`` or ``"eac"``.
        kwargs: Method-specific keyword arguments (e.g. ``p0``, ``bounds``).
        label: Optional tag carried through to the result (channel name, etc.).
    """

    x: np.ndarray
    y: np.ndarray
    expr: str
    var: str
    method: str = "lsi"
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: Any = None


def _fit_one(problem: FittingProblem) -> FittingResult:
    """Worker entry point: fit a single problem, returning a picklable result.

    Module-level so it is importable in spawned worker processes. A failed fit
    is captured as ``error`` rather than crashing the whole batch. The returned
    :class:`FittingResult` carries the problem ``label`` and rebuilds its model
    lazily on the caller side (nothing unpicklable crosses the boundary).
    """
    fitter = _FITTERS.get(problem.method)
    if fitter is None:
        return FittingResult(
            coeffs=np.array([]), expr=problem.expr, var=problem.var,
            label=problem.label,
            error=f"unknown method {problem.method!r} (use 'lsi' or 'eac')",
        )
    try:
        res = fitter(
            np.asarray(problem.x, dtype=float),
            np.asarray(problem.y, dtype=float),
            problem.expr,
            problem.var,
            **problem.kwargs,
        )
        res.label = problem.label
        return res
    except Exception as exc:  # keep the batch alive; report per-problem
        return FittingResult(
            coeffs=np.array([]), expr=problem.expr, var=problem.var,
            label=problem.label, error=f"{type(exc).__name__}: {exc}",
        )


def fit_many(
    problems: Sequence[FittingProblem],
    *,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> list[FittingResult]:
    """Fit many independent problems in parallel.

    Args:
        problems: Independent :class:`FittingProblem` specs.
        n_jobs: Worker count (``-1`` = all cores; ``1`` = serial, no pool).
        backend: ``"loky"`` (processes, default), ``"threading"`` (threads,
            rides the GIL-released native kernels), or ``"multiprocessing"``.
        verbose: Forwarded to :class:`joblib.Parallel` (progress chatter).

    Returns:
        A :class:`dtfit.FittingResult` per problem, in input order, each tagged
        with the problem's ``label``. A problem that failed has ``error`` set and
        an empty ``coeffs`` rather than aborting the batch.
    """
    problems = list(problems)
    if not problems:
        return []
    if n_jobs == 1:  # avoid pool setup/serialization overhead for serial runs
        return [_fit_one(p) for p in problems]
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_fit_one)(p) for p in problems
    )
    return cast("list[FittingResult]", list(results))
