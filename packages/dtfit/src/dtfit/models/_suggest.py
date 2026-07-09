"""Model inference: fit a set of candidate families and rank them.

``suggest_models`` answers the question the domain studies say is the whole game
-- *which model* -- with evidence: it fits each candidate family (self-seeded)
and ranks them by AIC (parsimony-penalised fit), so the user chooses from a
scored shortlist instead of guessing a string.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from dtfit.types import FittingResult
from dtfit.diagnostics import fit_report
from dtfit._signal import dominant_period
from ._model import Model
from ._catalog import CATALOG


# Which family categories are plausible for each detected coarse shape.
_SHAPE_CATEGORIES = {
    "oscillatory": {"oscillatory"},
    "peak": {"peak"},
    "monotone": {"trend", "growth", "decay", "sigmoid", "saturating"},
}

# Every category the shape detector can reason about. A catalog family whose
# category is OUTSIDE this set (e.g. a user-registered family's default
# "general") carries no shape signal to prune on, so ``_shortlist`` always keeps
# it -- an open vocabulary, so a registered family is never silently dropped.
_KNOWN_CATEGORIES = frozenset().union(*_SHAPE_CATEGORIES.values())


def _detect_categories(x: np.ndarray, y: np.ndarray) -> set[str]:
    """Coarse shape -> plausible family categories (for shortlisting).

    Deliberately permissive: when the shape is ambiguous it returns every
    category, so the recommender never silently drops the true family.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    cats: set[str] = set()
    _, strength = dominant_period(y)
    # a real cycle: a strong spectral peak *and* several zero-crossings of the
    # detrended signal (>= ~2 full periods). Crossing-count separates an
    # oscillation from a single broad hump, which a spectral period alone can't.
    t = np.arange(n)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    crossings = int(np.count_nonzero(np.diff(np.sign(resid)) != 0))
    # Robust monotonicity via Spearman rank correlation. A per-sample diff-sign
    # test is noise-dominated where the true slope is small -- a noisy sigmoid's
    # flat tails fail it, dropping the sigmoid/saturating families for exactly
    # the S-curves they describe. Spearman tolerates that (a logistic scores
    # ~0.98; a sine ~0). scipy's typed stub returns a private result class whose
    # ``statistic`` attribute it does not expose; the access is valid at runtime.
    rho = (spearmanr(x, y).statistic  # pyright: ignore[reportAttributeAccessIssue]
           if n > 2 and float(np.std(y)) > 0 else 0.0)
    monotone = abs(float(np.nan_to_num(rho))) > 0.85
    # The oscillatory tag is ADDITIVE, not vetoed by monotonicity. The old
    # ``(not monotone)`` veto dropped the whole oscillatory family for
    # trend+seasonal data (a sinusoid riding a trend has high Spearman rho against
    # x), which is exactly the trend-plus-cycle case the recommender must handle.
    # The veto is also unnecessary: ``crossings`` is measured on the LINE-DETRENDED
    # residual, so a monotone S-curve (a single hump once detrended) crosses zero
    # ~2 times, not >= 4, and broadband noise is rejected by the spectral-strength
    # gate. So a strong spectral peak with several detrended crossings tags
    # oscillatory even under a trend; the monotone tag (below) is added too, and
    # the AIC ranking decides which structure actually fits.
    oscillatory = strength > 0.12 and crossings >= 4
    if oscillatory:
        cats |= _SHAPE_CATEGORIES["oscillatory"]
    # an interior extremum that the series rises to and falls from -> a peak.
    # Kept independent of the oscillatory tag: a few separated peaks (e.g. a
    # double Gaussian) read as a low-frequency cycle to the crossing test, so we
    # shortlist the peak families alongside the oscillatory ones rather than drop
    # them -- the AIC ranking sorts out which structure actually fits.
    i = int(np.argmax(np.abs(y - np.median(y))))
    if not monotone and n > 10 and 0.1 * n < i < 0.9 * n:
        cats |= _SHAPE_CATEGORIES["peak"]
    # mostly-monotone -> trend/growth/decay/sigmoid/saturating
    if monotone:
        cats |= _SHAPE_CATEGORIES["monotone"]
    if not cats:  # ambiguous -> try everything
        cats = set().union(*_SHAPE_CATEGORIES.values())
    cats.add("trend")  # always keep a polynomial baseline in the running
    return cats


def _shortlist(x: np.ndarray, y: np.ndarray) -> list[Model]:
    cats = _detect_categories(x, y)
    out: list[Model] = []
    for factory in CATALOG.values():
        m = factory()
        # A known-category family is pruned to the detected shape (the historical
        # behaviour). A family with an UNKNOWN category (a registered/custom
        # one whose category is outside the recommender's vocabulary) has no
        # shape signal, so it is ALWAYS kept -- it must never vanish silently.
        if m.category in cats or m.category not in _KNOWN_CATEGORIES:
            out.append(m)
    return out


@dataclass
class Suggestion:
    """One ranked candidate: the family, its fit, and the goodness-of-fit report."""

    name: str
    model: Model
    result: FittingResult
    report: dict

    @property
    def aic(self) -> float:
        return float(self.report["aic"])

    @property
    def bic(self) -> float:
        return float(self.report["bic"])

    @property
    def r2(self) -> float:
        return float(self.report["r2"])

    def __repr__(self) -> str:
        return (f"Suggestion({self.name!r}, r2={self.r2:.4f}, "
                f"aic={self.aic:.1f}, params={self.result.params})")


def suggest_models(
    x: np.ndarray,
    y: np.ndarray,
    candidates: list[Model] | None = None,
    *,
    method: str = "auto",
    top: int | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Suggestion]:
    """Fit candidate model families to ``(x, y)`` and rank them by AIC.

    Args:
        x, y: Observed samples.
        candidates: Models to try. Default: a **shape-based shortlist** of the
            catalog (oscillatory data skips the peak/monotone families, etc.;
            ambiguous data falls back to the whole catalog so the true family is
            never dropped). Each is fit self-seeded via :meth:`Model.fit`.
        method: Fitting method passed to each model (``"auto"`` routes by shape).
        top: If given, return only the best ``top`` suggestions.
        include: Keep only candidates whose **name** (e.g. ``"logistic"``) or
            **category** (e.g. ``"decay"``, ``"oscillatory"``) is in this list --
            restrict the search to families you believe plausible.
        exclude: Drop candidates whose name or category is in this list -- prune
            families you know don't apply (e.g. ``exclude=["oscillatory"]`` on a
            monotone series) without post-filtering the result. Applied after
            ``include``.

    Returns:
        :class:`Suggestion` list sorted best-first (lowest AIC). Candidates whose
        fit fails are skipped with a :class:`UserWarning` naming the failed
        family. Use ``[s.name for s in suggest_models(x, y)][:3]`` for a quick
        shortlist, or inspect ``.report`` for the full diagnostics.
    """
    models = candidates if candidates is not None else _shortlist(x, y)
    if include is not None:
        inc = set(include)
        models = [m for m in models if m.name in inc or m.category in inc]
    if exclude is not None:
        exc = set(exclude)
        models = [m for m in models if m.name not in exc and m.category not in exc]
    out: list[Suggestion] = []
    for m in models:
        try:
            res = m.fit(x, y, method=method)
            rep = fit_report(res, x, y)
        except Exception as exc:
            # A failed candidate is skipped but never silently: the user whose
            # true family errored must see why it is absent from the ranking.
            warnings.warn(
                f"candidate model '{m.name}' failed: {exc}",
                UserWarning, stacklevel=2,
            )
            continue
        if not np.isfinite(rep["r2"]):
            continue
        out.append(Suggestion(m.name, m, res, rep))
    out.sort(key=lambda s: s.aic if np.isfinite(s.aic) else np.inf)
    return out[:top] if top else out
