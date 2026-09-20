"""Every historical stage of the method on one problem, as adapters onto
the historical APIs: the ten batch-fit stages from ``dtfit_legacy.book``
through ``dtfit.fit``, the six streaming filter factories, and the
map-reduce accumulators beside the image merge.

Nothing here loops over families, conditions or seeds, or returns a table:
that is notebook 17's job. A stage never raises for an expected condition;
:func:`_classify` sorts every exception a stage's inner call can raise
into ``"not applicable"`` (the criterion cannot be asked of this family)
or ``"failed"`` (anything else).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import curve_fit

from dtfit import Image, ImageFilter, Original, fit
from dtfit_legacy import (
    dsb_balance, eac_areas, find_degree, fit_dsb, lsi_integral_monomial,
)
from dtfit_legacy.integral import fit_eac, fit_lsi
from dtfit_legacy.scale import PartitionedBatchLSI, PartitionedEAC, \
    PartitionedLSI
from dtfit_legacy.streaming import EACFilter, LSIFilter

#: No Maclaurin spectrum at the origin, or fewer informative Maclaurin
#: orders than parameters: the criterion cannot be asked of this family.
NOT_APPLICABLE = "not applicable"

_UNDERDETERMINED = re.compile(
    r"^Only \d+ of the first \d+ Maclaurin orders constrain the \d+ "
    r"parameters")


@dataclass(frozen=True)
class StageResult:
    """Outcome of one batch-stage adapter on one draw.

    Attributes:
        key: The stage's key, one of ``BATCH_STAGES``'s.
        params: Fitted parameter vector in the model's sorted-name order;
            ``None`` unless ``status == "ok"``.
        cov: Parameter covariance (``n_params x n_params``); ``None``
            unless ``status == "ok"``, and ``None`` even then for a stage
            that produces no covariance (a square balance, or a source
            form with as many equations as parameters).
        status: ``"ok"`` (the fit ran and returned parameters), ``"not
            applicable"`` (see :data:`NOT_APPLICABLE`), or ``"failed"``
            (any other exception from the stage's inner call).
        message: Empty on ``"ok"``; the raised exception's own message on
            ``"not applicable"``; the exception's class name on
            ``"failed"``.
        converged: The stage's own convergence flag, forwarded from
            ``FittingResult.converged`` unchanged; ``None`` unless
            ``status == "ok"``, and ``None`` even then for a stage whose
            method reports no convergence flag (``S0_nlls``, which
            raises instead of returning an unconverged fit). ``False``
            here is a silent failure: a result came back, but the
            optimizer never settled.
    """

    key: str
    params: np.ndarray | None
    cov: np.ndarray | None
    status: str
    message: str
    converged: bool | None = None


def _classify(exc: Exception, key: str) -> tuple[str, str]:
    """Sort an exception a stage's inner call raised into ``"not
    applicable"`` or ``"failed"``, given the raising stage's key.

    ``KeyError('ComplexInfinity')`` comes from a Maclaurin spectrum that
    does not exist at the origin (a fractional power such as ``a*t**b``
    in ``power_law``): not applicable regardless of stage. The
    ``ValueError`` ``dsb_balance`` raises when fewer Maclaurin orders
    carry a parameter than the model has (an odd model such as
    ``a*atan(w*t)``, whose even orders all vanish) is the same kind of
    structural mismatch for ``S1_balance_book``, which reads the
    spectrum straight from the model with no degree choice: not
    applicable there. ``S2_balance_adequate`` can raise the identical
    message from ``fit_dsb`` after picking its own polynomial degree
    from the draw, so there the same condition is draw-dependent, not a
    family property, and is a failure like everything else, including
    the ``ValueError`` ``fit_dsb`` raises for a rank below the parameter
    count (a caller error, not a family property).
    """
    msg = str(exc)
    if isinstance(exc, KeyError) and "ComplexInfinity" in msg:
        return NOT_APPLICABLE, msg
    if key == "S1_balance_book" and isinstance(exc, ValueError) \
            and _UNDERDETERMINED.match(msg):
        return NOT_APPLICABLE, msg
    return "failed", type(exc).__name__


@dataclass(frozen=True)
class BatchStage:
    """One adapter of :data:`BATCH_STAGES` onto a historical fitting API.

    Attributes:
        key: Short identifier, for example ``"S3b_lsi_legendre"``.
        label: Plain-English name for a table column, for example
            ``"integral least squares, Legendre basis"``.
        needs_p0: Whether the stage's accuracy depends on a start close to
            the truth; ``False`` for a stage that solves symbolically or
            selects its own polynomial degree.
        run: ``run(x, y, expr, model, p0) -> StageResult``. ``model`` is
            a callable ``f(x, *params)`` (see
            :func:`dtfit_experimental.study.problems.model`); only
            ``S0_nlls`` uses it, the rest fit from ``expr`` directly.
            Never raises: every exception from the underlying call is
            classified into ``StageResult.status``.
    """

    key: str
    label: str
    needs_p0: bool
    run: Callable[
        [np.ndarray, np.ndarray, str, Callable[..., np.ndarray],
         np.ndarray | None],
        StageResult]


def _wrap(
    key: str, label: str, needs_p0: bool,
    inner: Callable[..., tuple[np.ndarray, np.ndarray | None, bool | None]],
) -> BatchStage:
    def run(x, y, expr, model, p0):
        try:
            coeffs, cov, converged = inner(x, y, expr, model, p0)
        except Exception as exc:  # noqa: BLE001
            status, message = _classify(exc, key)
            return StageResult(key, None, None, status, message)
        return StageResult(
            key, np.asarray(coeffs, dtype=float), cov, "ok", "", converged)
    return BatchStage(key, label, needs_p0, run)


def _s0(x, y, expr, model, p0):
    coeffs, cov = curve_fit(model, x, y, p0=p0, maxfev=4000)
    return coeffs, cov, None


def _s1(x, y, expr, model, p0):
    r = dsb_balance(x, y, expr, "t", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s2(x, y, expr, model, p0):
    deg = find_degree(x, y)
    coeffs = np.polynomial.polynomial.polyfit(x, y, deg)
    r = fit_dsb(coeffs, expr, "t", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s3a(x, y, expr, model, p0):
    r = lsi_integral_monomial(x, y, expr, "t", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s3b(x, y, expr, model, p0):
    r = fit_lsi(x, y, expr, "t", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s3c(x, y, expr, model, p0):
    r = eac_areas(x, y, expr, "t", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s3d(x, y, expr, model, p0):
    r = fit_eac(x, y, expr, "t", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s4a(x, y, expr, model, p0):
    r = fit(expr, Original(x, y), "t", basis="legendre", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s4b(x, y, expr, model, p0):
    r = fit(expr, Original(x, y), "t", basis="block", p0=p0)
    return r.coeffs, r.cov, r.converged


def _s4r(x, y, expr, model, p0):
    r = fit(expr, Original(x, y), "t", basis="legendre", robust=True, p0=p0)
    return r.coeffs, r.cov, r.converged


#: The ten stages the estimator grew through, source criteria to image,
#: each an adapter onto the historical API named in its docstring above.
BATCH_STAGES: tuple[BatchStage, ...] = (
    _wrap("S0_nlls", "nonlinear least squares (reference)", True, _s0),
    _wrap("S1_balance_book", "spectrum balance, source form", False, _s1),
    _wrap(
        "S2_balance_adequate", "spectrum balance, adequate degree", False,
        _s2),
    _wrap(
        "S3a_lsi_monomial", "integral least squares, monomial basis", True,
        _s3a),
    _wrap(
        "S3b_lsi_legendre", "integral least squares, Legendre basis", True,
        _s3b),
    _wrap("S3c_eac_book", "equal areas, source form", True, _s3c),
    _wrap("S3d_eac_direct", "equal areas, direct", True, _s3d),
    _wrap("S4a_image_legendre", "image, Legendre basis", True, _s4a),
    _wrap("S4b_image_block", "image, block basis", True, _s4b),
    _wrap("S4r_image_robust", "image, robust", True, _s4r),
)

#: Six streaming-filter factories, ``make(window_size, **overrides)``,
#: on the shared sinusoid ``"A*sin(w*t)"``. The streaming matrix uses the
#: five that are not ``"recursive_subareas"``, which is measured only in
#: the observability and cost sub-sections alongside the other two
#: recursive forms.
FILTERS: dict[str, Callable[..., Any]] = {
    "recursive_area": lambda window_size, **kw: EACFilter(
        "A*sin(w*t)", "t", window_size=window_size, n_sub=1,
        adaptive_window=False, **kw),
    "recursive_subareas": lambda window_size, **kw: EACFilter(
        "A*sin(w*t)", "t", window_size=window_size, n_sub=4,
        adaptive_window=False, **kw),
    "recursive_spectrum": lambda window_size, **kw: LSIFilter(
        "A*sin(w*t)", "t", window_size=window_size, order=5,
        adaptive_window=False, **kw),
    "image_legendre": lambda window_size, **kw: ImageFilter(
        "A*sin(w*t)", "t", basis="legendre", order=5,
        window_size=window_size, adaptive_window=False, **kw),
    "image_block": lambda window_size, **kw: ImageFilter(
        "A*sin(w*t)", "t", basis="block", order=2,
        window_size=window_size, adaptive_window=False, **kw),
    "image_legendre_adaptive": lambda window_size, **kw: ImageFilter(
        "A*sin(w*t)", "t", basis="legendre", order=5,
        window_size=window_size, adaptive_window=True, **kw),
}


class ImageAccumulator:
    """The discrete image's map-reduce counterpart to the integral
    accumulators, sharing their ``update``/``merge``/``fit`` interface so
    the map-reduce matrix can loop over :data:`ACCUMULATORS` uniformly.

    Each ``update`` builds a ``dtfit.Image`` from the chunk and folds it
    into the running merge; ``merge`` combines two accumulators the same
    way ``dtfit.Image.merge`` combines images.
    """

    def __init__(
        self, expr: str, var: str, *, domain: tuple[float, float],
        order: int = 6, basis: str = "legendre",
    ) -> None:
        """
        Args:
            expr: Model, a SymPy expression string in ``var``.
            var: Main variable name in ``expr``.
            domain: ``(x0, x1)`` every chunk's image is built on.
            order: Basis order (polynomial degree for Legendre, window
                count for block).
            basis: Name accepted by ``dtfit.image.bases.make_basis``.
        """
        self.expr, self.var, self.domain = expr, var, domain
        self.order, self.basis = order, basis
        self._image: Image | None = None

    def update(self, x: np.ndarray, y: np.ndarray) -> "ImageAccumulator":
        """Fold one chunk's image into the running merge.

        Args:
            x, y: One chunk's samples.

        Returns:
            ``self``, mutated in place, for chaining.
        """
        chunk = Image.of(
            Original(np.asarray(x), np.asarray(y), domain=self.domain),
            self.basis, self.order)
        self._image = chunk if self._image is None \
            else self._image.merge(chunk)
        return self

    def merge(self, other: "ImageAccumulator") -> "ImageAccumulator":
        """Associative reduce, mirroring the integral accumulators'.

        Args:
            other: Another accumulator on the same ``domain``, ``basis``
                and ``order``; unchanged.

        Returns:
            ``self``, mutated in place to hold both accumulators'
            combined image, for chaining. Unchanged if ``other`` has
            folded in no chunk yet.
        """
        if other._image is not None:
            self._image = other._image if self._image is None \
                else self._image.merge(other._image)
        return self

    def fit(self, *, p0=None):
        """Fit ``expr`` to the reduced image.

        Args:
            p0: Initial parameter guess; ``None`` is all ones.

        Returns:
            A ``dtfit.FittingResult`` on the accumulated image.

        Raises:
            TypeError: no chunk was folded in yet (``update`` never
                called), so there is no image to fit.
        """
        return fit(
            self.expr, self._image, self.var, basis=self.basis,
            order=self.order, p0=p0)


#: The map-reduce matrix's four accumulators, the image merge beside the
#: three integral accumulators it is compared against. Every factory takes
#: ``(expr, var, *, domain, ...)``; the rest of the keywords are the
#: accumulator's own (``order`` for the two LSI-family ones, ``n_windows``
#: for ``"integral_eac"``, ``order`` and ``n_channels`` for
#: ``"integral_batch_lsi"``).
ACCUMULATORS: dict[str, Callable[..., Any]] = {
    "image_merge": ImageAccumulator,
    "integral_lsi": PartitionedLSI,
    "integral_eac": PartitionedEAC,
    "integral_batch_lsi": PartitionedBatchLSI,
}
