"""Experimental structural adaptations of EAC and LSI.

The distribution has two tiers. The library tier is this package
(``import dtfit_experimental``): a small importable surface of adaptations
plus the backend helpers, staging code that may graduate into stable
``dtfit``. The study tier is :mod:`dtfit_experimental.experiments`, the
per-case and per-domain validation suite each adaptation is measured in.
That tier is a research tree rather than an API: exempt from the mypy gate,
ruff-relaxed, driven with ``python -m dtfit_experimental.experiments...``
instead of imported. Nothing in the library tier imports from it.
:mod:`dtfit_experimental.streaming` is a third, narrower surface:
``FilterBank`` and ``FusedChiSquareDetector``, experiment tooling over
``dtfit``'s filters rather than library API, moved here with the
experiments that use them.

The adaptations are new ways to compose the differential-transformation
fitting methods of :mod:`dtfit`, grounded in the methods' own math: linearity
of integration, orthogonal-basis projection, additive areas. They are
prototyped here and evaluated across the experiment suite; whatever holds up
on two or more domains is promoted into stable ``dtfit``, where it then lives
physically rather than being re-imported from here.

    from dtfit_experimental import (
        InformationFilter,    # inverse-covariance (info-form) fusion primitive
        FourierBasis,         # image-interface basis, fit(basis=...)
        ChebyshevBasis,       # image-interface basis, fit(basis=...)
        LaguerreBasis,        # image-interface basis, fit(basis=...)
        EdgeBlockBasis,       # block basis with explicit window edges
        SegmentBasis,         # Legendre per segment between explicit edges
        aggregated_image,     # image of data known as window totals
        fit_aggregated,       # the equal-areas fit of such data
        fit_aligned,          # windows aligned to detected jump epochs
        AlignedFit,           # what fit_aligned returns
        detect_jumps,         # the jump test on a fine window image
        LocalTimeFilter,      # window image filter in the newest sample's time
        shift_matrix,         # polynomial coefficients under an origin shift
    )

``FourierBasis``, ``ChebyshevBasis`` and ``LaguerreBasis`` are the
image-projection form, reached through ``dtfit.fit(basis=...)``.

These signatures may change until promotion. ``InformationFilter``, the
inverse-covariance primitive whose fusion is a plain addition, is coherent and
tested but no domain study exercises it and the covariance-form the block filter
/ the Legendre filter do not use it; it waits here for a sensor-fusion or embedded
domain. Measured verdicts live in
``experiments/results/19_adaptations_in_trial``.
"""

from dtfit._core._backend import available_backends, resolve_backend, Backend
from .bases import ChebyshevBasis, FourierBasis, LaguerreBasis
from .blocks import (
    AlignedFit,
    EdgeBlockBasis,
    SegmentBasis,
    aggregated_image,
    detect_jumps,
    fit_aggregated,
    fit_aligned,
)
from .weak_ode import (
    fit_logistic,
    fit_damped_oscillator,
    fit_lotka_volterra_prey,
    fit_michaelis_menten,
    seed_nlls,
    weak_operators,
)
from .information import InformationFilter
from .local_time import LocalTimeFilter, shift_matrix

__all__ = [
    "available_backends",
    "resolve_backend",
    "Backend",
    "InformationFilter",
    "LocalTimeFilter",
    "shift_matrix",
    "FourierBasis",
    "ChebyshevBasis",
    "LaguerreBasis",
    "EdgeBlockBasis",
    "SegmentBasis",
    "aggregated_image",
    "fit_aggregated",
    "fit_aligned",
    "AlignedFit",
    "detect_jumps",
    "weak_operators",
    "fit_logistic",
    "fit_michaelis_menten",
    "fit_lotka_volterra_prey",
    "fit_damped_oscillator",
    "seed_nlls",
]
