"""The historical stages of differential-transformation fitting.

Every method here is a stage the current ``dtfit`` estimator grew out of,
kept runnable so the evolution can be measured on one problem set:

* ``book``: the three criteria as the sources state them (spectrum
  balance, integral least squares on the monomial basis, equal areas from
  the truncated spectrum);
* ``dsb``: spectrum balance with adequacy-driven polynomial order;
* ``integral``: the integral criteria on the Legendre basis and with direct
  numerical areas, the pre-image the Legendre fit and the block fit;
* ``streaming``: the recursive (filter) forms of those criteria;
* ``scale``: their map-reduce and batched forms.

Nothing here receives new development. Depends on ``dtfit`` only.
"""

__version__ = "0.1.0"

from .book import (  # noqa: E402
    dsb_balance, eac_areas, intermediate_polynomial, lsi_integral_monomial,
    monomial_weight_matrix,
)
from .dsb import find_degree, fit_dsb  # noqa: E402

__all__ = [
    "dsb_balance", "eac_areas", "intermediate_polynomial",
    "lsi_integral_monomial", "monomial_weight_matrix", "find_degree",
    "fit_dsb", "__version__",
]
