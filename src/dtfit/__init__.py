"""dtfit -- differential-transformation fitting.

Methods for fitting models that are nonlinear in their parameters
(exponential, transcendental, mixed) for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations. Developed as
part of the author's PhD dissertation.

Public interface:
    NonlineRegressor: scikit-learn compatible estimator (fit/predict/score)
        wrapping the LSI/EDA/DSB methods; composes with sklearn Pipeline and
        GridSearchCV.
    EqualAreasFilter: online/streaming estimator (partial_fit) for real-time
        parameter tracking.
    nonline_fit(): functional fit of a nonlinear model using a chosen method.
    poly_fit(): classical polynomial fitting.
    Model: pipeline-based model that chains data preparation and fitting
        middleware and exposes fit()/predict().

Core dependencies: numpy, scipy, sympy, scikit-learn.
Optional extras: matplotlib (install with `pip install 'dtfit[viz]'`).
"""

from dtfit.infra import *
from dtfit.infra.middlewares import *
from dtfit.infra.metrics import *
from dtfit.extra import *
from dtfit.helpers import *
from dtfit.streaming import EqualAreasFilter
from dtfit.estimators import NonlineRegressor
from dtfit.log import enable_logging, logger

# Note: dtfit.simulation (synthetic data generators, noise, experiment
# middlewares) is deliberately NOT imported here. Import it explicitly when
# running experiments: `from dtfit.simulation import ...`.
