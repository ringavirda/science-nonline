"""scikit-learn compatible estimator wrapping the dtfit batch methods.

``NonlineRegressor`` exposes the LSI / EDA / DSB fitters through the standard
``fit`` / ``predict`` / ``score`` API with ``get_params``/``set_params`` from
``BaseEstimator``, so it composes with ``sklearn.pipeline.Pipeline``,
``GridSearchCV`` and ``cross_val_score``.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

import sympy as sp

from dtfit.helpers import FittingOptions
from dtfit.extra import fit_lsi, fit_eda, nonline_fit, find_degree, poly_fit
from dtfit.extra.dt import model_params


class NonlineRegressor(RegressorMixin, BaseEstimator):
    """Fit a model that is nonlinear in its parameters to 1-D data.

    Args:
        expr: Model expression, e.g. ``"a0 + a1*x + a2*exp(a3*x)"``.
        var: Main variable name in ``expr`` (the single input feature).
        method: ``"lsi"``, ``"eda"`` or ``"dsb"``.
        k_star: (LSI) number of spectral discretes to match.
        alpha: (LSI) discrete weight decay ``w_i = exp(-alpha*i)``.
        filter_data: (LSI) apply a Savitzky-Golay pre-filter.
        bounds: (LSI) optional per-parameter ``(min, max)`` bounds; enables a
            global search.
        active_ratio: (EDA) leading fraction of data used for window placement.
        poly_degree: (DSB) polynomial degree for the required pre-fit; if
            ``None`` it is selected automatically (BIC).
        p0: Optional initial guess for the parameters.

    Fitted attributes:
        coef_: Fitted coefficients (ordered by parameter name).
        model_: Callable model evaluated at the fitted coefficients.
        n_features_in_: Number of input features (always 1).
    """

    def __init__(
        self,
        expr,
        var="x",
        method="lsi",
        k_star=5,
        alpha=0.2,
        filter_data=True,
        bounds=None,
        active_ratio=0.8,
        poly_degree=None,
        p0=None,
    ):
        self.expr = expr
        self.var = var
        self.method = method
        self.k_star = k_star
        self.alpha = alpha
        self.filter_data = filter_data
        self.bounds = bounds
        self.active_ratio = active_ratio
        self.poly_degree = poly_degree
        self.p0 = p0

    def _as_1d(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != 1:
            raise ValueError(
                "NonlineRegressor supports a single input feature; got "
                f"{X.shape[1]}."
            )
        return X

    def fit(self, X, y) -> "NonlineRegressor":
        X = self._as_1d(X)
        x = X[:, 0]
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_in_ = 1
        options = FittingOptions()

        if self.method == "lsi":
            result = fit_lsi(
                x,
                y,
                self.expr,
                self.var,
                options,
                k_star=self.k_star,
                alpha=self.alpha,
                filter_data=self.filter_data,
                bounds=self.bounds,
                p0=self.p0,
            )
        elif self.method == "eda":
            result = fit_eda(
                x,
                y,
                self.expr,
                self.var,
                options,
                active_ratio=self.active_ratio,
                p0=self.p0,
            )
        elif self.method == "dsb":
            # The polynomial must carry at least as many coefficients as the
            # nonlinear spectrum, else the transfer system is underdefined.
            n_params = len(model_params(sp.sympify(self.expr), sp.Symbol(self.var)))
            min_degree = max(1, n_params - 1)
            degree = self.poly_degree
            if degree is None:
                degree = find_degree(
                    x, y, method="bic", min_degree=min_degree, options=options
                )
            # Enforce the spectrum-required floor (the model transfer needs at
            # least this many polynomial coefficients to be well-defined).
            degree = max(degree, min_degree)
            poly = poly_fit(x, y, degree, method="lite", options=options)
            result = nonline_fit(
                self.expr,
                self.var,
                "dsb",
                options,
                coeffs_poly=poly.coeffs,
            )
        else:
            raise ValueError(f"Unrecognized method: {self.method!r}")

        self.model_ = result.model
        self.coef_ = np.asarray(result.coeffs, dtype=float)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        x = self._as_1d(X)[:, 0]
        out = np.asarray(self.model_(x), dtype=float)
        if out.ndim == 0:  # constant model -> broadcast
            out = np.full(x.shape, float(out))
        return out
