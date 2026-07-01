"""scikit-learn compatible estimator wrapping the dtfit batch methods.

``NonlineRegressor`` exposes the LSI / EAC / DSB fitters through the standard
``fit`` / ``predict`` / ``score`` API with ``get_params``/``set_params`` from
``BaseEstimator``, so it composes with ``sklearn.pipeline.Pipeline``,
``GridSearchCV`` and ``cross_val_score``.
"""

from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

import sympy as sp

from dtfit.methods import fit_lsi, fit_eac, fit_dsb, find_degree, model_params


class NonlineRegressor(RegressorMixin, BaseEstimator):
    """Fit a model that is nonlinear in its parameters to 1-D data.

    Args:
        expr: Model expression, e.g. ``"a0 + a1*x + a2*exp(a3*x)"``. Defaults to a
            simple affine model so the estimator is constructible with no
            arguments (the scikit-learn estimator contract: ``NonlineRegressor()``
            must work for ``clone`` / meta-estimator introspection).
        var: Main variable name in ``expr`` (the single input feature).
        method: ``"lsi"``, ``"eac"`` or ``"dsb"``.
        k_star: (LSI) number of spectral discretes to match.
        alpha: (LSI) discrete weight decay ``w_i = exp(-alpha*i)``.
        filter_data: (LSI) apply a Savitzky-Golay pre-filter.
        bounds: (LSI) optional per-parameter ``(min, max)`` bounds; enables a
            global search.
        active_ratio: (EAC) leading fraction of data used for window placement.
        poly_degree: (DSB) polynomial degree for the required pre-fit; if
            ``None`` it is selected automatically (BIC).
        p0: Optional initial guess for the parameters.
        random_state: (LSI) seed for the deterministic global / differential-
            evolution search when ``bounds`` are given, so a bounded fit is
            reproducible under ``GridSearchCV``/``clone``. ``None`` uses the
            global RNG.

    Fitted attributes:
        coef_: Fitted coefficients (ordered by parameter name).
        model_: Callable model evaluated at the fitted coefficients.
        n_features_in_: Number of input features (always 1).
    """

    # Declared for type checkers; set at fit time by ``validate_data`` /
    # ``fit`` (sklearn populates the ``*_in_`` attributes during validation).
    coef_: np.ndarray
    model_: Any
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        expr="a0 + a1*x",
        var="x",
        method="lsi",
        k_star=5,
        alpha=0.2,
        filter_data=True,
        bounds=None,
        active_ratio=0.8,
        poly_degree=None,
        p0=None,
        random_state=0,
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
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True  # y is mandatory for fit
        return tags

    @staticmethod
    def _to_2d(X):
        """Promote a bare 1-D feature vector to a single column, leaving 2-D
        array-likes (and DataFrames, so column names survive) untouched."""
        if getattr(X, "ndim", None) == 1:
            return np.asarray(X, dtype=float).reshape(-1, 1)
        return X

    def _reject_multifeature(self, X) -> None:
        n_features = np.shape(X)[1] if np.ndim(X) == 2 else 1
        if n_features != 1:
            raise ValueError(
                "NonlineRegressor supports a single input feature; got "
                f"{n_features}."
            )

    def fit(self, X, y) -> "NonlineRegressor":
        X = self._to_2d(X)
        self._reject_multifeature(X)
        # cast: sklearn's validate_data stub mistypes the array argument as str.
        X, y = cast(Any, validate_data)(
            self, X, y, reset=True, dtype=np.float64, y_numeric=True
        )
        x = np.asarray(X, dtype=float)[:, 0]
        y = np.asarray(y, dtype=float).ravel()

        if self.method == "lsi":
            result = fit_lsi(
                x,
                y,
                self.expr,
                self.var,
                k_star=self.k_star,
                alpha=self.alpha,
                filter_data=self.filter_data,
                bounds=self.bounds,
                p0=self.p0,
                random_state=self.random_state,
            )
        elif self.method == "eac":
            result = fit_eac(
                x,
                y,
                self.expr,
                self.var,
                active_ratio=self.active_ratio,
                p0=self.p0,
            )
        elif self.method == "dsb":
            # The polynomial must carry at least as many coefficients as the
            # nonlinear spectrum, else the transfer system is underdefined.
            expr_sym = cast(sp.Expr, sp.sympify(self.expr))
            n_params = len(model_params(expr_sym, sp.Symbol(self.var)))
            min_degree = max(1, n_params - 1)
            degree = self.poly_degree
            if degree is None:
                degree = find_degree(x, y, method="bic")
            # Enforce the spectrum-required floor (the model transfer needs at
            # least this many polynomial coefficients to be well-defined).
            degree = max(degree, min_degree)
            # Ascending-order polynomial coefficients = the data's Maclaurin
            # spectrum DSB balances against (np.polyfit returns descending).
            coeffs_poly = np.polyfit(x, y, degree)[::-1]
            result = fit_dsb(coeffs_poly, self.expr, self.var, p0=self.p0)
        else:
            raise ValueError(f"Unrecognized method: {self.method!r}")

        self.model_ = result.model
        self.coef_ = np.asarray(result.coeffs, dtype=float)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = self._to_2d(X)
        # cast: sklearn's validate_data stub mistypes the array argument as str.
        X = cast(Any, validate_data)(self, X, reset=False, dtype=np.float64)
        x = np.asarray(X, dtype=float)[:, 0]
        out = np.asarray(self.model_(x), dtype=float)
        if out.ndim == 0:  # constant model -> broadcast
            out = np.full(x.shape, out.item())
        return out
