"""scikit-learn compatible estimator wrapping the dtfit batch methods.

``NonlineRegressor`` exposes the LSI / EAC / DSB fitters through the standard
``fit`` / ``predict`` / ``score`` API with ``get_params``/``set_params`` from
``BaseEstimator``, so it composes with ``sklearn.pipeline.Pipeline``,
``GridSearchCV`` and ``cross_val_score``.
"""

from typing import Any, cast

import numpy as np
from scipy import sparse as sp_sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    _check_sample_weight,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
    validate_data,
)

import sympy as sp

from dtfit.methods import (
    fit_lsi,
    fit_eac,
    fit_dsb,
    find_degree,
    model_params,
    normalize_p0,
)
from dtfit._pandas import as_series, capture_index, is_dataframe, is_series
from dtfit.types import FittingResult


class NonlineRegressor(RegressorMixin, BaseEstimator):
    """Fit a model that is nonlinear in its parameters to 1-D data.

    Args:
        expr: The model, in any of three equivalent forms (resolved by
            :func:`dtfit.methods.resolve_model`): a SymPy-expression **string**
            (e.g. ``"a0 + a1*x + a2*exp(a3*x)"``), a :class:`sympy.Expr`, or a
            plain Python **callable** ``f(x, *params)``. Defaults to a simple
            affine string so the estimator is constructible with no arguments
            (the scikit-learn estimator contract: ``NonlineRegressor()`` must
            work for ``clone`` / meta-estimator introspection). A callable is
            only supported on the ``"lsi"`` / ``"eac"`` routes -- ``"dsb"`` needs
            a symbolic spectrum and raises at fit time for a callable model.
        var: Main variable name in ``expr`` (the single input feature). For a
            callable model it is a label only.
        param_names: Parameter names for a **callable** model, in signature order
            (the parameters after the leading ``x``); introspected from the
            callable's signature when omitted, and only required when that
            signature cannot be introspected (e.g. a ``*args`` model). For a
            symbolic model it is optional and validated against the names parsed
            from the expression. Stored verbatim (no ``__init__`` validation) and
            forwarded to the fitter. See :func:`dtfit.methods.resolve_model`.
        method: ``"lsi"``, ``"eac"`` or ``"dsb"``.
        k_star: (LSI) number of spectral discretes to match.
        alpha: (LSI) extra exponential down-weight ``exp(-alpha*i)`` on
            high-order discretes; ``0.0`` (the :func:`dtfit.fit_lsi` default)
            relies on the built-in orthonormal weighting alone.
        filter_data: (LSI) apply a Savitzky-Golay pre-filter. Off by default --
            a fitter must not silently modify the user's data (matches
            :func:`dtfit.fit_lsi`).
        bounds: (LSI/EAC) optional parameter bounds: a per-parameter
            ``(min, max)`` pair list in sorted-name order or a partial
            ``{name: (min, max)}`` dict; passed through to the fitter untouched.
            For LSI, fully finite bounds enable a global search.
        active_ratio: (EAC) leading fraction of data used for window placement.
            Defaults to ``1.0`` (all samples, matching :func:`dtfit.fit_eac`).
        poly_degree: (DSB) polynomial degree for the required pre-fit; if
            ``None`` it is selected automatically (BIC).
        p0: Optional initial guess for the parameters: a sequence in
            sorted-name order or a ``{name: value}`` dict (passed through
            untouched).
        random_state: (LSI) seed for the deterministic global / differential-
            evolution search when ``bounds`` are given, so a bounded fit is
            reproducible under ``GridSearchCV``/``clone``. ``None`` uses the
            global RNG.
        robust: (LSI/EAC) robustify the fit via IRLS winsorization of sample
            residuals within ``huber_c`` robust sigmas (see the fitters).
        huber_c: (LSI/EAC) winsorization threshold in residual sigmas for
            ``robust=True``.
        nan_policy: (LSI/EAC) ``"raise"`` (default) rejects non-finite samples;
            ``"omit"`` drops NaN/inf ``(x, y)`` pairs before fitting.
        loss: (EAC) least-squares loss on the window-area residuals (e.g.
            ``"soft_l1"`` for outlier robustness).
        window_mode: (EAC) window placement -- ``"uniform"`` or ``"curvature"``.

    Fitted attributes:
        coef_: Fitted coefficients (ordered by parameter name).
        model_: Callable model evaluated at the fitted coefficients.
        result_: The full :class:`dtfit.FittingResult`, exposing ``cov``,
            ``stderr()``, ``confidence_intervals()``, ``converged`` and the v0.3
            fit-quality stats (``rsquared``, ``aic`` / ``bic``) that the LSI / EAC
            fitters record.
        n_features_in_: Number of input features (always 1).
    """

    # Declared for type checkers; set at fit time by ``validate_data`` /
    # ``fit`` (sklearn populates the ``*_in_`` attributes during validation).
    coef_: np.ndarray
    model_: Any
    result_: FittingResult
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        expr="a0 + a1*x",
        var="x",
        param_names=None,
        method="lsi",
        k_star=5,
        alpha=0.0,
        filter_data=False,
        bounds=None,
        active_ratio=1.0,
        poly_degree=None,
        p0=None,
        random_state=0,
        robust=False,
        huber_c=3.0,
        nan_policy="raise",
        loss="linear",
        window_mode="uniform",
    ):
        self.expr = expr
        self.var = var
        self.param_names = param_names
        self.method = method
        self.k_star = k_star
        self.alpha = alpha
        self.filter_data = filter_data
        self.bounds = bounds
        self.active_ratio = active_ratio
        self.poly_degree = poly_degree
        self.p0 = p0
        self.random_state = random_state
        self.robust = robust
        self.huber_c = huber_c
        self.nan_policy = nan_policy
        self.loss = loss
        self.window_mode = window_mode

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True  # y is mandatory for fit
        # Single-feature estimator: the natural input is a 1-D x vector, and a
        # single-column 2-D X is also accepted (for Pipeline use), so both
        # input-type tags hold. (``one_d_array`` also tells the sklearn check
        # suite to exercise the estimator with 1-D X, as it does for
        # ``IsotonicRegression``.)
        tags.input_tags.one_d_array = True
        # With nan_policy="omit" the LSI/EAC fitters drop non-finite pairs, so
        # NaN input is legitimately accepted.
        tags.input_tags.allow_nan = (
            self.nan_policy == "omit" and self.method in ("lsi", "eac")
        )
        # ``poor_score`` refers specifically to the check suite's shared
        # regression dataset: its informative feature is not column 0, so no
        # single-feature estimator can reach the R^2 > 0.5 the suite asserts.
        tags.regressor_tags.poor_score = True
        return tags

    def __getstate__(self):
        # Copy: BaseEstimator may hand back the live ``__dict__``.
        state = dict(super().__getstate__())
        # ``model_`` is a lambdified closure that does not pickle; drop it and
        # rebuild from ``result_`` on unpickle (:class:`dtfit.FittingResult`
        # pickles cleanly -- it re-lambdifies its model lazily from
        # ``expr`` + ``coeffs``).
        state.pop("model_", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "result_"):
            self.model_ = self.result_.model

    @staticmethod
    def _to_2d(X):
        """Promote a bare 1-D feature vector (array, Series, plain list, ...)
        to a single column, leaving 2-D array-likes (and DataFrames, so column
        names survive) untouched. Sparse inputs pass through so
        ``validate_data`` raises the standard scikit-learn sparse error."""
        if sp_sparse.issparse(X):
            return X
        ndim = getattr(X, "ndim", None)
        if ndim is None:  # plain sequences / duck-typed array wrappers
            X = np.asarray(X)
            ndim = X.ndim
        if ndim == 1:
            return np.asarray(X).reshape(-1, 1)
        return X

    def _reject_multifeature(self, X) -> None:
        n_features = np.shape(X)[1] if np.ndim(X) == 2 else 1
        if n_features != 1:
            raise ValueError(
                "NonlineRegressor supports a single input feature; got "
                f"{n_features}. dtfit's integral criteria are one-dimensional, so "
                "multivariate X (several predictors) is not supported. If instead "
                "you have a 1-D signal that is a sum of components along one axis "
                "(e.g. trend + cycle), compose 1-D models with `+` (see the "
                "'Multivariate data' docs note)."
            )

    def _sample_weight_to_sigma(self, sample_weight, X, order):
        """Translate sklearn ``sample_weight`` to the fitters' per-sample ``sigma``.

        Returns ``None`` when no weights are given (the unchanged, equal-weight
        path). Otherwise validates the weights against ``X`` (sklearn's
        :func:`~sklearn.utils.validation._check_sample_weight`), reorders them to
        match the x-sorted samples, and maps them to ``sigma = 1 / sqrt(weight)``
        -- the relative measurement standard deviation the LSI / EAC weighted
        integral fit consumes. A zero-weight sample is given a huge (but finite)
        ``sigma`` so it drops out of the fit without violating the fitters'
        strictly-positive-``sigma`` contract; negative weights, and all-zero
        weights, raise.

        Only the integral routes weight samples: ``method="dsb"`` has no
        per-sample weighting and raises here when a weight is supplied.
        """
        if sample_weight is None:
            return None
        if is_series(sample_weight):
            # A pandas Series of weights -> its float values (positional, like
            # every other array-like sample_weight); sklearn's _check_sample_weight
            # then validates the length against X.
            sample_weight = np.asarray(sample_weight, dtype=float)
        if self.method == "dsb":
            raise ValueError(
                "sample_weight is not supported by method='dsb' (the DSB "
                "transfer solve has no per-sample weighting); use method='lsi' "
                "or 'eac'."
            )
        sw = _check_sample_weight(sample_weight, X, dtype=np.float64)[order]
        if np.any(sw < 0.0):
            raise ValueError("sample_weight must be non-negative.")
        if not np.any(sw > 0.0):
            raise ValueError(
                "sample_weight must contain at least one non-zero (strictly "
                "positive) weight; all sample weights are zero."
            )
        with np.errstate(divide="ignore"):
            sigma = 1.0 / np.sqrt(sw)
        nonfinite = ~np.isfinite(sigma)
        if nonfinite.any():
            # weight == 0 -> 1/sqrt(0) = inf: give that sample a huge but finite
            # sigma so it is effectively ignored, keeping every sigma finite and
            # positive as the fitters require.
            finite_max = (
                float(np.max(sigma[~nonfinite])) if (~nonfinite).any() else 1.0
            )
            sigma[nonfinite] = 1e6 * finite_max
        return sigma

    def fit(self, X, y, sample_weight=None) -> "NonlineRegressor":
        """Fit the model to ``(X, y)``.

        Args:
            X: The single input feature (a 1-D vector or a single-column 2-D
                array / DataFrame).
            y: The target values.
            sample_weight: Optional per-sample weights (the scikit-learn
                convention). Translated to the integral fitters' per-sample
                ``sigma = 1 / sqrt(sample_weight)`` and forwarded with
                ``absolute_sigma=False`` (relative weights), so a down-weighted
                sample pulls the integral fit less without being dropped. Only the
                ``"lsi"`` / ``"eac"`` routes support it; ``"dsb"`` has no
                per-sample weighting and raises when a weight is given. A weight of
                ``0`` effectively ignores its sample (a huge ``sigma``); all-zero
                weights raise.
        """
        X = self._to_2d(X)
        # cast: sklearn's validate_data stub mistypes the array argument as str.
        # Validate first so sparse / non-finite / empty inputs get the standard
        # scikit-learn errors; only then apply the single-feature constraint.
        if self.__sklearn_tags__().input_tags.allow_nan:
            # nan_policy="omit": the fitter drops non-finite (x, y) pairs
            # itself, so relax the finite check (which guards X and y) by
            # validating the two separately.
            X, y = cast(Any, validate_data)(
                self,
                X,
                y,
                reset=True,
                validate_separately=(
                    {"dtype": np.float64, "ensure_all_finite": False},
                    {
                        "dtype": np.float64,
                        "ensure_2d": False,
                        "ensure_all_finite": False,
                    },
                ),
            )
            # Mirror the default path's y shape rules (1-D or single column).
            y = column_or_1d(y, warn=True)
            check_consistent_length(X, y)
        else:
            X, y = cast(Any, validate_data)(
                self, X, y, reset=True, dtype=np.float64, y_numeric=True
            )
        self._reject_multifeature(X)
        x = np.asarray(X, dtype=float)[:, 0]
        y = np.asarray(y, dtype=float).ravel()
        # scikit-learn contract: sample order must not matter. The integral
        # fitters (LSI quadrature, EAC windows) need monotone x, so sort the
        # (x, y) pairs -- a no-op for already-ordered curve data.
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]

        # v0.3: a callable model f(x, *params) is supported on the integral
        # routes only; DSB needs a symbolic spectrum to balance.
        model_is_callable = callable(self.expr)

        # v0.3: per-sample weights -> per-sample sigma = 1/sqrt(weight) for the
        # weighted integral fit (relative weights, the sklearn convention, so
        # absolute_sigma stays False). Aligned to the x-sorted samples.
        sigma = self._sample_weight_to_sigma(sample_weight, X, order)

        if self.method == "lsi":
            result = fit_lsi(
                x,
                y,
                self.expr,
                self.var,
                param_names=self.param_names,
                k_star=self.k_star,
                alpha=self.alpha,
                filter_data=self.filter_data,
                bounds=self.bounds,
                p0=self.p0,
                sigma=sigma,
                random_state=self.random_state,
                robust=self.robust,
                huber_c=self.huber_c,
                nan_policy=self.nan_policy,
            )
        elif self.method == "eac":
            result = fit_eac(
                x,
                y,
                self.expr,
                self.var,
                param_names=self.param_names,
                active_ratio=self.active_ratio,
                window_mode=self.window_mode,
                bounds=self.bounds,
                loss=self.loss,
                robust=self.robust,
                huber_c=self.huber_c,
                p0=self.p0,
                sigma=sigma,
                nan_policy=self.nan_policy,
            )
        elif self.method == "dsb":
            if model_is_callable:
                raise ValueError(
                    "method='dsb' requires a symbolic model (a sympy-expression "
                    "string or sympy.Expr): a Python callable carries no "
                    "differential spectrum to balance. Use method='lsi' or "
                    "'eac' for a callable model."
                )
            # The polynomial must carry at least as many coefficients as the
            # nonlinear spectrum, else the transfer system is underdefined.
            expr_sym = cast(sp.Expr, sp.sympify(self.expr))
            params = model_params(expr_sym, sp.Symbol(self.var))
            n_params = len(params)
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
            # fit_dsb takes p0 positionally only; normalize here so the
            # estimator's documented dict form works on every method route.
            p0 = normalize_p0(self.p0, [str(p) for p in params])
            result = fit_dsb(coeffs_poly, self.expr, self.var, p0=p0)
        else:
            raise ValueError(f"Unrecognized method: {self.method!r}")

        self.result_ = result
        self.model_ = result.model
        self.coef_ = np.asarray(result.coeffs, dtype=float)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict ``y`` for the single input feature ``X``.

        pandas in -> pandas out: when ``X`` is a pandas ``Series`` (or a
        single-column ``DataFrame``) the prediction is returned as a ``Series``
        aligned to ``X``'s index; the predicted values are unchanged. An ndarray
        / list input returns an ndarray exactly as before (bit-identical).
        """
        check_is_fitted(self, "model_")
        # pandas in -> pandas out: capture the sample index before validation
        # coerces X to a plain array, so the prediction can be realigned to it.
        # (A multi-column DataFrame is rejected by validate_data below, so the
        # captured index is only ever used for a Series / single-column frame.)
        x_index = capture_index(X) if (is_series(X) or is_dataframe(X)) else None
        X = self._to_2d(X)
        # cast: sklearn's validate_data stub mistypes the array argument as str.
        X = cast(Any, validate_data)(self, X, reset=False, dtype=np.float64)
        x = np.asarray(X, dtype=float)[:, 0]
        out = np.asarray(self.model_(x), dtype=float)
        if out.ndim == 0:  # constant model -> broadcast
            out = np.full(x.shape, out.item())
        # as_series returns the plain ndarray unchanged when x_index is None
        # (non-pandas input) or pandas is absent, so the ndarray path is intact.
        return as_series(out, x_index)
