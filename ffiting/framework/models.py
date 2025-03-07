"""Classes used to represent fitted data and models."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Callable, Optional, Self

import ffiting.framework as fr
from ..common import np, sp
from . import FittingModes, FittingOptions


@dataclass(frozen=True)
class ModelLite:
    """Readonly fitting result, containing both original and processed data,
    as well as callable model function.
    """

    var_main: str  # Primary free variable
    expr_raw: str  # Generic string model representation
    expr_sp: sp.Expr  # Symbolic object model representation
    model: Callable[[float | np.ndarray], float]  # Labdified fitted model
    coeffs: list[float]  # Model weights

    @property
    def label(self) -> str:
        """Returns a string representation of expression with coeffs substituted."""
        expr_coeffs: list[sp.Symbol] = sorted(
            self.expr_sp.free_symbols, key=lambda s: s.name
        )
        expr_coeffs.remove(sp.Symbol(self.var_main))
        expr: sp.Expr = self.expr_sp
        for i, a in enumerate(expr_coeffs):
            expr = expr.subs(a, self.coeffs[i])
        return str(expr)

    def __call__(self, data: float | np.ndarray) -> float | np.ndarray:
        """This overload calls internal models to generate fitted values."""
        return self.model(data)


class Model:
    """Comprehensive model class, containing additional analysis methods, such
    as metrics calculations, terms and spectra extraction, and additional
    fitting options and predictions.
    """

    def __init__(
        self,
        options: FittingOptions,
    ) -> None:
        # Private fields
        self.__options = options
        self.__expr: Optional[sp.Expr] = (
            options.expr_sp
            if options.expr_sp is not None
            else (None if options.expr_raw == "" else sp.parse_expr(options.expr_raw))
        )
        self.__data_raw: Optional[np.ndarray] = None
        self.__data_fit: Optional[np.ndarray] = None
        self.__model: Optional[Callable[[float | np.ndarray], float | np.ndarray]] = (
            None
        )
        self.__coeffs: Optional[list[float]] = None

    def is_fitted(self) -> bool:
        """Contains `True` when this model was already fitted, `False` otherwise."""
        return self.__model is not None

    def apply_fitted(self, model_new: Self | ModelLite) -> None:
        """Change internal state of this model to that of the given one if it is fitted."""
        if model_new.model is None:
            raise ValueError("Cannot apply unfitted model.")
        if not isinstance(model_new, ModelLite):
            self.__options = model_new.options
            self.__data_raw = model_new.data_raw
            self.__data_fit = model_new.data_fit
        else:
            self.__options.expr_raw = model_new.expr_raw

        self.__model = model_new.model
        self.__coeffs = model_new.coeffs
        self.__expr = model_new.expr_sp

    def fit(
        self, data: np.ndarray, options: Optional[FittingOptions] = None
    ) -> ModelLite:
        """Internal call to fitting functions that perform smoothing and return
        optimized values. Specific method and parameters can be specified through
        the `options` parameter. It can update internal fields of the instance,
        after which it qualifies as being fitted.

        Arguments:
            data (ndarray): Training dataset to use later.
            options (FittingOptions): An instance that describes how to conduct the
            operation:
                rank (int): manually specifies polynomial rank for fitting. Attempts to
                figure out the required rank if value is set to default of "0".
                expr_raw (str): string representation of a mathematical model using standard
                notation for inline expressions.
                raise_rank (bool): enables or disables automatic rank raise algorithm. By
                default is set to `False`.
                update_model (bool): changes internal values of the fields for the instance if
                set to `True`, which is default.

        Returns:
            ModelLite: An object for the achieved fit.
        """
        if options is None:
            options = self.__options

        if options.fitting_mode == FittingModes.POLY:
            fitted = fr.poly_fit_(data, options)
        elif options.fitting_mode == FittingModes.DSB or FittingModes.AUTO:
            fitted = fr.nonline_fit_(data, options)
        else:
            raise NotImplementedError("Stay tuned.")

        if options.update_model:
            self.__data_raw = data
            self.__data_fit = fitted(data)
            self.apply_fitted(fitted)
        return fitted

    @staticmethod
    def __only_fitted(func: Callable[[Self], Any]) -> Callable[[Self], Any]:
        """Internal class decorator to guard against methods being called before
        the model was fitted."""

        def inner(self) -> Any:
            if not self.is_fitted():
                raise RuntimeError("Unable to call unfitted model.")
            func(self)

        return inner

    @__only_fitted
    def predict(self, data_x: float | np.ndarray) -> float | np.ndarray:
        """Use internal fitted model to generate predicted results.

        Arguments:
            data_x (float | np.ndarray): base values for prediction.

        Returns:
            float | np.ndarray: can be value of both types, depending on the
            given data_x.
        """
        return self.__model(data_x)

    @property
    def options(self) -> FittingOptions:
        """Return fitting options that are currently in use."""
        return self.__options

    @options.setter
    def options(self, options_new: FittingOptions) -> None:
        """Updates internal options to the new ones given."""
        self.__options = options_new
        self.__expr: Optional[sp.Expr] = (
            options_new.expr_sp
            if options_new.expr_sp is not None
            else (
                None
                if options_new.expr_raw == ""
                else sp.parse_expr(options_new.expr_raw)
            )
        )

    @property
    def data_raw(self) -> np.ndarray | None:
        """Return the underlining training dataset."""
        return self.__data_raw

    @property
    def data_fit(self) -> np.ndarray | None:
        """Return data generated with fitted model."""
        return self.__data_fit

    @property
    def coeffs(self) -> list[float] | None:
        """Return weights of the fitted model."""
        return self.__coeffs

    @property
    def expr_sp(self) -> sp.Expr:
        """Return symbolic representation of the fitted model."""
        return self.__expr

    @__only_fitted
    def __call__(self, data: float | np.ndarray) -> float | np.ndarray:
        """This overload calls internal models to generate fitted values."""
        if data is float:
            return self.__model(data)
        if isinstance(data, (Collection, np.ndarray)):
            fitted = np.ndarray(data.shape)
            with np.nditer(data, flags=["f_index"]) as it:
                for val in it:
                    fitted[it.index] = self.__model(val)
            return fitted

        raise ValueError("Invalid argument, model can only accept float or array-like.")

    @__only_fitted
    def as_lambda(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Returns a lambda that generates fitted model values."""
        return self.__model

    @__only_fitted
    def as_lite(self) -> ModelLite:
        """Returns a simplified immutable representation of the model."""
        return ModelLite(
            self.__options.var_main,
            self.__data_raw,
            self.__data_fit,
            self.__model,
            self.__coeffs,
        )
