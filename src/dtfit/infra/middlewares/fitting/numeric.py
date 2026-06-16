import numpy as np
import sympy as sp
from typing import Optional
from dtfit.helpers import echo_if, collapse
from dtfit.infra import ModelData, Middleware
from dtfit.extra import filter_poly_coeffs, numeric_fit


class NumericOptimizeMw(Middleware):
    """
    Middleware for optimizing the fitting process. Priority is given to
    nonlinear fitting if the target is not specified and both polynomial and
    nonlinear data are available.
    """

    def __init__(
        self,
        solution0: Optional[np.ndarray] = None,
        method: str = "lm",
        maxfev: int = 10000,
        filter: bool = False,
    ):
        super().__init__()
        self.solution0 = solution0
        self.method = method
        self.maxfev = maxfev
        self.filter = filter

    def process(self, data: ModelData) -> ModelData:
        if data.nonline_expr is not None and data.nonline_var is not None:
            echo_if(
                data.options,
                "Performing nonlinear fitting optimization using numeric methods.",
            )
            if data.coeffs is None and self.solution0 is None:
                raise ValueError(
                    f"Starting coefficients weren't provided for nonlinear fitting."
                )
            solution0 = (
                self.solution0 if self.solution0 is not None else data.coeffs
            )

            var = sp.Symbol(data.nonline_var)
            nonline: sp.Expr = sp.parse_expr(data.nonline_expr)
            nonline_coeffs = sorted(nonline.free_symbols, key=lambda s: str(s))
            nonline_coeffs.remove(var)
            submodel = sp.lambdify([*nonline_coeffs, var], nonline, "scipy")

            def model(x, *coeffs):
                return submodel(*coeffs, x)

        else:
            echo_if(
                data.options,
                "Nonlinear model information is incomplete."
                + "Performing polynomial fitting optimization using numeric methods.",
            )
            if data.coeffs is not None:
                solution0 = data.coeffs
            elif self.solution0 is not None:
                solution0 = self.solution0
            else:
                raise ValueError(f"No starting coefficients were provided.")
            model = lambda x, *coeffs: np.polyval(coeffs, x)

        if solution0 is None:
            raise ValueError(f"Starting coefficients weren't provided.")

        fitted = numeric_fit(
            data.data_x,
            data.data_y,
            solution0,
            model,
            self.method,
            self.maxfev,
            data.options,
        )

        if self.filter:
            fitted.coeffs = filter_poly_coeffs(fitted.coeffs)
            echo_if(
                data.options,
                f"Filtered polynomial coefficients (degree {len(fitted.coeffs) - 1}):",
                fitted.coeffs,
            )

        data.coeffs = fitted.coeffs
        data.model = fitted.model
        data.data_fitted = collapse(fitted.model)(data.data_x)

        return data
