from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from dtfit.helpers import FittingOptions


@dataclass
class ModelData:
    """Container for data used in different fitting modes."""

    data_x: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Independent variable data points.
    data_y: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Dependent variable data points (observations).
    data_fitted: Optional[np.ndarray] = field(
        default=None
    )  # Fitted values after applying the nonlinear fitting method.

    degree: Optional[int] = field(
        default=None
    )  # Degree of the polynomial used for fitting.
    coeffs: Optional[np.ndarray] = field(
        default=None
    )  # Coefficients that are generated during fitting.
    model: Optional[Callable[[float], float]] = field(
        default=None
    )  # Callable model function that can be evaluated on data_x for predictions.

    nonline_expr: Optional[str] = field(
        default=None
    )  # Literal expression for the nonlinear model.
    nonline_var: Optional[str] = field(
        default=None
    )  # Literal main variable for the nonlinear model.
    nonline_rank: Optional[int] = field(
        default=None
    )  # Number or terms inside of the nonlinear model formula.

    options: FittingOptions = field(
        default_factory=FittingOptions
    )  # Additional options for fitting methods or data generation.

    def copy(self, options: Optional[FittingOptions] = None) -> "ModelData":
        """Creates a copy of the ModelData instance."""
        return ModelData(
            data_x=self.data_x.copy(),
            data_y=self.data_y.copy(),
            data_fitted=(
                self.data_fitted.copy()
                if self.data_fitted is not None
                else None
            ),
            degree=self.degree,
            coeffs=self.coeffs.copy() if self.coeffs is not None else None,
            model=self.model,
            nonline_expr=self.nonline_expr,
            nonline_var=self.nonline_var,
            nonline_rank=self.nonline_rank,
            options=options if options is not None else self.options,
        )
