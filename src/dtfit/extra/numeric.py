import warnings
from scipy.optimize import curve_fit, OptimizeWarning

import numpy as np
from typing import Callable
from dtfit.helpers import echo_if, FittingResult, FittingOptions


def numeric_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    solution0: np.ndarray,
    model: Callable,
    method: str = "lm",
    maxfev: int = 10000,
    options: FittingOptions = FittingOptions(),
) -> FittingResult:
    """Utilizes numeric methods to optimize given solution using original data.
    Requires a spectrum object and already calculated solution to operate.

    Arguments:
        data_y (np.ndarray): Original data that was used for fitting the solution.
        data_x (np.ndarray): Feature values for the dataset.
        solution0 (np.ndarray): Initial guess for the parameters to be optimized.
        model (Callable): A function representing the model to be optimized.
        method (str, optional): Optimization method to use, can be 'lm', 'trf',
        or 'dogbox'. Defaults to "lm".
        maxfev (int, optional): Maximum number of function evaluations. Defaults
        to 10000.
        options (FittingOptions, optional): An object containing options for the 
        fitting process. Defaults to a new FittingOptions instance.

    Returns:
        FittingResult: Object containing the optimized results and coefficients.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        solution, _ = curve_fit(
            model,
            data_x,
            data_y,
            p0=solution0,
            method=method,
            maxfev=maxfev,
            bounds=(-np.inf, np.inf),
        )
    coeffs = solution[::-1]
    fitted_model = lambda x: model(x, *solution)

    echo_if(
        options,
        f"Numeric optimization using method '{method}' completed. Optimized coefficients:",
        coeffs,
    )

    return FittingResult(model=fitted_model, coeffs=coeffs)
