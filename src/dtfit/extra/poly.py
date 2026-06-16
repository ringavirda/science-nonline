from sklearn.linear_model import Lasso

import numpy as np
from dtfit.helpers import echo_if, FittingResult, FittingOptions


def poly_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    degree: int,
    method: str,
    options: FittingOptions = FittingOptions(),
) -> FittingResult:
    """Fits a polynomial model to the given data.

    Arguments:
        data_x (np.ndarray): Feature values for the dataset. It is required for
        the fitting to work. data_y (np.ndarray): Target values for the dataset.
        It is required for the fitting to work. degree (int): Manually specifies
        polynomial degree for fitting. method (str): Method to use for fitting,
        either 'lite' or 'lasso'. options (FittingOptions, optional): An object
        containing additional options for fitting, including:
            - "echo" (bool): Whether to print debug information. Defaults to
              False.
            - "echo_method" (Callable[[str], None]): Method to use for printing
              debug information. Defaults to print.
    Returns:
        FittingResult: Object containing the fitted results and coefficients.
    """

    if method == "lite":
        result = poly_fit_lite(data_x, data_y, degree)
    elif method == "lasso":
        alpha = options.lasso_alpha
        max_iter = options.lasso_max_iter
        result = poly_fit_lasso(
            data_x, data_y, degree, alpha=alpha, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unsupported fitting method: {method}")

    echo_if(
        options,
        f"Fitted polynomial coefficients (degree {degree}):",
        result.coeffs,
    )

    return result


def poly_fit_lite(
    data_x: np.ndarray, data_y: np.ndarray, degree: int
) -> FittingResult:
    """
    Lightweight version of `poly_fit` without other extra infrastructure.

    Arguments:
        data_x (np.ndarray): Feature values for the dataset. It is required for
        the fitting to work. data_y (np.ndarray): Target values for the dataset.
        It is required for the fitting to work. degree (int): Manually specifies
        polynomial degree for fitting.

    Returns:
        FittingResult: Object containing the fitted results and coefficients.
    """
    coeffs = np.polyfit(data_x, data_y, degree)
    poly = np.poly1d(coeffs)

    return FittingResult(model=poly, coeffs=coeffs[::-1])


def poly_fit_lasso(
    data_x: np.ndarray,
    data_y: np.ndarray,
    degree: int,
    alpha: float = 0.1,
    max_iter: int = 100000,
) -> FittingResult:
    """
    Fits a polynomial model using Lasso regularization.

    Args:
        data_x (np.ndarray): Feature values for the dataset. data_y
        (np.ndarray): Target values for the dataset. degree (int): Polynomial
        degree for fitting. alpha (float, optional): Regularization strength for
        Lasso. Defaults to 0.1. max_iter (int, optional): Maximum number of
        iterations for the Lasso solver. Defaults to 100000. echo (bool,
        optional): Whether to print debug information. Defaults to False. alpha
        (float, optional): Regularization strength for Lasso. Defaults to 0.1.
        max_iter (int, optional): Maximum number of iterations for the Lasso
        solver. Defaults to 100000.

    Returns:
        FittingResult: Object containing the fitted results and coefficients.
    """

    X = np.vander(
        data_x if data_x is not None else np.arange(data_y.size),
        N=degree + 1,
        increasing=True,
    )

    scales = np.max(np.abs(X), axis=0)
    scales[scales == 0] = 1.0
    Xs = X / scales

    lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    lasso.fit(Xs, data_y)

    coeffs_scaled = lasso.coef_
    coeffs_fitted = coeffs_scaled / scales
    poly = np.poly1d(coeffs_fitted[::-1])

    return FittingResult(model=poly, coeffs=coeffs_fitted)


def filter_poly_coeffs(poly_c: np.ndarray, cutoff: float = 3.0) -> np.ndarray:
    """
    Filters all given coefficients in the array, assuming that they belong to
    polynomial model. Removes all coeffs that are too small to meaningfully
    influence the results. Cutoff is power lower than `coeff degree * -3 - 1`.

    Args:
        poly_c (np.ndarray): An array of polynomial coefficients to filter.
        cutoff (float): The cutoff value for filtering coefficients.

    Returns:
        np.ndarray: Collection of filtered coeffs.
    """
    poly_c_filtered = np.zeros(poly_c.size)

    for i in np.arange(poly_c.size):
        pos = np.log10(np.abs(poly_c[i]))
        cutoff_val = i * -cutoff - 1
        dec = pos > cutoff_val
        poly_c_filtered[i] = poly_c[i] if dec else 0

    return poly_c_filtered
