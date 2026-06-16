from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import numpy as np
from dtfit.helpers import echo_if, FittingOptions
from dtfit.extra.stats import rss


def find_degree(
    data_x: np.ndarray,
    data_y: np.ndarray,
    method: str,
    min_degree: int = 1,
    max_degree: int = 12,
    options: FittingOptions = FittingOptions(),
) -> int:
    """
    Determines the optimal polynomial degree for fitting the data.

    Arguments:
        data_x (np.ndarray): Feature values for the dataset. It is required for
        the fitting to work. data_y (np.ndarray): Target values for the dataset.
        It is required for the fitting to work. method (str): Method to use for
        degree selection, either 'direct' or 'crossval'. options (dict,
        optional): Additional options for degree selection, including:
            - "echo" (bool): Whether to print debug information. Defaults to
              False.
            - "echo_method" (Callable[[str], None]): Method to use for printing
              debug information. Defaults to print.
            - "crossval_cv_folds" (int): Number of folds for cross-validation
              when using 'crossval' method. Defaults to 5.

        Returns:
            int: Optimal polynomial degree for fitting the data.
    """
    if method == "bic":
        degree = find_degree_direct(
            data_x, data_y, min_degree, max_degree, criterion="bic"
        )
    elif method == "aic":
        degree = find_degree_direct(
            data_x, data_y, min_degree, max_degree, criterion="aic"
        )
    elif method == "crossval":
        cv_folds = options.crossval_cv_folds
        degree = find_degree_crossval(
            data_x, data_y, min_degree, max_degree, cv_folds=cv_folds
        )
    else:
        raise ValueError(f"Unsupported degree selection method: {method}")

    if degree == max_degree:
        echo_if(
            options,
            f"Warning: Maximum degree {max_degree} reached without finding a better fit.",
        )

    echo_if(options, f"Best polynomial degree selected by {method}: {degree}")

    return degree


def find_degree_direct(
    data_x: np.ndarray,
    data_y: np.ndarray,
    min_degree: int = 1,
    max_degree: int = 12,
    criterion: str = "bic",
) -> int:
    """
    Determine best polynomial degree using information criteria (AIC or BIC)
    computed from residual sum of squares (RSS) to select a parsimonious
    polynomial degree that balances fit quality and model complexity.

    Arguments:
        data_x (np.ndarray): 1-D array of x values. data_y (np.ndarray): 1-D
        array of y values to be fitted. max_degree (int): Maximum polynomial
        degree to consider (default 12). criterion (str): Selection criterion,
        either 'aic' or 'bic' (default 'bic').

    Returns:
        int: Selected polynomial degree (0 means constant).
    """

    if data_y is None or data_y.size == 0:
        raise ValueError("data must be a non-empty 1-D array")

    n = data_y.size
    max_degree = int(min(max_degree, max(0, n - 1)))
    min_degree = 0

    best_degree = min_degree
    best_score = np.inf

    for deg in range(min_degree, max_degree + 1):
        try:
            coeffs = np.polyfit(data_x, data_y, deg)
        except Exception:
            continue
        model = np.poly1d(coeffs)(data_x)
        rss_val = float(rss(data_y, model))

        if rss_val <= 0:
            return deg

        k = deg + 1
        aic = n * np.log(rss_val / n) + 2 * k
        bic = n * np.log(rss_val / n) + k * np.log(n)

        score = aic if criterion.lower() == "aic" else bic

        if score < best_score:
            best_score = score
            best_degree = deg

    return int(best_degree)


def find_degree_crossval(
    data_x, data_y, min_degree: int = 1, max_degree: int = 10, cv_folds: int = 5
):
    """
    Determines the optimal polynomial degree using Cross-Validation.

    Args:
        data_x (np.ndarray): 1-D array of x values. data_y (np.ndarray): 1-D
        array of y values to be fitted. max_degree (int, optional): Maximum
        polynomial degree to consider. Defaults to 10. cv_folds (int, optional):
        Number of folds for cross-validation. Defaults to 5.

    Returns:
        best_degree (int): The degree with the lowest Mean Squared Error
    """
    X = data_x.reshape(-1, 1)
    mse_scores = []

    degrees = range(min_degree, max_degree + 1)

    for degree in degrees:
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree)),
                ("linear_regression", LinearRegression()),
            ]
        )
        scores = cross_val_score(
            model, X, data_y, scoring="neg_mean_squared_error", cv=cv_folds
        )

        mse_scores.append(-np.mean(scores))

    return degrees[np.argmin(mse_scores)]
