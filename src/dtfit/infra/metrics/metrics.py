import scipy.stats as st

import numpy as np
from typing import Callable
from dataclasses import dataclass
from dtfit.extra.stats import *


@dataclass(frozen=True)
class DataMetrics:
    """
    Readonly container for main model stats, that can be used to evaluate
    fitting effectiveness.
    """

    std_div: float  # Standard Deviation
    std_err: float  # Standard Error
    lin_div: float  # Linear Deviation
    rse: float  # Residual Squared Error
    mse: float  # Mean Squared Error
    r_sq: float  # Determination Coefficient
    corr: float  # Correlation Coefficient
    ccord: float  # Concordance Coefficient

    def __str__(self) -> str:
        return (
            f"Standard Div: {self.std_div:.3f}\n"
            f"Standard Err: {self.std_err:.3f}\n"
            f"Linear Div: {self.lin_div:.3f}\n"
            f"Residual SE: {self.rse:.3f}\n"
            f"Mean SE: {self.mse:.3f}\n"
            f"Determination Coeff: {self.r_sq:.3f}\n"
            f"Correlation Coeff: {self.corr:.3f}\n"
            f"Concordance Coeff: {self.ccord:.3f}"
        )


def get_metrics(
    data_origin: np.ndarray, data_fitted: np.ndarray
) -> DataMetrics:
    """
    Calculates most used metrics for given model. Uses implementations from
    `Metrics` container and from `numpy` directly.

    Arguments:
        data_origin (np.ndarray): Vector of original data points.
        data_fitted (np.ndarray): Vector of fitted data points.

    Returns:
        DataMetrics: Data object with fields populated using data from given
        model.
    """
    return DataMetrics(
        rse=rse(data_origin, data_fitted),
        mse=mse(data_origin, data_fitted),
        r_sq=r_sq(data_origin, data_fitted),
        lin_div=lin_div(data_origin, data_fitted),
        std_div=std_div(data_origin, data_fitted),
        std_err=std_err(data_origin, data_fitted),
        corr=corr(data_origin, data_fitted),
        ccord=ccord(data_origin, data_fitted),
    )


@dataclass
class DataDistribution:
    """Container for distribution fitting results."""

    name: str
    p_value: float
    pdf: Callable[[np.ndarray], np.ndarray]


def get_distribution(
    data: np.ndarray,
) -> DataDistribution:
    """Fits several known distributions to the data and selects the best one
    based on the p-value from the Kolmogorov-Smirnov test.
    Arguments:
        data (np.ndarray): Input data to fit distributions to.
    Returns:
        DataDistribution: An object containing the name of the best-fitting
        distribution and its p-value.
    """

    dist_names = [
        "norm",
        "exponweib",
        "weibull_max",
        "weibull_min",
        "pareto",
        "genextreme",
    ]
    dist_results = []
    params = {}

    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        _, p = st.kstest(data, dist_name, args=param)
        dist_results.append((dist_name, p))

    best_dist, best_p = max(dist_results, key=lambda item: item[1])
    return DataDistribution(
        name=best_dist,
        p_value=best_p,
        pdf=lambda x: getattr(st, best_dist).pdf(x, *params[best_dist]),
    )
