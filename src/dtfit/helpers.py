import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.interpolate import interp1d

from dtfit.log import logger


@dataclass
class FittingResult:
    """Class to represent the result of fitting process. Contains all necessary information about the result."""

    model: Callable[[float], float]
    coeffs: np.ndarray
    cov: np.ndarray | None = None
    """Parameter covariance estimate (``n_params x n_params``) when the method
    can produce one from an overdetermined system; ``None`` otherwise. The
    square roots of its diagonal are the per-parameter standard errors."""


@dataclass
class FittingOptions:
    """Class to represent the options for fitting process. Contains all necessary information about the options."""

    echo: bool = field(default=False)
    echo_method: Callable = field(default=print)
    degree_filter: bool = field(default=False)
    degree_filter_cutoff: float = field(default=3.0)
    lasso_alpha: float = field(default=0.1)
    lasso_max_iter: int = field(default=100000)
    crossval_cv_folds: int = field(default=5)

    def update(self, options: dict[str, Any]) -> None:
        """Update the options with the given dictionary."""
        for key, value in options.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        """Return the options as a dictionary."""
        return {
            "echo": self.echo,
            "echo_method": self.echo_method,
            "degree_filter": self.degree_filter,
            "degree_filter_cutoff": self.degree_filter_cutoff,
            "lasso_alpha": self.lasso_alpha,
            "lasso_max_iter": self.lasso_max_iter,
            "crossval_cv_folds": self.crossval_cv_folds,
        }


def echo_if(
    options: FittingOptions,
    message: str,
    value: list[Any] | Any | None = None,
) -> None:
    """Emit a fitting message through the dtfit logger.

    Messages are logged at INFO level when ``options.echo`` is set, otherwise at
    DEBUG level. Output is silent unless the application configures logging (see
    :func:`dtfit.enable_logging`).

    Args:
        options (FittingOptions): Determines the log level (echo -> INFO).
        message (str): Message to log before any values.
        value (list[Any] | Any | None, optional): Value(s) to log after the
        message. Defaults to None.
    """
    level = logging.INFO if options.echo else logging.DEBUG
    if not logger.isEnabledFor(level):
        return

    logger.log(level, message)
    if value is not None:
        values = value if isinstance(value, list) else [value]
        for val in values:
            logger.log(level, "%s", val)


def collapse(
    func: Callable[[float], float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrapper for generation or modelling functions that cannot automatically parse arrays."""

    def inner(value: np.ndarray) -> np.ndarray:
        """Applies given function to each element of the array and returns new array."""
        res = np.empty(value.shape)
        for i, v in np.ndenumerate(value):
            res[i] = func(v)
        return res

    return inner


def scale_data(data: np.ndarray, coeff: float) -> np.ndarray:
    """Generates new dataset from the given one through resizing it. Implements
    interpolation to infer new points in the data, so it may have some minor error
    init. It does not extend the radius of given data, simply increases or decreases
    the amount of points in the vector.

    Arguments:
        data (ndarray): Vector to be scaled.
        coeff (float): A value to represent by how much the data needs to be resized.
        If it is set to 2 than vector with double the size returns, if 0.5 - half,
        if 1 - no changes.

    Returns:
        ndarray: New generated scaled vector from input data.
    """
    range_new = np.arange(0, data.size, 1 / coeff)
    inter = interp1d(np.arange(data.size), data)
    return inter(range_new)
