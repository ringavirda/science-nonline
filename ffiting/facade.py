""" FFitting interface for external access. Populated with functions that "simplify"
underlying framework infrastructure. Limited functionality.
"""

from .common import np
from . import ModelLite, Model, FittingModes, FittingOptions, Models


def poly_fit(data: np.ndarray, rank: int) -> ModelLite:
    """Wrapper for the default `poly_fit` function from `numpy`, that returns
    common for `ffitting` model type.

    Arguments:
        data (np.ndarray): raw data used to train the LSM model.
        rank (int): the amount of terms/coeffs the polynomial should have.

    Returns:
        ModelLite: An instance with fitted data.
    """
    return Models.ranked_poly(rank).fit(data)


def nonline_fit(data: np.ndarray, expr: str, var: str) -> ModelLite:
    """Primary fitting method that utilizes developed experimental approaches."""
    options = FittingOptions(expr_raw=expr, var=var, fitting_mode=FittingModes.DSB)
    return Model(options).fit(data)
