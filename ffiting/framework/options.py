"""Objects needed to configure fitting functions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from IPython.display import display

from ..common import sp


class FittingModes(Enum):
    """Contains available modes for fitting, can be specified through
    `FitOptions`.
    """

    AUTO = 0  # System will try to guess fitting method.
    POLY = 1  # Use well known Least Squares Method for polynomial fitting.
    DSB = 2  # Nonlinear fitting method with Differential Spectra Balance.
    DSBI = 3  # Nonlinear fitting method with DSB in Integral form.


@dataclass
class FittingOptions:
    """Object used to configure fitting methods, can change a lot in terms of
    execution.
    """

    # Choose one of existing fitting methods. Default mode is `AUTO`.
    fitting_mode: FittingModes
    # Raw string representation of nonline mathematical model to use. Required
    # for the nonlinear methods.
    expr_raw: str
    # Free variable to use in expression forms. Default is "x".
    var_main: str = field(default="x")
    # If set to `False` doesn't use the fitting results for the internal structure.
    # Default is `True`.
    update_model: bool = field(default=True)
    # Return result as a Model instance, instead of ModelLite. By default is
    # set to `False`.
    model_full: bool = field(default=False)
    # Perform additional numeric fitting if possible, may cause additional overhead.
    # Default value is `False`.
    numeric_optimize: bool = field(default=False)
    # Turns on the display for steps of the fitting.
    echo_on: bool = field(default=False)
    # Specifies the exact method to use for displaying echo values. Uses IPython's
    # `display` by default.
    echo_method: Callable[[Any | None], None] = field(default=display)
    # Try to increase underlying polynomial rank to extend the
    # flexibility of the DSB approach to larger data sets. It is set
    # to `False` by default.
    raise_rank: bool = field(default=False)
    # Override generated symbolic expression using this.
    expr_sp: Optional[sp.Expr] = field(default=None)
    # Used wherever the "ranking" is needed. Default is "0" usually standing for
    # "figure out automatically".
    rank: Optional[int] = field(default=None)
    # To substitute H value in integral methods.
    dsbi_h: float = field(default=1.0)
