from dtfit.helpers import FittingResult, FittingOptions
from .dt import fit_dsb, fit_lsi, fit_eda


def nonline_fit(
    expr: str,
    var: str,
    method: str = "lsi",
    options: FittingOptions = FittingOptions(),
    *,
    coeffs_poly=None,
    data_x=None,
    data_y=None,
    **kwargs,
) -> FittingResult:
    """Fit a model that is nonlinear in its parameters.

    Methods:
        ``"dsb"``: differential spectra balance (symbolic, analytical
            reference). Requires ``coeffs_poly`` from a prior polynomial fit.
        ``"lsi"``: least-squares integral (numeric). Fits raw ``data_x``/
            ``data_y`` directly.
        ``"eda"``: equal differential areas / equal areas (numeric). Fits raw
            ``data_x``/``data_y`` directly.

    Args:
        expr: The nonlinear model expression, as a string.
        var: The main variable used in the expression.
        method: One of ``"dsb"``, ``"lsi"``, ``"eda"``.
        options: Options for the fitting process (e.g. echo logging).
        coeffs_poly: Polynomial coefficients (required for ``"dsb"``).
        data_x, data_y: Observed samples (required for ``"lsi"``/``"eda"``).
        **kwargs: Method-specific keyword arguments forwarded to the fitter.

    Returns:
        FittingResult: The fitted model and its coefficients.
    """
    if method == "dsb":
        if coeffs_poly is None:
            raise RuntimeError(
                "method 'dsb' requires coeffs_poly from a polynomial fit."
            )
        return fit_dsb(coeffs_poly, expr, var, options, **kwargs)

    if method in ("lsi", "eda"):
        if data_x is None or data_y is None:
            raise RuntimeError(
                f"method '{method}' requires data_x and data_y."
            )
        fitter = fit_lsi if method == "lsi" else fit_eda
        return fitter(data_x, data_y, expr, var, options, **kwargs)

    raise RuntimeError(f"Unrecognized fitting method: {method!r}")
