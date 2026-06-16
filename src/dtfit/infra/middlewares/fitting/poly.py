from dtfit.helpers import echo_if, collapse
from dtfit.infra import ModelData, Middleware
from dtfit.extra import poly_fit, filter_poly_coeffs


class FittingPolyMw(Middleware):
    """Middleware for processing data during the fitting process."""

    def __init__(
        self,
        method: str = "lite",
        degree: int = -1,
        filter: bool = False,
    ) -> None:
        """
        Initialize the FittingPolyMw middleware.

        Arguments:
            method (str): Fitting method to use, either 'lite' or 'lasso'.
            Defaults to 'lite'. degree (int): Degree of the polynomial to fit.
            If < 0, the degree will be determined automatically. Defaults to -1.
            filter (bool): Whether to apply a filter to the fitted polynomial
            coefficients when using the 'lite' method. Defaults to False.
        """
        super().__init__()
        self.method = method
        self.degree = degree
        self.filter = filter

    def process(self, data: ModelData) -> ModelData:
        if self.degree < 0:
            if data.degree is not None:
                self.degree = data.degree
            else:
                raise ValueError(
                    "Polynomial degree must be specified and non-negative."
                )

        fitted = poly_fit(
            data.data_x,
            data.data_y,
            self.degree,
            self.method,
            data.options,
        )

        if self.filter and self.method == "lite":
            fitted.coeffs = filter_poly_coeffs(
                fitted.coeffs,
                cutoff=data.options.degree_filter_cutoff,
            )
            echo_if(
                data.options,
                f"Filtered polynomial coefficients: {fitted.coeffs}",
            )

        data.degree = self.degree
        data.model = fitted.model
        data.coeffs = fitted.coeffs
        data.data_fitted = collapse(fitted.model)(data.data_x)

        return data
