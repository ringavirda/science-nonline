from dtfit.helpers import collapse
from dtfit.infra import ModelData, Middleware
from dtfit.extra import nonline_fit


class FittingNonlineMw(Middleware):
    """Middleware that fits a nonlinear-in-parameters model to the data.

    ``method="lsi"`` and ``method="eda"`` (numeric) fit the raw data directly.
    ``method="dsb"`` (symbolic reference) additionally requires polynomial
    coefficients, so a polynomial fit must precede it in the pipeline.
    Method-specific keyword arguments are forwarded to the underlying fitter.
    """

    def __init__(
        self,
        expr: str,
        var: str,
        method: str = "lsi",
        rank: int = -1,
        **method_kwargs,
    ) -> None:
        super().__init__()
        self.expr = expr
        self.var = var
        self.method = method
        self.rank = rank
        self.method_kwargs = method_kwargs

    def process(self, data: ModelData) -> ModelData:
        if (
            data.data_x is None
            or data.data_y is None
            or len(data.data_x) == 0
        ):
            raise ValueError(
                "Input data (data_x, data_y) must be available for nonline "
                "fitting."
            )

        if self.method == "dsb":
            if data.coeffs is None:
                raise ValueError(
                    "method 'dsb' needs polynomial coefficients; add a "
                    "polynomial fit before it in the pipeline."
                )
            result = nonline_fit(
                self.expr,
                self.var,
                self.method,
                data.options,
                coeffs_poly=data.coeffs,
                **self.method_kwargs,
            )
        else:
            result = nonline_fit(
                self.expr,
                self.var,
                self.method,
                data.options,
                data_x=data.data_x,
                data_y=data.data_y,
                **self.method_kwargs,
            )

        data.coeffs = result.coeffs
        data.model = result.model
        data.nonline_expr = self.expr
        data.nonline_var = self.var
        data.data_fitted = collapse(result.model)(data.data_x)

        return data
