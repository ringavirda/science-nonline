
from dtfit.infra import ModelData, Middleware
from dtfit.extra import find_degree


class FindPolyDegreeMw(Middleware):
    """Middleware for finding the optimal polynomial degree for fitting."""

    def __init__(
        self,
        method: str = "crossval",
        min_degree: int = 1,
        max_degree: int = 12,
    ) -> None:
        """
        Initialize the FindPolyDegreeMw middleware.

        Arguments:
            method (str): Method for determining polynomial degree. Options are
            'bic' for direct Bayesian information criteria (BIC), 'aic' for
            direct Akaike information criteria (AIC), and 'crossval' for
            cross-validation. Defaults to 'crossval'. Note that 'crossval' may
            be more computationally intensive than   the direct methods, but can
            provide a more robust degree selection in some cases.
            min_degree (int): Minimum degree of the polynomial to consider.
            Defaults to 1.
            max_degree (int): Maximum degree of the polynomial to consider.
            Defaults to 12.
        """
        super().__init__()
        self.method = method
        self.min_degree = min_degree
        self.max_degree = max_degree

    def process(self, data: ModelData) -> ModelData:

        data.degree = find_degree(
            data.data_x,
            data.data_y,
            min_degree=self.min_degree,
            max_degree=self.max_degree,
            method=self.method,
            options=data.options,
        )

        return data
