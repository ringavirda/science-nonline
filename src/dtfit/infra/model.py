"""
Base model class and data container for the fitting framework.
This is a common module that cannot use `common` imports to avoid circular
dependencies.
"""

import numpy as np
from dtfit.helpers import FittingOptions, collapse
from .model_data import ModelData
from .middleware import Middleware, MiddlewareStack


class Model:
    """Base class for different fitting modes."""

    def __init__(
        self,
        data: ModelData | None = None,
        options: FittingOptions | None = None,
    ) -> None:
        self.__stack = MiddlewareStack()
        self.__data = data or ModelData(options=options or FittingOptions())
        if options is not None:
            self.__data.options.update(options.as_dict())

    @property
    def data(self) -> ModelData:
        """Get the current model data."""
        return self.__data

    @property
    def stack(self) -> MiddlewareStack:
        """Get the middleware stack for this model."""
        return self.__stack

    def fit(
        self, data_x: np.ndarray | None = None, data_y: np.ndarray | None = None
    ) -> ModelData:
        """Execute the fitting process."""
        if data_x is not None:
            self.__data.data_x = data_x
        if data_y is not None:
            self.__data.data_y = data_y
        self.__data = self.__stack.exec(self.__data)
        return self.__data

    def predict(self, data_x: np.ndarray) -> np.ndarray:
        """Predict values based on the fitted model."""
        if self.data.model is None:
            raise RuntimeError("Model is not fitted yet.")

        return collapse(self.data.model)(data_x)

    def use(self, *middlewares: Middleware, clear: bool = False) -> None:
        """Add middleware components to the model's processing stack."""
        if clear:
            self.__stack.clear()
        for middleware in middlewares:
            self.__stack.add(middleware)
