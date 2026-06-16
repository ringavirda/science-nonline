import numpy as np
from dtfit.helpers import echo_if
from dtfit.infra import ModelData, Middleware


class DataInsertionMw(Middleware):
    """Middleware for processing data during data preparation."""

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y

    def process(self, data: ModelData) -> ModelData:
        data.data_x = self.data_x
        data.data_y = self.data_y

        echo_if(
            data.options,
            f"Inserted provided data into data_x and data_y arrays.",
        )
        return data
