import numpy as np
from dtfit.infra import ModelData, Middleware


class DataNormalizationMw(Middleware):
    """Middleware for processing data during data preparation."""

    def __init__(self, type: str) -> None:
        super().__init__()
        self.type = type

    def process(self, data: ModelData) -> ModelData:
        if self.type == "min-max":
            data.data_x = (data.data_x - np.min(data.data_x)) / (
                np.max(data.data_x) - np.min(data.data_x)
            )
            data.data_y = (data.data_y - np.min(data.data_y)) / (
                np.max(data.data_y) - np.min(data.data_y)
            )
        elif self.type == "z-score":
            data.data_x = (data.data_x - np.mean(data.data_x)) / np.std(
                data.data_x
            )
            data.data_y = (data.data_y - np.mean(data.data_y)) / np.std(
                data.data_y
            )
        return data
