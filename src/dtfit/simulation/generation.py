import numpy as np
from typing import Callable
from dtfit.helpers import echo_if
from dtfit.infra import ModelData, Middleware


class DataGenerationMw(Middleware):
    """Middleware for processing data during data preparation."""

    def __init__(self, size: int, generator: Callable[[float], float]) -> None:
        super().__init__()
        self.size = size
        self.generator = generator

    def process(self, data: ModelData) -> ModelData:
        data.data_x = np.arange(self.size)
        data.data_y = np.array([self.generator(x) for x in data.data_x])

        echo_if(
            data.options,
            f"Generated data with {self.size} points. Populated data_x and data_y arrays.",
        )

        return data
