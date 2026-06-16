from typing import Optional
from dataclasses import dataclass
from dtfit.helpers import echo_if
from dtfit.infra import ModelData, Middleware
from .noise import (
    apply_normal_noise,
    apply_uniform_noise,
    apply_abnormal_noise,
    apply_uniform_abnormal_noise,
)


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""

    mean: float = 0.0
    std: float = 1.0
    distribution: str = "normal"  # 'normal', 'uniform', etc.
    seed: Optional[int] = None  # For reproducibility


@dataclass
class AbnormalsConfig:
    """Configuration for abnormal data generation."""

    fraction: float = 0.1  # Fraction of data points to be made abnormal
    magnitude: float = 10.0  # Magnitude of abnormalities
    distribution: str = "uniform"  # 'normal', 'uniform', etc.
    seed: Optional[int] = None  # For reproducibility


class DataPollutionMw(Middleware):
    """Middleware for processing data during data preparation."""

    def __init__(
        self,
        noise_config: NoiseConfig | None = NoiseConfig(),
        abnormals_config: AbnormalsConfig | None = AbnormalsConfig(),
    ) -> None:
        super().__init__()
        self.noise_config = noise_config
        self.abnormals_config = abnormals_config

    def process(self, data: ModelData) -> ModelData:
        if data.data_y is None:
            raise ValueError("Cannot apply noise because no data available.")

        if self.noise_config is not None:
            if self.noise_config.distribution == "normal":
                data.data_y = apply_normal_noise(
                    data.data_y,
                    self.noise_config.mean,
                    self.noise_config.std,
                    self.noise_config.seed,
                )
            elif self.noise_config.distribution == "uniform":
                data.data_y = apply_uniform_noise(
                    data.data_y,
                    -self.noise_config.std,
                    self.noise_config.std,
                    self.noise_config.seed,
                )
            else:
                raise ValueError(
                    f"Unsupported noise distribution: {self.noise_config.distribution}"
                )

            echo_if(
                data.options,
                f"Applied noise to data_y with characteristics:\n"
                + f"noise distribution '{self.noise_config.distribution}', mean={self.noise_config.mean}, std={self.noise_config.std}.",
            )

            if self.abnormals_config:
                if self.abnormals_config.distribution == "uniform":
                    data.data_y = apply_uniform_abnormal_noise(
                        data.data_y,
                        -self.noise_config.std,
                        self.noise_config.std,
                        self.abnormals_config.magnitude,
                        self.abnormals_config.fraction,
                        self.abnormals_config.seed,
                    )
                elif self.abnormals_config.distribution == "normal":
                    data.data_y = apply_abnormal_noise(
                        data.data_y,
                        self.noise_config.mean,
                        self.noise_config.std,
                        self.abnormals_config.magnitude,
                        self.abnormals_config.fraction,
                        self.abnormals_config.seed,
                    )
                else:
                    raise ValueError(
                        f"Unsupported abnormalities distribution: {self.abnormals_config.distribution}"
                    )
                echo_if(
                    data.options,
                    f"Applied abnormalities to data_y with characteristics:\n"
                    + f"abnormalities distribution '{self.abnormals_config.distribution}' and magnitude={self.abnormals_config.magnitude}, fraction={self.abnormals_config.fraction}.",
                )
            else:
                echo_if(
                    data.options,
                    "No abnormalities configuration provided, skipping abnormalities application.",
                )
        else:
            echo_if(
                data.options,
                "No noise configuration provided, skipping noise application.",
            )

        return data
