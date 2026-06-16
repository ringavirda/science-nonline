import numpy as np
from typing import Optional


def apply_normal_noise(
    data: np.ndarray, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Applies normal noise to the input data."""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(mean, std, data.shape)
    return data + noise


def apply_uniform_noise(
    data: np.ndarray, low: float = -1.0, high: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Applies uniform noise to the input data."""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.uniform(low, high, data.shape)
    return data + noise


def apply_abnormal_noise(
    data: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
    coeff: float = 3.0,
    density: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Applies abnormal noise to the input data."""
    if seed is not None:
        np.random.seed(seed)
    abnormal_count = int((data.size * density) / 100)
    abnormal_indices = np.random.choice(data.size, size=abnormal_count, replace=False)
    abnormal_noise = np.random.normal(mean, std * coeff, size=abnormal_count)
    noisy_data = data.copy()
    noisy_data[abnormal_indices] += abnormal_noise
    return noisy_data


def apply_uniform_abnormal_noise(
    data: np.ndarray,
    low: float = -1.0,
    high: float = 1.0,
    coeff: float = 3.0,
    density: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Applies uniform abnormal noise to the input data."""
    if seed is not None:
        np.random.seed(seed)
    abnormal_count = int((data.size * density) / 100)
    abnormal_indices = np.random.choice(data.size, size=abnormal_count, replace=False)
    abnormal_noise = np.random.uniform(low * coeff, high * coeff, size=abnormal_count)
    noisy_data = data.copy()
    noisy_data[abnormal_indices] += abnormal_noise
    return noisy_data
