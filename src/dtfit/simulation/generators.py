import numpy as np
from typing import Callable

Generator = Callable[[float], float]


def gen_ranked_poly(coeffs: list[float]) -> Callable[[float], float]:
    """Creates dataset generator using polynomial as base. Can be configured
    to the necessary poly rank or be polluted with noise.

    Formula:
        g(x, n) = sum(0, n, (i) => coeffs[i] * x^i)

    Args:
        x (float): Input value for the polynomial function.
        coeffs (list[float]): List of poly coefficients, determines poly rank.

    Returns:
        float: Result of the polynomial function.
    """

    def poly(x: float) -> float:
        result = 0.0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x**i)
        return result

    return poly


def gen_exponential4(
    coeffs: list[float],
) -> Callable[[float], float]:
    """Creates exponential dataset generator. Has only 4 coefficients, can be
    polluted.

    Formula:
        g(x) = coeffs[0] + coeffs[1] * x + coeffs[2] * exp(coeffs[3] * x)

    Args:
        x (float): Input value for the exponential function.
        coeffs (list[float]): Collection of 4 model coefficients.
    Returns:
        float: Result of the exponential function.
    """

    if len(coeffs) != 4:
        raise ValueError("Expected 4 coefficients for exponential generator.")

    def func(x: float) -> float:
        return coeffs[0] + coeffs[1] * x + coeffs[2] * np.exp(coeffs[3] * x)

    return func


def gen_transcendental(coeffs: list[float]) -> Callable[[float], float]:
    """Creates a dataset generator with transcendental rule. Has only 3 model
    coefficients.

    Formula:
        g(x) = coeffs[0] * cos(coeffs[1] * x) + coeffs[2] * sin(coeffs[1] * x)

    Args:
        x (float): Input value for the transcendental function.
        coeffs (list[float]): Collection of 3 model coefficients.

    Returns:
        float: Result of the transcendental function.
    """

    if len(coeffs) != 3:
        raise ValueError("Expected 3 coefficients for transcendental generator.")

    def func(x: float) -> float:
        return coeffs[0] * np.sin(coeffs[1] * x) + coeffs[2] * np.cos(coeffs[1] * x)

    return func


def gen_combined_lint(
    coeffs: list[float],
) -> Callable[[float], float]:
    """Uses combined linear and transcendental rule to generate complex
    harmonic data. Parametrized with 5 arguments.

    Formula:
        g(x) = coeffs[0] + coeffs[1] * x + coeffs[2] * cos(coeffs[3] * x) +
        coeffs[4] * sin(coeffs[3] * x)

    Args:
        x (float): Input value for the combined linear and transcendental function.
        coeffs (list[float]): Collection with 5 known model coefficients.

    Returns:
        float: Result of the combined linear and transcendental function.
    """
    if len(coeffs) != 5:
        raise ValueError(
            "Expected 5 coefficients for combined linear and transcendental generator."
        )

    def func(x: float) -> float:
        return (
            coeffs[0]
            + coeffs[1] * x
            + coeffs[2] * np.sin(coeffs[3] * x)
            + coeffs[4] * np.cos(coeffs[3] * x)
        )

    return func
