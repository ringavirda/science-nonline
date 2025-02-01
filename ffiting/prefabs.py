""" Contains common data generation functions for model training use as well as most used generic models.
"""

from ffiting.framework.discretes import PolySpectrum
from .common import np, sp
from .framework import (
    Model,
    ModelLite,
    collapse,
    FittingOptions,
    FittingModes,
    apply_noise,
)


class Prefabs:
    """Contains common constants."""

    var_main = "x"
    quadratic_raw = "a0 + a1 * x + ... + an * x ^ n"
    exponential_raw = "a0 + a1 * x + a2 * exp(a3*x)"
    transcendental_raw = "a0 * sin(a1 * x) + a2 * cos(a1 * x)"
    combined_lint_raw = "a0 + a1 * x + a2 * sin(a3 * x) + a4 * cos(a3 * x)"


class Generators(Prefabs):
    """Container for generation objects."""

    @classmethod
    def ranked_poly(cls, coeffs: list[float], noise=True) -> ModelLite:
        poly = PolySpectrum(len(coeffs) - 1, cls.var_main)
        poly_m = poly.apply_trained(coeffs)
        return (
            poly_m
            if not noise
            else ModelLite(
                poly_m.expr_raw, poly_m.expr_sp, apply_noise(poly_m.model), coeffs
            )
        )

    @classmethod
    def exponential(
        cls, coeffs: tuple[float, float, float, float], noise=True
    ) -> ModelLite:
        def func(x: float) -> float:
            return coeffs[0] + coeffs[1] * x + coeffs[2] * np.exp(coeffs[3] * x)

        if noise:
            func = apply_noise(func)

        return ModelLite(
            cls.quadratic_raw,
            sp.parse_expr(cls.quadratic_raw),
            collapse(func),
            coeffs,
        )

    @classmethod
    def transcendental(
        cls, coeffs: tuple[float, float, float], noise=True
    ) -> ModelLite:
        def func(x: float) -> float:
            return coeffs[0] * np.sin(coeffs[1] * x) + coeffs[2] * np.cos(coeffs[1] * x)

        if noise:
            func = apply_noise(func)

        return ModelLite(
            cls.transcendental_raw,
            sp.parse_expr(cls.transcendental_raw),
            collapse(func),
            coeffs,
        )

    @classmethod
    def combined_lint(
        cls, coeffs: tuple[float, float, float, float, float], noise=True
    ) -> ModelLite:
        def func(x: float) -> float:
            return (
                coeffs[0]
                + coeffs[1] * x
                + coeffs[2] * np.sin(coeffs[3] * x)
                + coeffs[4] * np.cos(coeffs[3] * x)
            )

        if noise:
            func = apply_noise(func)

        return ModelLite(
            cls.combined_lint_raw,
            sp.parse_expr(cls.combined_lint_raw),
            collapse(func),
            coeffs,
        )


class Models(Prefabs):
    """Container for generic model objects that can be trained."""

    @classmethod
    def ranked_poly(cls, rank: int) -> Model:
        return PolySpectrum(rank, cls.var_main).as_model()

    @classmethod
    def exponential(cls, mode: FittingModes) -> Model:
        options = FittingOptions(
            fitting_mode=mode, expr_raw=cls.exponential_raw, rank=3
        )
        return Model(options)
    
    @classmethod
    def transcendental(cls, mode: FittingModes) -> Model:
        options = FittingOptions(
            fitting_mode=mode, expr_raw=cls.transcendental_raw, rank=2
        )
        return Model(options)
    
    @classmethod
    def combined_lint(cls, mode: FittingModes) -> Model:
        options = FittingOptions(
            fitting_mode=mode, expr_raw=cls.combined_lint_raw, rank=5
        )
        return Model(options)
