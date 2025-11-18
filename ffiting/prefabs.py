"""Contains common data generation functions for model training use as well
as most used generic models.
"""

from ffiting.framework.discretes import PolySpectrum
from ffiting.framework.utils import NoiseConfig
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
    exponential_raw = "a0 + a1*x + a2*exp(a3*x)"
    transcendental_raw = "a0*cos(a1*x) + a2*sin(a1*x)"
    combined_lint_raw = "a0 + a1*x + a2*sin(a3*x) + a4*cos(a3*x)"


class Generators(Prefabs):
    """Container for generation objects."""

    @classmethod
    def ranked_poly(
        cls, coeffs: list[float], noise=False, noise_config=NoiseConfig()
    ) -> ModelLite:
        """Creates dataset generator using polynomial as base. Can be configured
        to the necessary poly rank or be polluted with noise.

        Formula:
            z(x, n) = sum(0, n, (i) => coeffs[i] * x^i)

        Args:
            coeffs (list[float]): List of poly coefficients, determines poly rank.
            noise (bool, optional): Sets default pollution for the generator.
            Defaults to False.
            noise_config (NoiseConfig, optional): Additional config data for the
            noise generator. Defaults to NoiseConfig().

        Returns:
            ModelLite: Model that can be used as data generated. Parametrized with
            given coeffs.
        """
        poly = PolySpectrum(len(coeffs), cls.var_main)
        poly_m = ModelLite(
            poly.var_main, poly.expr_raw, poly.expr_sp, np.poly1d(coeffs[::-1]), coeffs
        )
        return (
            poly_m
            if not noise
            else ModelLite(
                poly_m.var_main,
                poly_m.expr_raw,
                poly_m.expr_sp,
                lambda data: apply_noise(poly_m.model(data), noise_config),
                coeffs,
            )
        )

    @classmethod
    def exponential(
        cls,
        coeffs: tuple[float, float, float, float],
        noise=False,
        noise_config=NoiseConfig(),
    ) -> ModelLite:
        """Creates exponential dataset generator. Has only 4 coefficients, can be
        polluted.

        Formula:
            z(x) = coeffs[0] + coeffs[1] * x + coeff[2] * exp(coeff[3] * x)

        Args:
            coeffs (tuple[float, float, float, float]): Collection of 4 model
            coefficients.
            noise (bool, optional): Sets default pollution for the generator.
            Defaults to False.
            noise_config (NoiseConfig, optional): Additional config data for the
            noise generator. Defaults to NoiseConfig().

        Returns:
            ModelLite: Model that can be used as data generated. Parametrized with
            given coeffs.
        """

        def func(x: float) -> float:
            return coeffs[0] + coeffs[1] * x + coeffs[2] * np.exp(coeffs[3] * x)

        return ModelLite(
            cls.var_main,
            cls.exponential_raw,
            sp.parse_expr(cls.exponential_raw),
            lambda data: (
                apply_noise(collapse(func)(data), noise_config)
                if noise
                else collapse(func)(data)
            ),
            coeffs,
        )

    @classmethod
    def transcendental(
        cls, coeffs: tuple[float, float, float], noise=False, noise_config=NoiseConfig()
    ) -> ModelLite:
        """Creates a dataset generator with transcendental rule. Has only 3 model
        coefficients and can be polluted by default.

        Formula:
            z(x) = coeffs[0] * cos(coeffs[1] * x) + coeffs[2] * sin(coeffs[1] * x)

        Args:
            coeffs (tuple[float, float, float]): Collection of 3 model coefficients.
            noise (bool, optional): Sets default pollution for the generator.
            Defaults to False.
            noise_config (NoiseConfig, optional): Additional config data for the
            noise generator. Defaults to NoiseConfig().

        Returns:
            ModelLite: Model that can be used as data generated. Parametrized with
            given coeffs.
        """

        def func(x: float) -> float:
            return coeffs[0] * np.sin(coeffs[1] * x) + coeffs[2] * np.cos(coeffs[1] * x)

        return ModelLite(
            cls.var_main,
            cls.transcendental_raw,
            sp.parse_expr(cls.transcendental_raw),
            lambda data: (
                apply_noise(collapse(func)(data), noise_config)
                if noise
                else collapse(func)(data)
            ),
            coeffs,
        )

    @classmethod
    def combined_lint(
        cls,
        coeffs: tuple[float, float, float, float, float],
        noise=False,
        noise_config=NoiseConfig(),
    ) -> ModelLite:
        """Uses combined linear and transcendental rule to generate complex
        harmonic data. Parametrized with 5 arguments and can be polluted.

        Formula:
            z(x) = coeffs[0] + coeffs[1] * x + coeffs[2] * cos(coeffs[3] * x) +
            coeffs[4] * sin(coeffs[3] * x)

        Args:
            coeffs (tuple[float, float, float, float, float]): Collection with 5
            known model coefficients.
            noise (bool, optional): Sets default pollution for the generator.
            Defaults to False.
            noise_config (NoiseConfig, optional): Additional config data for the
            noise generator. Defaults to NoiseConfig().

        Returns:
            ModelLite: Model that can be used as data generated. Parametrized with
            given coeffs.
        """

        def func(x: float) -> float:
            return (
                coeffs[0]
                + coeffs[1] * x
                + coeffs[2] * np.sin(coeffs[3] * x)
                + coeffs[4] * np.cos(coeffs[3] * x)
            )

        return ModelLite(
            cls.var_main,
            cls.combined_lint_raw,
            sp.parse_expr(cls.combined_lint_raw),
            lambda data: (
                apply_noise(collapse(func)(data), noise_config)
                if noise
                else collapse(func)(data)
            ),
            coeffs,
        )


class Models(Prefabs):
    """Container for generic model objects that can be trained."""

    @classmethod
    def ranked_poly(cls, rank: int) -> Model:
        """Creates trainable model object for a polynomial with specific rank.

        Formula:
            f(x, n) = sum(0, n, (i) => a_i * x^i)

        Args:
            rank (int): Sets the rank for the created model.

        Returns:
            Model: Trainable object for use with other library infrastructure.
        """
        return PolySpectrum(rank, cls.var_main, coeff_sig="a").as_model()

    @classmethod
    def exponential(cls, mode: FittingModes) -> Model:
        """Returns an exponential object for future training.

        Formula:
            f(x) = a0 + a1*x + a2*exp(a3*x)

        Args:
            mode (FittingModes): Selects one of the available fitting methods for
            created model.

        Returns:
            Model: Trainable object for use with other library infrastructure.
        """
        options = FittingOptions(
            fitting_mode=mode, expr_raw=cls.exponential_raw, rank=4
        )
        return Model(options)

    @classmethod
    def transcendental(cls, mode: FittingModes) -> Model:
        """Generates new trainable object of transcendental model.

        Formula:
            f(x) = a0*cos(a1*x) + a2*sin(a1*x)

        Args:
            mode (FittingModes): Selects one of the available fitting methods for
            created model.

        Returns:
            Model: Trainable object for use with other library infrastructure.
        """
        options = FittingOptions(
            fitting_mode=mode, expr_raw=cls.transcendental_raw, rank=3
        )
        return Model(options)

    @classmethod
    def combined_lint(cls, mode: FittingModes) -> Model:
        """Creates new model object with combined linear and transcendental rule
        for future training.

        Formula:
            f(x) = a0 + a1*x + a2*cos(a3*x) + a4*sin(a3*x)

        Args:
            mode (FittingModes): Selects one of the available fitting methods for
            created model.

        Returns:
            Model: Trainable object for use with other library infrastructure.
        """
        options = FittingOptions(
            fitting_mode=mode, expr_raw=cls.combined_lint_raw, rank=5
        )
        return Model(options)
