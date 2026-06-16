class Prefabs:
    """Contains common constants."""

    var_main = "x"
    exponential_raw = "a0 + a1*x + a2*exp(a3*x)"
    transcendental_raw = "a0*cos(a1*x) + a2*sin(a1*x)"
    combined_lint_raw = "a0 + a1*x + a2*sin(a3*x) + a4*cos(a3*x)"


# class Models(Prefabs):
#     """Container for generic model objects that can be trained."""

#     @classmethod
#     def ranked_poly(cls, rank: int) -> Model:
#         """Creates trainable model object for a polynomial with specific rank.

#         Formula:
#             f(x, n) = sum(0, n, (i) => a_i * x^i)

#         Args:
#             rank (int): Sets the rank for the created model.

#         Returns:
#             Model: Trainable object for use with other library infrastructure.
#         """
#         return PolySpectrum(rank, cls.var_main, coeff_sig="a").as_model()

#     @classmethod
#     def exponential(cls, mode: FittingModes) -> Model:
#         """Returns an exponential object for future training.

#         Formula:
#             f(x) = a0 + a1*x + a2*exp(a3*x)

#         Args:
#             mode (FittingModes): Selects one of the available fitting methods for
#             created model.

#         Returns:
#             Model: Trainable object for use with other library infrastructure.
#         """
#         options = FittingOptions(
#             fitting_mode=mode, expr_raw=cls.exponential_raw, rank=4
#         )
#         return Model(options)

#     @classmethod
#     def transcendental(cls, mode: FittingModes) -> Model:
#         """Generates new trainable object of transcendental model.

#         Formula:
#             f(x) = a0*cos(a1*x) + a2*sin(a1*x)

#         Args:
#             mode (FittingModes): Selects one of the available fitting methods for
#             created model.

#         Returns:
#             Model: Trainable object for use with other library infrastructure.
#         """
#         options = FittingOptions(
#             fitting_mode=mode, expr_raw=cls.transcendental_raw, rank=3
#         )
#         return Model(options)

#     @classmethod
#     def combined_lint(cls, mode: FittingModes) -> Model:
#         """Creates new model object with combined linear and transcendental rule
#         for future training.

#         Formula:
#             f(x) = a0 + a1*x + a2*cos(a3*x) + a4*sin(a3*x)

#         Args:
#             mode (FittingModes): Selects one of the available fitting methods for
#             created model.

#         Returns:
#             Model: Trainable object for use with other library infrastructure.
#         """
#         options = FittingOptions(
#             fitting_mode=mode, expr_raw=cls.combined_lint_raw, rank=5
#         )
#         return Model(options)

#     @classmethod
#     def custom(cls, expr_raw: str, mode: FittingModes) -> Model:
#         """Creates new model object with custom user-defined expression for future training.

#         Args:
#             expr_raw (str): The raw expression string defining the model.
#             mode (FittingModes): Selects one of the available fitting methods for
#             created model.
#         Returns:
#             Model: Trainable object for use with other library infrastructure.
#         """
#         options = FittingOptions(fitting_mode=mode, expr_raw=expr_raw)
#         return Model(options)
