"""Classes to do with differential spectra analysis."""

from dataclasses import dataclass
from typing import Self
from sympy.abc import w, k, H, n

from ..common import np, sp
from . import Model, ModelLite, collapse, FittingOptions, FittingModes


@dataclass
class Discretes:
    """Symbolic representations for numeric operations through differential
    discretes.
    """

    I = sp.Piecewise((w, sp.Eq(k, 0)), (0, True))
    C = sp.Piecewise(((H**k), sp.Eq(k, n)), (0, True))
    E = ((w * H) ** k) / sp.factorial(k)
    SIN = (((w * H) ** k) / sp.factorial(k)) * sp.sin((sp.pi * k) / 2)
    COS = (((w * H) ** k) / sp.factorial(k)) * sp.cos((sp.pi * k) / 2)


class Spectrum:
    """Base class for the differential spectra analysis. It allows to transform
    generic symbolic formula into a spectrum, that can be utilized through
    balancing against another one.
    """

    def __init__(self, expr: str, var_main: str) -> None:
        self.expr_sp: sp.Expr = sp.parse_expr(expr)
        self.expr_raw: str = expr
        self.expr_coeffs: list[sp.Symbol] = sorted(
            self.expr_sp.free_symbols, key=lambda s: s.name
        )
        self.var_main: sp.Symbol = sp.var(var_main)
        self.expr_coeffs.remove(self.var_main)
        self.expr_rank: int = len(self.expr_coeffs)

    def __reflect(self, sequence: sp.Expr, add: bool = True) -> sp.Expr:
        reflection = None
        for term in sequence.args:
            discrete = term
            if term.is_Symbol:
                if term == self.var_main:
                    discrete = Discretes.C.subs(((w, 1), (n, 1)))
                elif add:
                    discrete = Discretes.I.subs(w, term)
            elif term.is_Pow:
                discrete = Discretes.C.subs(n, term.args[1])
            elif term.is_Mul:
                discrete = self.__reflect(term, add=False)
            elif isinstance(term, sp.exp):
                discrete = (
                    Discretes.E.subs(w, term.args[0].args[0])
                    if term.args[0].is_Mul
                    else Discretes.E.subs(w, 1)
                )
            elif isinstance(term, sp.sin):
                discrete = (
                    Discretes.SIN.subs(w, term.args[0].args[0])
                    if term.args[0].is_Mul
                    else Discretes.SIN.subs(w, 1)
                )
            elif isinstance(term, sp.cos):
                discrete = (
                    Discretes.COS.subs(w, term.args[0].args[0])
                    if term.args[0].is_Mul
                    else Discretes.COS.subs(w, 1)
                )
            if not reflection:
                reflection = discrete
            elif add:
                reflection += discrete
            else:
                reflection *= discrete
        return reflection

    def ranked(self, rank: int = 0) -> list[sp.Expr]:
        """Convert the underlying expression into a differential spectrum of
        specific rank.

        Arguments:
            rank (int): manually sets the number of discretes in the spectrum. By
            default is set to "0" which means to use expression coefficient
            count as rank.

        Returns:
            list[sp.Expr]: A list of expressions that represent discretes in the spectrum.
        """
        reflection = self.__reflect(self.expr_sp)
        spectrum = []
        rank = len(self.expr_coeffs) if rank == 0 else rank
        for i in range(rank):
            spectrum.append(reflection.subs(k, i))
        return spectrum

    def apply_trained(self, coeffs: np.ndarray) -> ModelLite:
        """Using given fitted coefficients transform this spectrum into collapsable
        model object.

        Arguments:
            coeffs (np.ndarray): values that will be substituted into the underlying expression.

        Returns:
            ModelLite: An object of the given fit.
        """
        if len(coeffs) != len(self.expr_coeffs):
            raise RuntimeError("Invalid amount of trained coeffs.")
        model: sp.Expr = self.expr_sp
        for i, a in enumerate(self.expr_coeffs):
            model = model.subs(a, coeffs[i])
        func = sp.lambdify(self.var_main, model)
        return ModelLite(
            str(self.var_main), self.expr_raw, self.expr_sp, collapse(func), coeffs
        )

    def as_model(self, mode: FittingModes) -> Model:
        """Converts this spectrum into trainable model.

        Arguments:
            mode (FittingMode): Select which fitting mode to use for the model (type).

        Returns:
            Model: A trainable instance.
        """
        options = FittingOptions(
            mode,
            self.expr_raw,
            expr_sp=self.expr_sp,
            rank=self.expr_rank,
            var_main=self.var_main,
        )
        return Model(options)

    @staticmethod
    def from_model(model: Model | ModelLite) -> Self:
        """Creates differential spectrum representation of this model for symbolic operations."""
        return Spectrum(model.options.expr_raw, model.options.var_main)


class PolySpectrum(Spectrum):
    """Concrete spectrum implementation for polynomial expression that simplifies
    the process due to it being possible to construct valid expression directly
    from the rank value.
    """

    def __init__(self, rank: int, var_main: str) -> None:
        super().__init__(self.__construct_expr(rank, var_main), var_main)

    def __construct_expr(self, rank: int, main_var: str) -> str:
        terms: list[str] = ["c0"]
        for i in range(1, rank):
            terms.append(f"c{i}*{main_var}**{i}")
        return " + ".join(terms)

    def as_model(self, mode=FittingModes.POLY) -> Model:
        """Polynomial specific override that initializes return model as poly type."""
        options = FittingOptions(
            mode,
            self.expr_raw,
            expr_sp=self.expr_sp,
            rank=self.expr_rank,
            var_main=self.var_main,
        )
        return Model(options)
