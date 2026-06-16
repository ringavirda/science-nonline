"""Generic differential-transform helpers built on SymPy Taylor expansion.

The differential transform of ``f`` about ``t0=0`` with sampling interval ``H``
is ``F(k) = (H**k / k!) * f^(k)(0)``. In a *spectra balance* every equation
sets a model discrete equal to the corresponding data discrete at the same
order ``k``, so the common ``H**k`` factor cancels on both sides and the balance
reduces to matching plain Taylor (Maclaurin) coefficients ``f^(k)(0)/k!``.

That observation removes any need for hand-written per-function discrete rules
(``exp``, ``sin``, ``cos``, monomials, ...): the Taylor coefficient of *any*
expression SymPy can differentiate is available directly. These helpers provide
that coefficient sequence and the parameter list for a model expression.
"""

import sympy as sp


def model_params(f_sym: sp.Expr, t: sp.Symbol) -> list[sp.Symbol]:
    """Return the free parameters of ``f_sym`` (all symbols except ``t``),
    ordered by name for a stable coefficient layout."""
    return sorted(
        (s for s in f_sym.free_symbols if s != t), key=lambda s: s.name
    )


def taylor_coeffs(
    f_sym: sp.Expr, t: sp.Symbol, order: int
) -> list[sp.Expr]:
    """Symbolic Maclaurin coefficients ``a_k = f^(k)(0) / k!`` for
    ``k = 0 .. order`` (inclusive).

    This is the ``H``-free differential spectrum of ``f_sym``: it equals the
    classical differential-transform discrete ``F(k)`` up to the ``H**k`` factor
    that cancels in a spectra balance. Works for any differentiable expression,
    unlike the previous table of closed-form discretes.
    """
    coeffs: list[sp.Expr] = []
    deriv = f_sym
    for k in range(order + 1):
        coeffs.append(sp.simplify(deriv.subs(t, 0) / sp.factorial(k)))
        if k < order:
            deriv = sp.diff(deriv, t)
    return coeffs
