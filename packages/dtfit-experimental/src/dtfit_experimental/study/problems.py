"""The evolution matrix's own problem set: eight symbolic model families
with their truths, and the two streaming scenarios notebook 17 measures.

Separate from :mod:`dtfit_experimental.study.families`, the sixteen
parameter-estimation families used by the estimation study: those move on
their own and this catalogue stays apart so the stored evolution matrix
does not shift under them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sp

#: Family name -> (SymPy expression string in ``t``, truth value by
#: parameter name, domain half-width ``H`` so a grid spans ``[0, H]``).
PROBLEMS: dict[str, tuple[str, dict[str, float], float]] = {
    "exp_decay": ("a*exp(b*t)", {"a": 3.0, "b": -1.2}, 2.0),
    "logistic": (
        "L/(1 + exp(-k*(t - c)))", {"L": 5.0, "c": 3.0, "k": 1.4}, 8.0),
    "gauss_peak": (
        "a*exp(-(t - m)**2/(2*s**2))", {"a": 2.0, "m": 2.5, "s": 0.6}, 5.0),
    "damped_sine": (
        "a*exp(-d*t)*sin(w*t)", {"a": 2.0, "d": 0.3, "w": 3.0}, 6.0),
    "arctan": ("a*atan(w*t)", {"a": 1.0, "w": 2.0}, 3.0),
    "michaelis_menten": (
        "Vm*t/(Km + t)", {"Km": 0.8, "Vm": 1.2}, 8.0),
    "power_law": ("a*t**b", {"a": 2.0, "b": 0.7}, 5.0),
    "two_exp": (
        "a*exp(b*t) + c*exp(d*t)",
        {"a": 2.0, "b": -0.5, "c": 1.0, "d": -3.0}, 6.0),
}


def truth_vector(
    expr: str, truth: dict[str, float]
) -> tuple[list[str], np.ndarray]:
    """Parameter names and truth values in the model's canonical order.

    Args:
        expr: SymPy expression string in ``t``.
        truth: Truth value for every free parameter of ``expr``.

    Returns:
        ``(names, values)``: the free parameters of ``expr`` sorted
        alphabetically, the order every stage adapter and ``dtfit.fit`` on
        a string model uses, and the matching truth values.

    Raises:
        KeyError: ``truth`` has no entry for one of ``expr``'s free
            symbols.
        sympy.SympifyError: ``expr`` does not parse as a SymPy
            expression.
    """
    t = sp.Symbol("t")
    names = sorted(str(s) for s in sp.sympify(expr).free_symbols - {t})
    return names, np.array([truth[n] for n in names])


def model(expr: str, names: list[str]) -> Callable[..., np.ndarray]:
    """A broadcast-safe callable model lambdified from a SymPy expression.

    Args:
        expr: SymPy expression string in ``t``.
        names: Parameter names, in the order the returned callable takes
            them after ``x``.

    Returns:
        ``f(x, *params) -> array``, shaped like ``x``, the calling
        convention every stage adapter and ``scipy.optimize.curve_fit``
        use. A model with no dependence on ``x`` (a bare constant) still
        broadcasts to ``x``'s shape rather than collapsing to a scalar.

    Raises:
        sympy.SympifyError: ``expr`` does not parse as a SymPy
            expression.
    """
    t = sp.Symbol("t")
    f = sp.lambdify(
        [t, *[sp.Symbol(n) for n in names]], sp.sympify(expr), "numpy")

    def call(x: np.ndarray, *p: float) -> np.ndarray:
        return np.asarray(f(x, *p), dtype=float) + 0.0 * np.asarray(x)

    return call


#: Filter window, in samples, for a filter run over a ``stream`` record.
W = 60
#: Sample step, seconds, shared by both streaming scenarios.
DT = 0.1
#: Time of the amplitude jump, seconds, in the ``"jump"`` scenario.
JUMP_AT = 30.0
#: Length of the streaming record, seconds.
T_END = 60.0


def stream(
    kind: str, seed: int, *, noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One streaming scenario: ``A*sin(1.5*t)`` with either a mid-stream
    amplitude jump or a linear amplitude drift.

    Args:
        kind: ``"jump"`` (amplitude steps from 2.0 to 3.5 at ``t = 30``) or
            ``"drift"`` (amplitude ramps linearly from 2.0 to 3.5 over the
            record).
        seed: Seed for the additive noise.
        noise: Relative noise level, dimensionless; the additive noise
            standard deviation is ``noise * 2.0``, the amplitude's low end.

    Returns:
        ``(t, y, amplitude)``: the time grid (0 to 60 s in steps of 0.1 s),
        the noisy signal, and the noise-free amplitude at every sample.

    Raises:
        KeyError: ``kind`` is neither ``"jump"`` nor ``"drift"``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T_END, DT)
    if kind == "jump":
        amp = np.where(t < JUMP_AT, 2.0, 3.5)
    elif kind == "drift":
        amp = 2.0 + 1.5 * t / T_END
    else:
        raise KeyError(f"unknown streaming scenario {kind!r}")
    y = amp * np.sin(1.5 * t) + noise * 2.0 * rng.standard_normal(t.size)
    return t, y, amp
