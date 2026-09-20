"""The sixteen-family catalogue shared by the estimation notebooks.

Sixteen model families nonlinear in their parameters, across mechanics,
electronics, spectroscopy, kinetics, biology, reliability and signal
processing, each carrying a real-world domain, a closed-form model, ground
truth and a seed. :func:`simulate` draws one noisy replicate of a family
through the study's own noise convention; :func:`param_error` scores an
estimate against it. :func:`catalog` and :func:`catalog_seed` bridge to
dtfit's own accuracy-gate scenarios, a second and independently maintained
corpus, so the notebooks can compare the two without copying either.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dtfit_experimental.study import paths
from dtfit_experimental.study.montecarlo import noisy


@dataclass(frozen=True)
class Family:
    """One parametric model family drawn from a real domain.

    Attributes:
        key: Short identifier, unique across :data:`FAMILIES`.
        domain: One-line description of the field the family comes from.
        shape: One-word shape label (oscillatory, peak, sigmoid, monotone,
            ...).
        expr: The model as a sympy-parseable string in ``var``.
        var: Name of the independent variable ``expr`` is written in.
        names: Parameter names in the order :attr:`func` takes them; not
            necessarily sympy's sorted order (see :func:`param_error`).
        func: Callable ``func(x, *params) -> array`` with ``params`` in
            ``names`` order.
        truth: Ground-truth parameter values used to simulate data, keyed
            by name.
        span: ``(x0, x1)``, the sampling domain in ``var``'s own units.
        p0: Initial guess in ``names`` order.
        bounds: ``(lo, hi)`` pairs in ``names`` order.
        osc: Name of the angular-frequency parameter that routes a fit
            through the oscillatory recipe (a raised default order and an
            FFT-seeded frequency), or ``None`` for a family with no such
            parameter.
    """

    key: str
    domain: str
    shape: str
    expr: str
    var: str
    names: tuple[str, ...]
    func: Callable[..., np.ndarray]
    truth: dict[str, float]
    span: tuple[float, float]
    p0: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    osc: str | None = None


def _f_damped(t, A, w, z):
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t)


def _f_sine(t, A, c, p, w):
    return c + A * np.sin(w * t + p)


def _f_firstorder(t, K, tau):
    return K * (1 - np.exp(-t / tau))


def _f_biexp(t, a, b, c, d):
    return a * np.exp(-b * t) + c * np.exp(-d * t)


def _f_decay_offset(t, a, b, c):
    return c + a * np.exp(-b * t)


def _f_expgrow(t, a, b):
    return a * np.exp(b * t)


def _f_power(t, a, b):
    return a * (t + 1.0) ** b


def _f_stretched(t, A, q, tau):
    return A * np.exp(-(t / tau) ** q)


def _f_gauss(t, A, mu, s):
    return A * np.exp(-(t - mu) ** 2 / (2 * s ** 2))


def _f_lorentz(t, A, g, mu):
    return A / (1 + ((t - mu) / g) ** 2)


def _f_double_gauss(t, A1, A2, m1, m2, s1, s2):
    return (A1 * np.exp(-(t - m1) ** 2 / (2 * s1 ** 2))
            + A2 * np.exp(-(t - m2) ** 2 / (2 * s2 ** 2)))


def _f_logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def _f_gompertz(t, A, b, c):
    return A * np.exp(-b * np.exp(-c * t))


def _f_weibull(t, K, k, lam):
    return K * (1 - np.exp(-(t / lam) ** k))


def _f_mm(t, Km, Vmax):
    return Vmax * t / (Km + t)


def _f_hill(t, K, Vmax, nh):
    return Vmax * t ** nh / (K ** nh + t ** nh)


FAMILIES: tuple[Family, ...] = (
    Family(key="damped", domain="mechanical / control", shape="oscillatory",
           expr="A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)", var="t",
           names=("A", "w", "z"), func=_f_damped,
           truth={"A": 2.0, "w": 3.0, "z": 0.15}, span=(0, 6),
           p0=(1.0, 2.0, 0.1), bounds=((0.1, 5), (1, 6), (0.01, 0.9)),
           osc="w"),
    Family(key="sine", domain="signal / vibration", shape="oscillatory",
           expr="c + A*sin(w*t + p)", var="t", names=("A", "c", "p", "w"),
           func=_f_sine, truth={"A": 2.0, "c": 1.5, "p": 0.5, "w": 3.0},
           span=(0, 6), p0=(2.0, 1.5, 0.0, 3.0),
           bounds=((0.3, 5), (0, 4), (-np.pi, np.pi), (1, 6)), osc="w"),
    Family(key="firstorder", domain="electrical / RC", shape="saturating-exp",
           expr="K*(1-exp(-t/tau))", var="t", names=("K", "tau"),
           func=_f_firstorder, truth={"K": 3.0, "tau": 1.2}, span=(0, 6),
           p0=(1.0, 1.0), bounds=((0.1, 10), (0.05, 5))),
    Family(key="biexp", domain="pharmacokinetics", shape="multi-exp",
           expr="a*exp(-b*t) + c*exp(-d*t)", var="t",
           names=("a", "b", "c", "d"), func=_f_biexp,
           truth={"a": 2.0, "b": 2.0, "c": 1.0, "d": 0.3}, span=(0, 6),
           p0=(1.5, 1.5, 1.0, 0.5),
           bounds=((0.1, 5), (0.2, 5), (0.1, 5), (0.05, 2))),
    Family(key="decay_offset", domain="thermal / sensor (Newton cooling)",
           shape="decay-to-baseline", expr="c + a*exp(-b*t)", var="t",
           names=("a", "b", "c"), func=_f_decay_offset,
           truth={"a": 3.0, "b": 0.8, "c": 1.0}, span=(0, 8),
           p0=(2.0, 1.0, 0.5), bounds=((0.5, 6), (0.1, 3), (0.0, 3))),
    Family(key="expgrow", domain="growth / finance", shape="monotone",
           expr="a*exp(b*t)", var="t", names=("a", "b"), func=_f_expgrow,
           truth={"a": 1.5, "b": 0.6}, span=(0, 4), p0=(1.0, 1.0),
           bounds=((0.2, 5), (0.05, 2))),
    Family(key="power", domain="physics / scaling law", shape="monotone",
           expr="a*(t+1)**b", var="t", names=("a", "b"), func=_f_power,
           truth={"a": 2.0, "b": 1.4}, span=(0, 6), p0=(1.0, 1.0),
           bounds=((0.2, 5), (0.3, 3))),
    Family(key="stretched", domain="disordered relaxation (KWW)",
           shape="multi-exp", expr="A*exp(-(t/tau)**q)", var="t",
           names=("A", "q", "tau"), func=_f_stretched,
           truth={"A": 3.0, "q": 1.6, "tau": 2.0}, span=(0.02, 8),
           p0=(2.0, 1.0, 1.5), bounds=((0.5, 6), (0.3, 3), (0.3, 5))),
    Family(key="gauss", domain="spectroscopy", shape="peak",
           expr="A*exp(-(t-mu)**2/(2*s**2))", var="t",
           names=("A", "mu", "s"), func=_f_gauss,
           truth={"A": 4.0, "mu": 3.0, "s": 0.8}, span=(0, 6),
           p0=(2.0, 2.5, 1.0), bounds=((0.5, 8), (1, 5), (0.2, 2))),
    Family(key="lorentz", domain="spectroscopy (resonance)", shape="peak",
           expr="A/(1 + ((t-mu)/g)**2)", var="t", names=("A", "g", "mu"),
           func=_f_lorentz, truth={"A": 4.0, "g": 0.7, "mu": 3.0},
           span=(0, 6), p0=(2.0, 0.5, 2.5),
           bounds=((0.5, 8), (0.2, 2), (1, 5))),
    Family(key="double_gauss", domain="chromatography", shape="peak",
           expr="A1*exp(-(t-m1)**2/(2*s1**2)) + A2*exp(-(t-m2)**2/(2*s2**2))",
           var="t", names=("A1", "A2", "m1", "m2", "s1", "s2"),
           func=_f_double_gauss,
           truth={"A1": 4.0, "A2": 2.5, "m1": 2.0, "m2": 4.0, "s1": 0.5,
                  "s2": 0.7}, span=(0, 6),
           p0=(3.0, 2.0, 1.5, 3.5, 0.6, 0.6),
           bounds=((1, 8), (1, 6), (0.5, 3), (3, 5.5), (0.2, 1.5),
                   (0.2, 1.5))),
    Family(key="logistic", domain="epidemiology", shape="sigmoid",
           expr="K/(1+exp(-r*(t-t0)))", var="t", names=("K", "r", "t0"),
           func=_f_logistic, truth={"K": 5.0, "r": 1.5, "t0": 3.0},
           span=(0, 6), p0=(3.0, 1.0, 2.5), bounds=((1, 10), (0.3, 4), (1, 5))),
    Family(key="gompertz", domain="tumour / population growth",
           shape="sigmoid", expr="A*exp(-b*exp(-c*t))", var="t",
           names=("A", "b", "c"), func=_f_gompertz,
           truth={"A": 5.0, "b": 3.0, "c": 0.8}, span=(0, 8),
           p0=(3.0, 2.0, 0.5), bounds=((1, 10), (0.5, 8), (0.1, 3))),
    Family(key="weibull", domain="reliability (failure CDF)",
           shape="sigmoid", expr="K*(1-exp(-(t/lam)**k))", var="t",
           names=("K", "k", "lam"), func=_f_weibull,
           truth={"K": 4.0, "k": 2.2, "lam": 3.0}, span=(0.02, 8),
           p0=(3.0, 1.5, 2.0), bounds=((1, 8), (0.5, 4), (0.5, 6))),
    Family(key="mm", domain="enzyme kinetics", shape="rational-saturating",
           expr="Vmax*t/(Km+t)", var="t", names=("Km", "Vmax"), func=_f_mm,
           truth={"Km": 1.5, "Vmax": 5.0}, span=(0.05, 8),
           p0=(1.0, 3.0), bounds=((0.2, 5), (1, 10))),
    Family(key="hill", domain="pharmacology (dose-response)",
           shape="rational-saturating",
           expr="Vmax*t**nh/(K**nh + t**nh)", var="t",
           names=("K", "Vmax", "nh"), func=_f_hill,
           truth={"K": 2.0, "Vmax": 5.0, "nh": 2.5}, span=(0.02, 8),
           p0=(1.5, 3.0, 2.0), bounds=((0.3, 5), (1, 10), (1, 5))),
)


def simulate(
    family: Family,
    rng: np.random.Generator,
    *,
    n: int = 220,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one noisy replicate of ``family``.

    Args:
        family: Catalogue entry to draw from.
        rng: Generator drawing the noise.
        n: Sample count over ``family.span``, at least 2. ``n=1`` returns a
            single sample with ``sigma`` fallen back to ``noise`` (the flat
            signal path of :func:`~dtfit_experimental.study.montecarlo.noisy`);
            ``n=0`` raises.
        noise: Relative noise level forwarded to
            :func:`dtfit_experimental.study.montecarlo.noisy` (a fraction
            of the clean signal's range), in ``[0, inf)``.

    Returns:
        ``(x, y, clean)``: sample positions, noisy response and clean
        response, each of shape ``(n,)``.

    Raises:
        ValueError: ``n=0``, from ``numpy.ptp`` on the empty clean array.
    """
    x = np.linspace(family.span[0], family.span[1], n)
    clean = family.func(x, *[family.truth[k] for k in family.names])
    y, _ = noisy(clean, noise, rng)
    return x, y, clean


def param_error(estimate: dict[str, float], family: Family) -> float:
    """Median relative parameter-recovery error, in percent.

    Args:
        estimate: Fitted values keyed by parameter name. Either the
            sympy-sorted or the declared order of ``family.names`` works to
            build this dict, since the lookup here is by name, not
            position.
        family: Catalogue entry the estimate is scored against.

    Returns:
        The median over ``family.names`` of
        ``abs(estimate[name] - family.truth[name]) / abs(family.truth[name])``,
        in percent. ``nan`` when ``family.names`` is empty.

    Raises:
        KeyError: ``estimate`` is missing one of ``family.names``.
        ZeroDivisionError: ``family.truth`` holds ``0.0`` for one of
            ``family.names``.
    """
    errs = [
        abs(estimate[k] - family.truth[k]) / abs(family.truth[k])
        for k in family.names
    ]
    if not errs:
        return float("nan")
    return float(np.median(errs)) * 100.0


def catalog() -> list[Any]:
    """dtfit's own accuracy-gate scenarios, imported live from the checkout.

    Resolves ``packages/dtfit/tests`` from
    ``study.paths.experiments_dir().parents[1]`` and loads its
    ``accuracy/scenarios.py`` module straight off disk (by file location,
    not by inserting the directory on ``sys.path``), reading it live rather
    than copying it, so the catalogue always compares against exactly the
    corpus dtfit's own tests check. The loaded module is registered under
    ``sys.modules`` only for the duration of the call (``scenarios.py``'s
    ``@dataclass`` needs it there) and removed again before returning.

    Returns:
        The corpus's ``accuracy.scenarios.SCENARIOS`` list. An empty list,
        with one line printed to stdout,
        when the checkout holding ``packages/dtfit/tests/accuracy`` is not
        present alongside the experiments tree -- an installed wheel
        carries no such tree, and a resolver pointed at the wrong
        directory finds no corpus there either.
    """
    try:
        tests_dir = paths.experiments_dir().parents[1] / "dtfit" / "tests"
    except FileNotFoundError as exc:
        print(f"families.catalog: {exc}")
        return []
    scenarios_path = tests_dir / "accuracy" / "scenarios.py"
    if not scenarios_path.is_file():
        print(f"families.catalog: no accuracy corpus at {tests_dir}")
        return []
    module_name = "_dtfit_accuracy_scenarios"
    spec = importlib.util.spec_from_file_location(module_name, scenarios_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return list(module.SCENARIOS)
    finally:
        del sys.modules[module_name]


def catalog_seed(
    scenario: Any, x: np.ndarray, y: np.ndarray
) -> tuple[list[float] | None, list[tuple[float, float]] | None]:
    """The corpus's own seed for one scenario, so every method starts alike.

    Args:
        scenario: An entry of :func:`catalog`'s return, a dtfit
            ``accuracy.scenarios.Scenario``.
        x: Sample positions.
        y: Sample values matching ``x``.

    Returns:
        ``(p0, bounds)`` from the scenario's model's own seeder
        (``Model._seed_arrays``): ``p0`` in the model's canonical
        (sympy-sorted) parameter order, ``bounds`` the matching pairs, or
        ``None`` for either when the seeder finds no closed form for this
        model on ``(x, y)``, or for ``bounds`` alone when every bound it
        found is infinite.
    """
    return scenario.model()._seed_arrays(x, y)
