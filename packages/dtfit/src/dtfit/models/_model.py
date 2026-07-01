"""The :class:`Model` -- a named, self-seeding, composable model family.

A :class:`Model` bundles everything needed to *fit a structurally-correct model
without hand-writing sympy strings and guessing ``p0``/``bounds``*: the
expression, its parameters, a shape tag (which decides the estimator variant),
and a **data-driven seeder** that reads sensible initial values and bounds off
the data. It fits via the stable engines (routing through
:func:`dtfit.auto_estimate`), and models compose with ``+`` (e.g. trend +
seasonal, a sum of peaks).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sp

from dtfit.types import FittingResult
from dtfit.methods import fit_lsi, fit_eac
from dtfit.auto import auto_estimate

# A seeder reads (x, y) and returns ``{param_name: (p0, lo, hi)}``.
Seeder = Callable[[np.ndarray, np.ndarray], dict[str, tuple[float, float, float]]]


def _params_of(expr: str, var: str) -> tuple[str, ...]:
    t = sp.Symbol(var)
    f = sp.sympify(expr)
    return tuple(sorted((str(s) for s in f.free_symbols if s != t)))


class Model:
    """A model family: expression + parameters + shape + data-driven seeder.

    Args:
        expr: The sympy model expression (e.g. ``"a*exp(b*x)"``).
        var: The main variable name.
        name: A short human label.
        shape: Routing tag -- one of ``"bulk"``, ``"oscillatory"``,
            ``"transient"``, ``"peak"``, ``"composite"`` -- which decides the
            estimator variant in :meth:`fit` (``method="auto"``).
        freq_param: Name of the angular-frequency parameter, if oscillatory
            (forwarded to the LSI oscillatory recipe).
        seeder: ``(x, y) -> {name: (p0, lo, hi)}`` producing data-driven initial
            values and bounds; ``None`` falls back to ones / unbounded.
    """

    def __init__(
        self,
        expr: str,
        var: str = "x",
        *,
        name: str = "",
        shape: str = "bulk",
        category: str = "general",
        freq_param: str | None = None,
        seeder: Seeder | None = None,
    ) -> None:
        self.expr = expr
        self.var = var
        self.name = name or expr
        self.shape = shape
        self.category = category
        self.freq_param = freq_param
        self.seeder = seeder
        self.params: tuple[str, ...] = _params_of(expr, var)

    def __repr__(self) -> str:
        return f"Model({self.name!r}, expr={self.expr!r}, shape={self.shape!r})"

    # seeding
    def seed(self, x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float, float]]:
        """The data-driven ``{name: (p0, lo, hi)}`` seed map (empty if none)."""
        if self.seeder is None:
            return {}
        return self.seeder(np.asarray(x, float), np.asarray(y, float))

    def _eval_seed(self, x: np.ndarray, seed: dict) -> np.ndarray | None:
        """Evaluate the model at its *seed* parameter values (a cheap approximate
        curve, no fit) -- used to detrend before seeding a composed component."""
        if not seed:
            return None
        t = sp.Symbol(self.var)
        f = sp.sympify(self.expr)
        params = sorted((s for s in f.free_symbols if s != t), key=str)
        vals = [seed.get(str(p), (1.0, 0.0, 0.0))[0] for p in params]
        try:
            fn = sp.lambdify(t, f.subs(dict(zip(params, vals))), "numpy")
            v = np.asarray(fn(np.asarray(x, float)), dtype=float)
        except Exception:
            return None
        x = np.asarray(x, float)
        if v.ndim == 0:
            v = np.full_like(x, float(v))
        return v if np.all(np.isfinite(v)) else None

    def _seed_arrays(self, x, y):
        d = self.seed(x, y)
        if not d:
            return None, None
        p0, bounds = [], []
        for nm in self.params:
            if nm in d:
                v, lo, hi = d[nm]
                p0.append(float(v))
                bounds.append((float(lo), float(hi)))
            else:
                p0.append(1.0)
                bounds.append((-np.inf, np.inf))
        # Bounds enable the global (DE) search, which needs *all* bounds finite;
        # a seed with any unbounded parameter falls back to a local fit from p0.
        if all(np.isfinite(lo) and np.isfinite(hi) for lo, hi in bounds):
            return p0, bounds
        return p0, None

    # fitting
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        method: str = "auto",
        p0=None,
        bounds=None,
    ) -> FittingResult:
        """Fit this model to ``(x, y)``.

        ``method="auto"`` (default) routes by :attr:`shape` through
        :func:`dtfit.auto_estimate`; ``"lsi"``/``"eac"``/``"adaptive"`` force a
        specific engine. Seeds and bounds come from the model's seeder unless
        overridden.
        """
        sp0, sb = self._seed_arrays(x, y)
        p0 = sp0 if p0 is None else p0
        bounds = sb if bounds is None else bounds
        if method == "auto":
            shape = self.shape if self.shape != "composite" else "bulk"
            return auto_estimate(x, y, self.expr, self.var, shape=shape,
                                 freq_param=self.freq_param, p0=p0, bounds=bounds)
        if method == "lsi":
            return fit_lsi(x, y, self.expr, self.var, p0=p0, bounds=bounds,
                           freq_param=self.freq_param)
        if method in ("eac", "adaptive"):
            eb = ([b[0] for b in bounds], [b[1] for b in bounds]) if bounds else None
            wm = "curvature" if method == "adaptive" else "uniform"
            # Both EAC paths forward the model's self-seeded bounds; the adaptive
            # (curvature) path previously dropped them, so a seeded model fit
            # silently ran unconstrained there.
            return fit_eac(x, y, self.expr, self.var, window_mode=wm,
                           p0=p0, bounds=eb)
        raise ValueError(
            f"unknown method {method!r}; expected auto/lsi/eac/adaptive"
        )

    # composition
    def __add__(self, other: "Model") -> "Model":
        """Compose two models additively (e.g. ``trend + seasonal``).

        Colliding parameter names in ``other`` are renamed; the seeders compose
        so the combined model is still self-seeding.
        """
        if other.var != self.var:
            raise ValueError(
                f"cannot add models on different variables: {self.var!r} vs {other.var!r}"
            )
        rename: dict[str, str] = {}
        used = set(self.params)
        for p in other.params:
            new = p
            i = 2
            while new in used:
                new = f"{p}_{i}"
                i += 1
            if new != p:
                rename[p] = new
            used.add(new)
        other_expr = sp.sympify(other.expr)
        if rename:
            other_expr = other_expr.subs({sp.Symbol(k): sp.Symbol(v)
                                          for k, v in rename.items()})
        combined = f"({self.expr}) + ({sp.sstr(other_expr)})"
        freq = self.freq_param or (
            rename.get(other.freq_param, other.freq_param) if other.freq_param else None
        )

        def seeder(x, y):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            s_seed = self.seed(x, y)
            d = dict(s_seed)
            # Seed the second component on the *residual* after removing the
            # first's seed-approximation, so e.g. a cycle is seeded on detrended
            # data (correct frequency/amplitude) rather than on the raw trend.
            resid = y
            approx = self._eval_seed(x, s_seed)
            if approx is not None:
                resid = y - approx
            for k, v in other.seed(x, resid).items():
                d[rename.get(k, k)] = v
            return d

        return Model(combined, self.var, name=f"{self.name}+{other.name}",
                     shape="composite", category="composite", freq_param=freq,
                     seeder=seeder)
