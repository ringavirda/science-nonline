"""The :class:`Model` -- a named, self-seeding, composable model family.

A :class:`Model` bundles everything needed to *fit a structurally-correct model
without hand-writing sympy strings and guessing ``p0``/``bounds``*: the
expression, its parameters, a shape tag (which decides the estimator variant),
and a **data-driven seeder** that reads sensible initial values and bounds off
the data. It fits via the stable engines (routing through
:func:`dtfit.auto_estimate`), and models compose with ``+`` (e.g. trend +
seasonal, a sum of peaks).

A model may be given symbolically (a SymPy expression string, the historical
form) **or** as a plain Python callable ``f(x, *params)``. A callable is
resolved through :func:`dtfit.methods.resolve_model`, so its parameter order is
the callable's *signature* order (not the sorted order symbolic models use) and
it fits through the same engines. Composition with ``+`` and the seed-detrend
evaluator require *symbolic* operands; a callable model raises a clear error
there (see :meth:`__add__`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import sympy as sp

from dtfit.types import FittingResult
from dtfit.methods import fit_lsi, fit_eac, resolve_model
from dtfit.auto import auto_estimate

# A seeder reads (x, y) and returns ``{param_name: (p0, lo, hi)}``.
Seeder = Callable[[np.ndarray, np.ndarray], dict[str, tuple[float, float, float]]]

# A model is a SymPy-expression string (symbolic) or a callable ``f(x, *params)``.
ModelExpr = str | Callable[..., Any]


def _params_of(
    expr: ModelExpr, var: str, param_names: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Canonical parameter order of ``expr``.

    Symbolic expressions parse their free symbols (sorted by name, the historical
    layout); a callable is delegated to :func:`dtfit.methods.resolve_model`, whose
    :attr:`~dtfit.methods.ModelSpec.names` are the callable's *signature* order
    (introspected, or ``param_names`` when given / not introspectable).
    """
    if callable(expr):
        return resolve_model(expr, var, param_names=param_names).names
    t = sp.Symbol(var)
    f = sp.sympify(expr)
    names = tuple(sorted((str(s) for s in f.free_symbols if s != t)))
    if param_names is not None:
        given = tuple(str(n) for n in param_names)
        if sorted(given) != sorted(names):
            raise ValueError(
                f"param_names {list(given)} do not match the expression's "
                f"parameters {list(names)}."
            )
    return names


class Model:
    """A model family: expression + parameters + shape + data-driven seeder.

    Args:
        expr: The model. Either a SymPy expression string (e.g. ``"a*exp(b*x)"``)
            or a plain Python callable ``f(x, *params)`` (see
            :meth:`from_callable`). A callable is resolved via
            :func:`dtfit.methods.resolve_model` and its parameters keep their
            *signature* order.
        var: The main variable name. For a callable it is a label only (defaults
            to ``"x"``).
        name: A short human label.
        shape: Routing tag -- one of ``"bulk"``, ``"oscillatory"``,
            ``"transient"``, ``"peak"``, ``"composite"`` -- which decides the
            estimator variant in :meth:`fit` (``method="auto"``).
        freq_param: Name of the angular-frequency parameter, if oscillatory
            (forwarded to the LSI oscillatory recipe).
        seeder: ``(x, y) -> {name: (p0, lo, hi)}`` producing data-driven initial
            values and bounds; ``None`` falls back to ones / unbounded.
        param_names: For a callable ``expr`` whose parameter names cannot be
            introspected (a builtin, or an ``f(x, *params)`` signature), the names
            of the parameters after the leading ``x`` (in call order). Ignored for
            a symbolic ``expr`` except that, if given, it is validated against the
            parsed names.

    Attributes:
        is_symbolic: ``True`` for a symbolic (string) model, ``False`` for a
            callable one.
        expr: The SymPy expression string when symbolic, else ``None``.
        func: The Python callable when non-symbolic, else ``None``.
        params: The parameter names in canonical order (sorted for a symbolic
            model, signature order for a callable) -- the layout of ``p0`` /
            ``bounds`` and of :attr:`dtfit.types.FittingResult.names`.
    """

    def __init__(
        self,
        expr: ModelExpr,
        var: str = "x",
        *,
        name: str = "",
        shape: str = "bulk",
        category: str = "general",
        freq_param: str | None = None,
        seeder: Seeder | None = None,
        param_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        pnames = tuple(param_names) if param_names is not None else None
        if callable(expr):
            # Resolve once to validate the callable and pin the canonical
            # (signature-order) parameter names; keep the callable for the fitters.
            spec = resolve_model(expr, var, param_names=pnames)
            self.is_symbolic = False
            self.expr: str | None = None
            self.func: Callable[..., Any] | None = expr
            self.var = spec.var
            self.params: tuple[str, ...] = spec.names
            self.name = name or getattr(expr, "__name__", "") or "callable"
        else:
            expr_str = expr if isinstance(expr, str) else str(expr)
            self.is_symbolic = True
            self.expr = expr_str
            self.func = None
            self.var = var
            self.params = _params_of(expr_str, var, pnames)
            self.name = name or expr_str
        self.shape = shape
        self.category = category
        self.freq_param = freq_param
        self.seeder = seeder

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        names: tuple[str, ...] | list[str] | None = None,
        *,
        var: str = "x",
        name: str = "",
        shape: str = "bulk",
        category: str = "general",
        freq_param: str | None = None,
        seeder: Seeder | None = None,
    ) -> "Model":
        """Build a :class:`Model` from a Python callable ``func(x, *params)``.

        A convenience wrapper over the constructor's callable path. ``names`` are
        the parameter names after the leading ``x`` (in call order); they are
        introspected from the signature when omitted and are only *required* for a
        callable with no inspectable signature (a builtin) or an ``*params``
        signature. The parameter order is the callable's signature order.
        """
        return cls(
            func,
            var,
            name=name,
            shape=shape,
            category=category,
            freq_param=freq_param,
            seeder=seeder,
            param_names=names,
        )

    def __repr__(self) -> str:
        what = repr(self.expr) if self.is_symbolic else "<callable>"
        return f"Model({self.name!r}, expr={what}, shape={self.shape!r})"

    # seeding
    def seed(self, x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float, float]]:
        """The data-driven ``{name: (p0, lo, hi)}`` seed map (empty if none)."""
        if self.seeder is None:
            return {}
        return self.seeder(np.asarray(x, float), np.asarray(y, float))

    def _eval_seed(self, x: np.ndarray, seed: dict) -> np.ndarray | None:
        """Evaluate the model at its *seed* parameter values (a cheap approximate
        curve, no fit) -- used to detrend before seeding a composed component.

        Symbolic only: a callable model cannot participate in composition (see
        :meth:`__add__`), so this returns ``None`` for one.
        """
        if not seed or not self.is_symbolic:
            return None
        assert self.expr is not None
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
        # Partially-infinite bounds are kept: the solvers skip the global (DE)
        # stage unless every bound is finite, but the local (trf) solve honours
        # mixed bounds directly (see ``solve_weighted_nlls``), so a seeder's
        # positivity guard (e.g. sigma > 0) survives an otherwise unbounded
        # seed instead of being dropped wholesale. A *fully* unbounded seed
        # carries no constraint at all, so it maps to ``None`` -- keeping the
        # unconstrained (LM) solver instead of pointlessly forcing the bounded
        # path (which measurably degrades e.g. tanh_step accuracy).
        if all(np.isneginf(lo) and np.isposinf(hi) for lo, hi in bounds):
            return p0, None
        return p0, bounds

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
        overridden. A callable model is passed straight through to the fitters
        (which resolve it via :func:`dtfit.methods.resolve_model`).
        """
        sp0, sb = self._seed_arrays(x, y)
        p0 = sp0 if p0 is None else p0
        bounds = sb if bounds is None else bounds
        # The fitters accept either a sympy string (symbolic) or the callable
        # itself; both resolve to the same canonical parameter order.
        model: ModelExpr = self.expr if self.is_symbolic else self.func  # type: ignore[assignment]
        # For a callable, pin the names the fitter must use: a callable is
        # re-resolved inside the fitter, and re-introspection would either fail
        # (an ``f(x, *params)`` signature) or, for a user-renamed callable, yield
        # the raw signature names instead of ``self.params``. Forwarding
        # ``param_names`` keeps the canonical order the Model already committed to.
        # A symbolic model re-parses to the same sorted names, so it needs none.
        pnames: tuple[str, ...] | None = None if self.is_symbolic else self.params
        if method == "auto":
            # A composite (e.g. trend + sine) fits as 'bulk' LSI but still carries
            # its freq_param: the cycle is resolved by the tight FFT seed the
            # composed seeder computes on the detrended residual, which is
            # empirically more robust here than forcing the full oscillatory
            # recipe (whose raised order can over-fit a trend+cycle spectrum).
            shape = self.shape if self.shape != "composite" else "bulk"
            # Forward ``param_names`` so a callable's committed names survive the
            # re-resolution inside the base fitters (a ``*params`` signature would
            # otherwise fail to re-introspect, and a renamed callable would revert
            # to its raw signature names). Symbolic models pass ``None``.
            return auto_estimate(x, y, model, self.var, shape=shape,
                                 freq_param=self.freq_param, p0=p0, bounds=bounds,
                                 param_names=pnames)
        if method == "lsi":
            return fit_lsi(x, y, model, self.var, p0=p0, bounds=bounds,
                           freq_param=self.freq_param, param_names=pnames)
        if method in ("eac", "adaptive"):
            wm = "curvature" if method == "adaptive" else "uniform"
            # Both EAC paths forward the model's self-seeded bounds; the adaptive
            # (curvature) path previously dropped them, so a seeded model fit
            # silently ran unconstrained there. The pair list is fit_eac's
            # canonical form — no scipy-tuple conversion (ambiguous for
            # 2-parameter models and lossy for partially-infinite bounds).
            return fit_eac(x, y, model, self.var, window_mode=wm,
                           p0=p0, bounds=bounds, param_names=pnames)
        raise ValueError(
            f"unknown method {method!r}; expected auto/lsi/eac/adaptive"
        )

    # composition
    def __add__(self, other: "Model") -> "Model":
        """Compose two models additively (e.g. ``trend + seasonal``).

        Colliding parameter names in ``other`` are renamed; the seeders compose
        so the combined model is still self-seeding.

        Symbolic composition only: both operands must be symbolic (string)
        models -- a callable's expression cannot be manipulated / renamed
        symbolically, so composing one raises :class:`TypeError`.
        """
        if not self.is_symbolic or not other.is_symbolic:
            raise TypeError(
                "cannot compose a callable model with '+': symbolic composition "
                "requires both operands to be SymPy-expression models (rename / "
                "detrend needs a manipulable expression). Compose the symbolic "
                "forms, or fit the callable model on its own."
            )
        assert self.expr is not None and other.expr is not None
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
