"""Public data types shared across the fitting methods."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Literal, overload

import numpy as np

# Array-like initial parameter guess accepted by the fitters; coerced with
# ``np.asarray`` internally, so a plain list of floats is fine.
InitialGuess = Sequence[float] | np.ndarray | None


@functools.lru_cache(maxsize=256)
def _parse_model(expr: str, var: str):
    """Parse ``(expr, var)`` to ``(sympy, t, f, params)`` -- cached.

    ``sympify`` + free-symbol sorting is pure overhead when the same expression
    is re-parsed (every ``.model`` / ``predict(return_std=True)`` access, and
    across the many :class:`FittingResult` objects a batch fit produces that all
    share one model). SymPy expressions are immutable, so the parse is safely
    memoized on ``(expr, var)``. ``params`` is returned as a tuple for hashing;
    callers that need a list copy it.
    """
    import sympy as sp

    t = sp.Symbol(var)
    f = sp.sympify(expr)
    params = tuple(sorted((s for s in f.free_symbols if s != t), key=str))
    return sp, t, f, params


class FittingResult:
    """A fitted nonlinear model: parameters, their uncertainty, and the model.

    Carries enough to be **self-describing** -- the model expression, variable
    and parameter names alongside the coefficients -- so it exposes named
    parameters, uncertainty (from the covariance), prediction with error bands,
    and round-trips to a plain dict for storage / deployment.

    Attributes:
        coeffs: Fitted coefficients, ordered by parameter name (the layout the
            DT fitters use).
        cov: Parameter covariance estimate (``n_params x n_params``) when the
            method can produce one from an overdetermined system; ``None``
            otherwise. Its diagonal's square roots are the standard errors.
        expr: The model expression (e.g. ``"a*exp(b*t)"``), when known. Enables
            :meth:`to_dict` round-tripping and prediction error bands.
        var: The model's main variable name.
        names: Parameter names aligned with ``coeffs``.
        model: a precomputed NumPy callable of the fitted model; if omitted it is
            lambdified lazily from ``expr`` + ``coeffs``.
        label: Optional tag carried through batch/parallel fits (a channel name,
            grid cell, ...); ``None`` for a plain single fit.
        error: Set to a message instead of coefficients when a fit failed inside
            a batch (:func:`dtfit.fit_many`), so one bad problem does not abort
            the batch; ``None`` for a successful fit.
        converged: Whether the underlying optimizer reported convergence. The
            iterative fitters (``fit_lsi`` / ``fit_eac``) set this from the
            solver; ``None`` means the method does not report it. A *successful
            call* with ``converged is False`` is the silent-failure case to
            watch -- a result is returned but the optimizer did not settle.
        message: The optimizer's termination message, when available.
        x_range: ``(min, max)`` of the training ``x``, recorded so
            :meth:`predict` can warn on extrapolation; ``None`` when unknown.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        cov: np.ndarray | None = None,
        expr: str | None = None,
        var: str | None = None,
        names: tuple[str, ...] | Sequence[str] = (),
        model: Callable[..., Any] | None = None,
        *,
        label: Any = None,
        error: str | None = None,
        converged: bool | None = None,
        message: str | None = None,
        x_range: tuple[float, float] | None = None,
    ) -> None:
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.cov = cov
        self.expr = expr
        self.var = var
        self.names: tuple[str, ...] = tuple(names)
        self._model = model
        self.label = label
        self.error = error
        self.converged = converged
        self.message = message
        self.x_range = (
            None if x_range is None
            else (float(x_range[0]), float(x_range[1]))
        )

    def __repr__(self) -> str:
        if self.error is not None:
            return f"FittingResult(error={self.error!r}, label={self.label!r})"
        conv = "" if self.converged is not False else ", converged=False"
        return (f"FittingResult(expr={self.expr!r}, "
                f"params={self.params!r}{conv})")

    # Picklability: the lazily-built ``_model`` is a lambdified closure that does
    # not survive a process-pool round trip, so drop it on pickling -- it rebuilds
    # from ``expr``/``coeffs`` on first access. This is what lets a worker return
    # a fitted result across the ``fit_many`` (loky) boundary.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # the fitted model (callable) -- precomputed or rebuilt from expr
    @property
    def model(self) -> Callable[..., Any]:
        """The fitted model as a NumPy callable ``f(x) -> y``."""
        if self._model is None:
            self._model = self._lambdify(self.coeffs)
        return self._model

    def _sympy(self):
        if self.expr is None or self.var is None:
            raise ValueError(
                "this FittingResult has no expr/var; the operation needs the "
                "model expression (only the precomputed callable is available)."
            )
        sp, t, f, params = _parse_model(self.expr, self.var)
        return sp, t, f, list(params)

    def _lambdify(self, coeffs: np.ndarray) -> Callable[..., Any]:
        sp, t, f, params = self._sympy()
        return sp.lambdify(t, f.subs(dict(zip(params, coeffs))), "numpy")

    # named parameters
    @property
    def params(self) -> dict[str, float]:
        """Fitted parameters as a ``{name: value}`` mapping."""
        names = self.names or tuple(f"p{i}" for i in range(self.coeffs.size))
        return {n: float(c) for n, c in zip(names, self.coeffs)}

    # uncertainty quantification
    def stderr(self) -> dict[str, float]:
        """Per-parameter standard errors (``sqrt`` of the covariance diagonal)."""
        if self.cov is None:
            raise ValueError("no covariance available for this fit (cov is None).")
        se = np.sqrt(np.clip(np.diag(np.asarray(self.cov, float)), 0.0, None))
        names = self.names or tuple(f"p{i}" for i in range(self.coeffs.size))
        return {n: float(s) for n, s in zip(names, se)}

    def confidence_intervals(self, level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Per-parameter confidence intervals at ``level`` (normal approximation)."""
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2.0))
        se = self.stderr()
        return {n: (float(v) - z * se[n], float(v) + z * se[n])
                for n, v in self.params.items()}

    @overload
    def predict(self, x: np.ndarray, *,
                return_std: Literal[False] = False,
                warn_extrapolation: bool = ...) -> np.ndarray: ...
    @overload
    def predict(self, x: np.ndarray, *,
                return_std: Literal[True],
                warn_extrapolation: bool = ...) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(self, x: np.ndarray, *, return_std: bool = False,
                warn_extrapolation: bool = False):
        """Evaluate the fitted model at ``x``.

        With ``return_std=True`` also returns the 1-sigma prediction standard
        deviation propagated from the parameter covariance (delta method); this
        needs ``cov`` and the model ``expr``.

        With ``warn_extrapolation=True`` a :class:`UserWarning` is issued when any
        ``x`` falls outside the fitted training range (:attr:`x_range`) -- a
        nonlinear model can extrapolate to nonsense, so this catches the most
        common curve-fitting mistake. No-op when the training range is unknown.
        """
        x = np.asarray(x, dtype=float)
        if warn_extrapolation and self.x_range is not None and x.size:
            lo, hi = self.x_range
            xmin, xmax = float(np.min(x)), float(np.max(x))
            if xmin < lo or xmax > hi:
                warnings.warn(
                    f"predict() called outside the fitted range "
                    f"[{lo:.6g}, {hi:.6g}] (got [{xmin:.6g}, {xmax:.6g}]); "
                    "the nonlinear model is extrapolating.",
                    UserWarning,
                    stacklevel=2,
                )
        y = np.asarray(self.model(x), dtype=float)
        if np.ndim(y) == 0:
            y = np.full_like(x, float(y))
        if not return_std:
            return y
        if self.cov is None:
            raise ValueError("prediction std needs a covariance (cov is None).")
        sp, t, f, params = self._sympy()
        base = self.coeffs.astype(float)
        # Lambdify the model ONCE over ``(t, *params)`` and finite-difference each
        # parameter against it, rather than re-lambdifying ``f.subs(...)`` inside
        # the loop -- that was ``len(params)`` separate SymPy compiles on every
        # uncertainty-band call. Numerics are unchanged (same step, same formula).
        f_lam = sp.lambdify((t, *params), f, "numpy")
        # Finite-difference against f_lam's OWN baseline (not self.model(x)); if a
        # caller passed an explicit model= that isn't exactly f.subs(coeffs) the
        # two would disagree and skew the band. For the normal case (model derived
        # from expr) y0 == y to machine precision, so numerics are unchanged.
        y0 = np.asarray(f_lam(x, *base), dtype=float)
        if np.ndim(y0) == 0:
            y0 = np.full_like(x, float(y0))
        jac = np.empty((x.size, len(params)))
        for k in range(len(params)):
            step = 1e-6 * max(1.0, abs(base[k]))
            cp = base.copy()
            cp[k] += step
            vk = np.asarray(f_lam(x, *cp), dtype=float)
            if np.ndim(vk) == 0:
                vk = np.full_like(x, float(vk))
            jac[:, k] = (vk - y0) / step
        var = np.einsum("ij,jk,ik->i", jac, np.asarray(self.cov, float), jac)
        return y, np.sqrt(np.clip(var, 0.0, None))

    # serialization / deployment
    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation (JSON-friendly) for storage / shipping.

        Captures the expression, variable, names, coefficients and covariance --
        everything needed to rebuild the model with :meth:`from_dict`. Requires
        ``expr``/``var`` (a fit with only a precomputed callable cannot be
        serialized).
        """
        if self.expr is None or self.var is None:
            raise ValueError(
                "cannot serialize a FittingResult without expr/var "
                "(only a precomputed model callable is available)."
            )
        return {
            "expr": self.expr,
            "var": self.var,
            "names": list(self.names),
            "coeffs": self.coeffs.tolist(),
            "cov": None if self.cov is None else np.asarray(self.cov, float).tolist(),
            "x_range": None if self.x_range is None else list(self.x_range),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FittingResult":
        """Rebuild a :class:`FittingResult` from :meth:`to_dict` output."""
        cov = d.get("cov")
        xr = d.get("x_range")
        return cls(
            coeffs=np.asarray(d["coeffs"], dtype=float),
            cov=None if cov is None else np.asarray(cov, dtype=float),
            expr=d["expr"],
            var=d["var"],
            names=tuple(d.get("names", ())),
            x_range=None if xr is None else (float(xr[0]), float(xr[1])),
        )

    # human-readable summary
    def summary(self) -> str:
        """A short text summary of the fit (parameters +/- standard errors)."""
        lines = [f"FittingResult: {self.expr or '<callable>'}"]
        se = self.stderr() if self.cov is not None else None
        for n, v in self.params.items():
            if se is not None:
                lines.append(f"  {n} = {v:.6g} +/- {se[n]:.3g}")
            else:
                lines.append(f"  {n} = {v:.6g}")
        if self.converged is False:
            msg = f" ({self.message})" if self.message else ""
            lines.append(f"  [warning] optimizer did not converge{msg}")
        return "\n".join(lines)
