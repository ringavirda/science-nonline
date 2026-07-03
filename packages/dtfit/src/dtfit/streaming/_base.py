"""Shared plumbing for the streaming filters (EACFilter / LSIFilter).

Both are Kalman-style recursive estimators that differ only in the *measurement*
(an integrated **area** vs a **Legendre spectrum**) and the measurement-specific
hot path (``partial_fit`` and the drift step). Everything else -- the parameter /
uncertainty read-out, the external-regressor handling, prediction at the current
estimate, and the covariance re-arm hook -- is identical and lives here, so the
two filters share one implementation. Each subclass sets the attributes these
methods read (``p``, ``P``, ``params``, ``regressors``, ``_f``, ``_has_reg``,
``drift_inflation``) in its own ``__init__``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np


class _RecursiveFilter:
    """Mixin base: the measurement-agnostic surface of a streaming filter."""

    # Set by each subclass's ``__init__`` (declared here for the type checkers).
    params: list                 # the model's sympy parameter symbols
    p: np.ndarray                # current parameter estimate
    P: np.ndarray                # parameter (Kalman state) covariance
    regressors: list[str]        # external-regressor channel names ([] if none)
    drift_inflation: float
    _f: Callable[..., Any]       # compiled model callable f(t[, regressors], *p)
    _jac: list[Callable[..., Any]]  # compiled d f / d p_k (per parameter)
    _dfdt: Callable[..., Any]    # compiled d f / d t (for coast(); no-regressor models)
    _d2fdt2: Callable[..., Any]  # compiled d^2 f / d t^2 (for coast(order=2))
    _dfdt_jac: list[Callable[..., Any]]    # compiled d/dp (d f/d t) (coast_cov)
    _d2fdt2_jac: list[Callable[..., Any]]  # compiled d/dp (d^2 f/d t^2) (coast_cov)
    # Extrapolable (regressor-dependent) vs nuisance (time-only drift) split, for
    # coasting a regressor model forward with supplied future regressor values.
    _f_reg: Callable[..., Any] | None      # terms containing a regressor
    _f_drift: Callable[..., Any] | None    # time-only nuisance terms
    _f_drift_dt: Callable[..., Any] | None
    _f_drift_d2t: Callable[..., Any] | None
    _has_reg: bool
    _t: list[float]              # current window sample times (newest last)

    @property
    def params_(self) -> dict[str, float]:
        """Current parameter estimate as a ``{name: value}`` mapping."""
        return {str(s): float(v) for s, v in zip(self.params, self.p)}

    @property
    def param_cov_(self) -> np.ndarray:
        """Current parameter covariance ``P`` (the running Kalman state
        covariance), shape ``(n_params, n_params)`` -- the streaming analogue of
        :attr:`dtfit.FittingResult.cov`. Its diagonal's square roots are the
        standard errors (:attr:`stderr_`). Large early on, it contracts as the
        parameters become identified and re-inflates on a detected drift."""
        return self.P

    @property
    def stderr_(self) -> dict[str, float]:
        """Per-parameter running standard errors -- ``sqrt`` of the
        :attr:`param_cov_` diagonal -- as a ``{name: value}`` mapping. The online
        twin of :meth:`dtfit.FittingResult.stderr`, giving an uncertainty band on
        the streamed estimate (embedded control, fault detection)."""
        se = np.sqrt(np.clip(np.diag(self.P), 0.0, None))
        return {str(s): float(v) for s, v in zip(self.params, se)}

    def _compile_model(self, model, t_sym, reg_syms) -> None:
        """Lambdify the model, its per-parameter Jacobian, its time derivatives
        (for :meth:`coast`) and the mixed derivatives (for :meth:`coast_cov`) once,
        off the hot path -- then split it for regressor coasting. Regressor symbols
        are passed positionally before the parameters. Shared by both filters'
        ``__init__`` (the block was previously duplicated verbatim in each)."""
        import sympy as sp

        self._f = sp.lambdify([t_sym, *reg_syms, *self.params], model, "numpy")
        self._jac = [
            sp.lambdify([t_sym, *reg_syms, *self.params], sp.diff(model, p), "numpy")
            for p in self.params
        ]
        # Time derivatives, for coast() dead-reckoning through gaps. Only
        # meaningful without external regressors (a measured regressor has no
        # closed-form time derivative); coast() guards on that.
        self._dfdt = sp.lambdify(
            [t_sym, *reg_syms, *self.params], sp.diff(model, t_sym), "numpy"
        )
        self._d2fdt2 = sp.lambdify(
            [t_sym, *reg_syms, *self.params], sp.diff(model, t_sym, 2), "numpy"
        )
        # Mixed derivatives d/dp_k(df/dt) and d/dp_k(d2f/dt2), for coast_cov()'s
        # gap-growing uncertainty band (no-regressor models only).
        self._dfdt_jac = [
            sp.lambdify([t_sym, *reg_syms, *self.params],
                        sp.diff(sp.diff(model, t_sym), p), "numpy")
            for p in self.params
        ]
        self._d2fdt2_jac = [
            sp.lambdify([t_sym, *reg_syms, *self.params],
                        sp.diff(sp.diff(model, t_sym, 2), p), "numpy")
            for p in self.params
        ]
        # Extrapolable/nuisance split for coast() on regressor models.
        self._compile_regressor_coast(model, t_sym, reg_syms)

    def _compile_regressor_coast(self, model, t_sym, reg_syms) -> None:
        """Split the model into extrapolable (regressor-dependent) and nuisance
        (time-only drift) parts so :meth:`coast` can roll a regressor model
        forward. Called by each subclass ``__init__`` (has the symbolic model)."""
        import sympy as sp

        self._f_reg = self._f_drift = None
        self._f_drift_dt = self._f_drift_d2t = None
        if not reg_syms:
            return
        reg_set = set(reg_syms)
        terms = sp.Add.make_args(sp.expand(model))
        reg_terms = [tm for tm in terms if tm.free_symbols & reg_set]
        drift_terms = [tm for tm in terms if not (tm.free_symbols & reg_set)]
        f_reg = sp.Add(*reg_terms) if reg_terms else sp.Integer(0)
        f_drift = sp.Add(*drift_terms) if drift_terms else sp.Integer(0)
        self._f_reg = sp.lambdify(
            [t_sym, *reg_syms, *self.params], f_reg, "numpy"
        )
        self._f_drift = sp.lambdify([t_sym, *self.params], f_drift, "numpy")
        self._f_drift_dt = sp.lambdify(
            [t_sym, *self.params], sp.diff(f_drift, t_sym), "numpy"
        )
        self._f_drift_d2t = sp.lambdify(
            [t_sym, *self.params], sp.diff(f_drift, t_sym, 2), "numpy"
        )

    def _reg_tuple(self, regressors) -> tuple:
        """Coerce one regressor sample to a tuple ordered like ``self.regressors``."""
        if regressors is None:
            raise ValueError(
                "this model declares external regressors; pass them to partial_fit"
            )
        if isinstance(regressors, Mapping):
            return tuple(float(regressors[r_]) for r_ in self.regressors)
        vals = np.atleast_1d(np.asarray(regressors, dtype=float))
        if vals.size != len(self.regressors):
            raise ValueError(
                f"expected {len(self.regressors)} regressors, got {vals.size}"
            )
        return tuple(float(v) for v in vals)

    def _predict_cols(self, xa: np.ndarray, regressors) -> list[np.ndarray]:
        """Regressor columns broadcast to ``xa`` for :meth:`predict`."""
        if regressors is None:
            raise ValueError("predict() needs regressor values for this model")
        if isinstance(regressors, Mapping):
            return [np.broadcast_to(np.asarray(regressors[r_], float), xa.shape)
                    for r_ in self.regressors]
        arr = np.asarray(regressors, float)
        if arr.ndim == 2 and arr.shape[1] == len(self.regressors):
            return [arr[:, c] for c in range(arr.shape[1])]
        return [np.broadcast_to(arr.reshape(-1)[c], xa.shape)
                for c in range(len(self.regressors))]

    def inflate(self, factor: float | None = None) -> None:
        """Inflate the parameter covariance so new data dominates -- a public
        hook for an *external* maneuver/change detector to re-arm the filter for
        fast re-adaptation without discarding the current estimate.

        Args:
            factor: Covariance multiplier; defaults to ``drift_inflation``.
        """
        self.P = self.P * (self.drift_inflation if factor is None else float(factor))

    def predict(self, x: np.ndarray, regressors=None) -> np.ndarray:
        """Evaluate the model at the current parameter estimate.

        With external regressors, ``regressors`` supplies their value(s) at ``x``
        (a ``{name: array-or-scalar}`` mapping broadcast to ``x``'s shape, or an
        ``(len(x), n_reg)`` array)."""
        xa = np.asarray(x, dtype=float)
        if not self._has_reg:
            # Broadcast to x's shape so a t-independent model (e.g. "c0") still
            # returns an (len(x),) array rather than a bare scalar, matching the
            # documented contract and the regressor/predict_cov branches.
            return np.broadcast_to(np.asarray(self._f(xa, *self.p), float), xa.shape)
        cols = self._predict_cols(xa, regressors)
        return self._f(xa, *cols, *self.p)

    def coast(self, x, *, order: int = 1, regressors=None) -> np.ndarray:
        """Extrapolate *beyond the fitted window* by dead-reckoning, not by
        evaluating the model off its support.

        :meth:`predict` evaluates the fitted model ``f(x)`` directly. That is
        exact inside the window the parameters were identified on, but a
        higher-order model *diverges* once ``x`` runs past it -- a fitted cubic's
        ``c3 * x**3`` term blows up -- so a measurement gap (no ``partial_fit``
        while ``x`` advances) turns a good fit into an unbounded extrapolation.

        ``coast`` instead anchors at the last in-window sample ``a = self._t[-1]``
        and propagates a Taylor expansion from there:

        * ``order=1`` -- position + velocity: ``f(a) + f'(a)*(x - a)`` (constant
          velocity / frozen rate). Bounded for any model; the safe default.
        * ``order=2`` -- also ``+ 1/2 f''(a)*(x - a)**2`` (constant acceleration).

        It reduces to :meth:`predict` at and before the anchor (``x <= a``), so it
        is a drop-in for the whole track: exact where the window supports ``x``,
        bounded dead-reckoning past it.

        **With external regressors**, pass ``regressors`` (the *future* regressor
        value(s) at ``x`` -- e.g. an IMU-propagated motion basis). The model is
        then split into its **extrapolable** part (terms containing a regressor,
        rolled forward with the supplied future regressor) and its **nuisance**
        part (time-only drift terms, which *would* blow up if extrapolated), and
        only the nuisance drift is dead-reckoned (frozen rate). This rolls a fused
        model forward using the sensed regressor instead of a crude finite
        difference. Without ``regressors``, a regressor model raises (the future
        regressor is unknown) -- call :meth:`predict` there.

        Args:
            x: Query time(s).
            order: 1 (constant-velocity coast, default) or 2 (constant-acceleration).
            regressors: Future regressor value(s) at ``x`` (required iff the model
                declares external regressors), as for :meth:`predict`.
        """
        xa = np.asarray(x, dtype=float)
        if self._has_reg:
            if regressors is None or self._f_reg is None:
                raise NotImplementedError(
                    "coast() on a model with external regressors needs the "
                    "future regressor value(s): pass regressors=<value(s) at x> "
                    "(e.g. IMU-propagated), or use predict()."
                )
            cols = self._predict_cols(xa, regressors)
            in_support = self._f(xa, *cols, *self.p)
            if not self._t:
                return np.asarray(in_support, dtype=float)
            a_arr = np.asarray(float(self._t[-1]), dtype=float)
            dt = xa - float(a_arr)
            # Extrapolable part: evaluate at the future time AND the supplied
            # future regressor (rolled forward, not extrapolated blindly).
            reg_part = np.asarray(self._f_reg(xa, *cols, *self.p), dtype=float)
            # Nuisance drift part: dead-reckon from the anchor (frozen rate).
            assert self._f_drift is not None and self._f_drift_dt is not None
            fd = float(np.asarray(self._f_drift(a_arr, *self.p)))
            vd = float(np.asarray(self._f_drift_dt(a_arr, *self.p)))
            drift = fd + vd * dt
            if order >= 2:
                assert self._f_drift_d2t is not None
                ad = float(np.asarray(self._f_drift_d2t(a_arr, *self.p)))
                drift = drift + 0.5 * ad * dt * dt
            coasted = reg_part + drift
            return np.where(dt > 0.0, coasted, np.asarray(in_support, dtype=float))
        in_support = self._f(xa, *self.p)
        if not self._t:
            return np.asarray(in_support, dtype=float)  # nothing anchored yet
        a = float(self._t[-1])
        dt = xa - a
        f_a = float(np.asarray(self._f(np.asarray(a, dtype=float), *self.p)))
        v_a = float(np.asarray(self._dfdt(np.asarray(a, dtype=float), *self.p)))
        coasted = f_a + v_a * dt
        if order >= 2:
            acc_a = float(np.asarray(self._d2fdt2(np.asarray(a, dtype=float), *self.p)))
            coasted = coasted + 0.5 * acc_a * dt * dt
        # In-support points (at or before the anchor) keep the exact fitted
        # prediction; only points past the anchor are dead-reckoned.
        return np.where(dt > 0.0, coasted, np.asarray(in_support, dtype=float))

    def coast_cov(self, x, *, order: int = 1) -> np.ndarray:
        """Predictive variance of :meth:`coast` -- the uncertainty band around a
        gap dead-reckon, propagated from the parameter covariance ``P``.

        This is the companion to :meth:`coast` that :meth:`predict_cov` is to
        :meth:`predict`. The coasted value ``c(x) = f(a) + f'(a) dt [+ 1/2
        f''(a) dt^2]`` (anchor ``a = self._t[-1]``, ``dt = x - a``) is a function
        of the parameters, so its variance is ``J_c(x)^T P J_c(x)`` with the coast
        Jacobian ``J_c[k] = df/dp_k(a) + d f'/dp_k(a) dt [+ 1/2 d f''/dp_k(a)
        dt^2]``. The ``dt`` / ``dt^2`` factors make the band **grow with gap
        length** -- confidence correctly decays the longer the filter coasts --
        so a downstream fuser gets an honest, widening ``coast(x) +/-
        sqrt(coast_cov(x))`` pseudo-measurement across a measurement dropout.

        At and before the anchor (``x <= a``) it returns :meth:`predict_cov`, so
        it is a drop-in for the whole track. Not defined for models with external
        regressors (as :meth:`coast`).

        Args:
            x: Query time(s).
            order: Match the :meth:`coast` order (1 constant-velocity, default; 2
                constant-acceleration).
        """
        if self._has_reg:
            raise NotImplementedError(
                "coast_cov() is undefined for models with external regressors; "
                "use predict_cov()."
            )
        xa = np.asarray(x, dtype=float)
        base_cov = self.predict_cov(xa)
        if not self._t:
            return base_cov
        a = float(self._t[-1])
        dt = xa - a
        a_arr = np.asarray(a, dtype=float)
        jcols = []
        for k in range(len(self.p)):
            jp = float(np.asarray(self._jac[k](a_arr, *self.p)))
            jpt = float(np.asarray(self._dfdt_jac[k](a_arr, *self.p)))
            col = jp + jpt * dt
            if order >= 2:
                jptt = float(np.asarray(self._d2fdt2_jac[k](a_arr, *self.p)))
                col = col + 0.5 * jptt * dt * dt
            jcols.append(np.broadcast_to(np.asarray(col, dtype=float), xa.shape))
        jac = np.stack(jcols, axis=-1)
        var = np.einsum("...i,ij,...j->...", jac, self.P, jac)
        # Past the anchor: the propagated (gap-growing) variance; at/before it the
        # exact in-support predict_cov.
        return np.where(dt > 0.0, np.clip(var, 0.0, None), base_cov)

    def predict_cov(self, x, regressors=None) -> np.ndarray:
        """Predictive variance of the model **output** at ``x``, propagated from
        the parameter covariance: ``Var[f(x)] = J(x)ᵀ P J(x)`` where
        ``J(x) = ∂f/∂params`` (the delta method).

        :attr:`stderr_` gives the uncertainty of each *parameter*; this maps that
        covariance into *output* space, so the streamed estimate carries a
        calibrated one-sigma band ``predict(x) ± sqrt(predict_cov(x))``. That is
        what lets a downstream consumer *fuse* the dtfit output (as a
        pseudo-measurement with a known variance) or gate on its confidence.
        Nearly free: the parameter Jacobian ``∂f/∂p`` is already compiled.

        Note this is the variance from the *estimate's* uncertainty only; add the
        measurement-noise floor separately for a full predictive interval.

        Args:
            x: Query time(s).
            regressors: Regressor value(s) at ``x`` (required iff the model
                declares external regressors), as for :meth:`predict`.

        Returns:
            Variance array shaped like ``x``.
        """
        xa = np.asarray(x, dtype=float)
        if self._has_reg:
            cols = self._predict_cols(xa, regressors)
            jac_cols = [np.broadcast_to(np.asarray(j(xa, *cols, *self.p), float), xa.shape)
                        for j in self._jac]
        else:
            jac_cols = [np.broadcast_to(np.asarray(j(xa, *self.p), float), xa.shape)
                        for j in self._jac]
        jac = np.stack(jac_cols, axis=-1)          # (..., n_params)
        var = np.einsum("...i,ij,...j->...", jac, self.P, jac)
        return np.clip(var, 0.0, None)
