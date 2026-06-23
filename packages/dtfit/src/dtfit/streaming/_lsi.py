"""Recursive Legendre-spectrum filter -- the streaming counterpart of LSI.

This is the online analogue of the batch integral-least-squares method
(:func:`dtfit.methods._lsi.fit_lsi`). It is structurally identical to
:class:`dtfit.streaming._eac.EACFilter`: a Kalman-style recursive
estimator whose "measurement" is an innovation between an experimental quantity
and the model's prediction of it, with a measurement Jacobian of integrated
parameter sensitivities. The *only* difference is the measurement itself.

``EACFilter`` measures **areas** -- the signal projected onto piecewise
indicator functions (the zeroth moment over each sub-window). This filter
instead measures the **Legendre spectrum** -- the signal projected onto the
first ``order + 1`` orthogonal Legendre polynomials over the window. That gives
three concrete advantages over a stack of contiguous sub-areas:

* **Observability.** A single window yields ``order + 1`` independent equations
  (not just 1, or a handful of correlated sub-areas), so coupled multi-parameter
  and oscillatory models are identified faster. Areas see only net signed area
  (a low-order moment) and are nearly blind to frequency/phase; Legendre moments
  resolve shape.
* **Conditioning / scaling.** The Legendre basis is orthogonal, so the
  measurement covariance is naturally **diagonal** (``R_j ∝ (2j+1)``, the LSI
  orthonormal weight). Contiguous sub-areas are correlated and shrink in
  magnitude, which is why the area filter needs an ``adapt_r`` rescaling hack.
* **Drift test.** The diagonal ``R`` makes a proper multivariate Normalized
  Innovation Squared (NIS) a clean chi-squared with ``order + 1`` dof.

The empirical spectrum is computed by a cached Legendre projection on a window
normalized to ``[-1, 1]`` (an ``O(W·order)`` mat-vec, assuming roughly uniform
streaming); the model spectrum is computed by Gauss-Legendre quadrature so the
model is integrated exactly -- faithfully mirroring the batch LSI scheme. The
symbolic model and its derivatives are compiled once in ``__init__``; the hot
path contains no SymPy and is ``O(W·order·params)`` per sample.
"""

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp
from numpy.polynomial import legendre as L
from scipy.stats import chi2

from dtfit._core._kernels import legendre_project
from dtfit.types import InitialGuess


class LSIFilter:
    """Online integral-least-squares (LSI) parameter tracker with drift detection.

    Drop-in sibling of :class:`EACFilter` with the same ``partial_fit`` /
    ``predict`` / ``params_`` API; it swaps the area measurement for an
    orthogonal Legendre-spectrum measurement (streaming LSI).

    As with :class:`EACFilter`, the constructor exposes the full knob set; most
    callers should start from a **preset** classmethod and only pass overrides:
    :meth:`tracking` (responsive auto-sized window) or :meth:`robust`
    (outlier/anomaly-resilient gains).
    """

    @classmethod
    def tracking(cls, expr: str, var: str, **overrides: Any) -> "LSIFilter":
        """Responsive preset: auto-sized window for tracking drifting parameters.

        Equivalent to ``LSIFilter(expr, var, adaptive_window=True, ...)``; any
        keyword in ``overrides`` wins over the preset.
        """
        return cls(expr, var, **{"adaptive_window": True, **overrides})

    @classmethod
    def robust(cls, expr: str, var: str, **overrides: Any) -> "LSIFilter":
        """Outlier-resilient preset: innovation winsorizing, data-set measurement
        noise, gentle drift re-arm. Equivalent to ``LSIFilter(expr, var,
        robust=True, adapt_noise=True, drift_reset="inflate", ...)``; ``overrides``
        win.
        """
        return cls(expr, var, **{
            "robust": True, "adapt_noise": True, "drift_reset": "inflate",
            **overrides,
        })

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        regressors: str | Sequence[str] | None = None,
        p0: InitialGuess = None,
        window_size: int = 50,
        min_window: int | None = None,
        adaptive_window: bool = False,
        window_tol: float = 0.001,
        order: int = 5,
        q_diag: list[float] | None = None,
        r: float = 1.0,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        adapt_r: bool = False,
        adapt_noise: bool = False,
        robust: bool = False,
        huber_c: float = 3.0,
        drift_reset: str = "full",
        drift_inflation: float = 100.0,
    ) -> None:
        """
        Args:
            expr: Model expression, e.g. ``"A * sin(w * t)"``. It may also
                reference **external regressors** (see ``regressors``), e.g.
                ``"c0 + c1*t + S"`` where ``S`` is a measured side-channel -- the
                model is then ``f(t, regressors, params)``.
            var: Main variable name in ``expr``.
            regressors: Optional name(s) of external-regressor channels appearing
                in ``expr`` (everything else free is a parameter). When given, each
                ``partial_fit`` / ``predict`` call must supply the regressor
                value(s) for that sample. The model is still scored by the *same*
                Legendre-spectrum measurement -- only now the model can depend on
                exogenous signals (e.g. an IMU-derived motion basis), not just on
                ``t`` -- so a far richer physical model fuses into the integral
                least-squares update without leaving the filter.
            p0: Initial parameter estimate (defaults to ones).
            window_size: Target (maximum) sliding-window length used for the
                spectral projection. The window **grows** from ``min_window`` up
                to this size as samples arrive, then slides; a larger window
                smooths more (a more rigid estimate), a smaller one is more
                responsive.
            min_window: Smallest window at which the filter starts producing an
                estimate -- the measurement is *accumulative*, so rather than
                idling until ``window_size`` samples have arrived, the filter
                projects whatever is in the (growing) window once it holds at
                least this many points. Defaults to ``order + 2`` (the fewest
                points that admit an order-``order`` Legendre projection), so the
                estimate begins acquiring almost immediately, like a pointwise
                filter, instead of after a full-window dead time. Clamped to
                ``[order + 2, window_size]``.
            adaptive_window: If True, size the window **automatically from the
                data** instead of using a fixed ``window_size``. The window grows
                from ``min_window`` while *more data still moves the estimate* (the
                EWMA of the relative update step exceeds ``window_tol``) and stops
                once the estimate has stabilized, capped at ``window_size`` (now
                the maximum). A model with global parameters keeps shifting as the
                window widens, so it grows a wide window on its own; a
                locally-observable one stabilizes quickly and keeps a short window
                -- no per-model hand-tuning. On a detected drift the window
                collapses back to ``min_window`` and re-grows, so it re-acquires a
                new regime from the freshest samples.
            window_tol: Relative-movement threshold for ``adaptive_window``; the
                window stops growing once successive updates move the estimate by
                less than this (EWMA of ``|Δp|/|p|``, default 0.1%). Smaller grows
                a wider window.
            order: Legendre spectral order; the measurement is the first
                ``order + 1`` Legendre coefficients of the window. More orders
                means richer observability (and a larger measurement vector).
                Clamped so ``order + 1 <= window_size``.
            q_diag: Process-noise variances (per parameter); larger values let a
                parameter drift faster. Defaults to 0.01 each.
            r: Base measurement-noise variance. The per-coefficient variance is
                ``R_j = r * (2j + 1)`` -- the LSI orthonormal weighting, which
                down-weights the noisier high-order coefficients.
            alpha: Significance level for the multivariate NIS sudden-jump test.
            cusum_k: CUSUM slack (reference value) in innovation standard
                deviations, applied to the zeroth-coefficient (mean/area) arm.
                Set to ``inf`` to disable the CUSUM test.
            cusum_h: CUSUM decision threshold in accumulated standard deviations.
            adapt_r: If True, adapt ``r`` online from an EWMA of the normalized
                innovation power (Mehra-style).
            adapt_noise: If True, set the measurement-noise covariance *entirely*
                from the data: ``R_diag = v * diag(proj @ proj.T)`` where ``v`` is an
                online EWMA of the residual variance. The spectral-coefficient noise
                is exactly the per-sample noise pushed through the Legendre
                projection, so this is the statistically correct, self-tuning
                measurement noise -- no hand-set ``r`` (which it overrides). It damps
                the gain (smoother output) when the stream is noisy/anomaly-ridden
                and frees it (responsive) when clean, automatically. Pairs naturally
                with ``robust=True``: the winsorization shields the update direction
                while ``v`` (estimated from the raw residual) still senses the noise.
            robust: If True, gate each Kalman update by the normalized innovation:
                a window whose per-dof Mahalanobis innovation exceeds ``huber_c``
                has its (diagonal) measurement-noise inflated -- shrinking the gain
                -- so an outlier-corrupted window cannot yank the estimate. The
                drift detector still sees the raw spectral innovation, so a genuine
                regime shift is detected and re-armed (the inflated covariance then
                disables the gate during re-adaptation).
            huber_c: Robust gate threshold in innovation standard deviations
                (per degree of freedom); ~3 keeps clean windows unweighted.
            drift_reset: On a detected drift, ``"full"`` resets the covariance to
                its large initial value and clears the window; ``"inflate"``
                instead multiplies the covariance by ``drift_inflation`` and
                keeps the current estimate and window (gentler re-adaptation).
            drift_inflation: Covariance inflation factor for
                ``drift_reset="inflate"``.
        """
        t_sym = sp.Symbol(var)
        if regressors is None:
            self.regressors: list[str] = []
        elif isinstance(regressors, str):
            self.regressors = [regressors]
        else:
            self.regressors = list(regressors)
        reg_syms = [sp.Symbol(r_) for r_ in self.regressors]
        # Bind var + regressor names to plain Symbols so names that clash with a
        # SymPy singleton (S, I, E, N, ...) are still usable as regressors.
        _locals = {var: t_sym, **dict(zip(self.regressors, reg_syms))}
        # sympy's stub omits the (str, locals=dict) overload it accepts at runtime.
        model = sp.sympify(expr, locals=_locals)  # pyright: ignore[reportCallIssue]
        self._has_reg = bool(self.regressors)
        _exclude = {t_sym, *reg_syms}
        self.params = sorted(
            (s for s in model.free_symbols if s not in _exclude), key=str
        )
        n = len(self.params)
        if n == 0:
            raise RuntimeError("Model expression has no free parameters to fit.")

        self.p = np.ones(n) if p0 is None else np.asarray(p0, dtype=float)
        self._p_init = np.eye(n) * 10.0
        self.P = self._p_init.copy()
        self.Q = np.diag(q_diag if q_diag is not None else [0.01] * n)
        self.R0 = float(r)
        self.W = int(window_size)
        self.order = max(1, min(int(order), self.W - 1))
        # Accumulative warm-up: start measuring once the growing window holds at
        # least this many points (the fewest an order-`order` projection admits),
        # rather than idling until the window is full.
        floor = self.order + 2
        self.min_window = (
            min(self.W, floor) if min_window is None
            else int(min(self.W, max(floor, min_window)))
        )
        # Automatic window sizing: when enabled, ``window_size`` is the MAX window
        # (a memory cap); the effective window grows from ``min_window`` while the
        # parameters stay under-identified (max relative covariance > window_tol)
        # and stops once they are pinned -- so a model whose parameters are global
        # (a polynomial's coefficients, a saturating time constant) grows a wide
        # window on its own, while a locally-observable one (an oscillation) keeps
        # a short window. No per-model hand-tuning.
        self.adaptive_window = bool(adaptive_window)
        self.window_tol = float(window_tol)
        self._W_eff = self.min_window           # current effective window (adaptive)
        self._W_ref = 2 * (self.order + 1)       # comfortable measurement size
        self._move_ewma = 1.0                    # EWMA of relative estimate movement

        # Per-coefficient measurement variance: the LSI orthonormal weight.
        j = np.arange(self.order + 1)
        self._R_diag = self.R0 * (2.0 * j + 1.0)
        self._norm = (2.0 * j + 1.0) / 2.0  # standard-Legendre projection scale

        # Empirical spectrum: cached Legendre projection on a window normalized
        # to [-1, 1]. For roughly uniform streaming the in-window sample
        # positions are constant, so the projection is a single cached mat-vec.
        tau = np.linspace(-1.0, 1.0, self.W)
        vander = L.legvander(tau, self.order)          # (W, order+1)
        self._proj = np.linalg.pinv(vander)            # (order+1, W)

        # Model spectrum: Gauss-Legendre quadrature on [-1, 1] (model integrated
        # exactly, as in batch LSI).
        n_quad = max(2 * (self.order + 1), 16)
        self._nodes, self._qw = L.leggauss(n_quad)
        self._legvander_q = L.legvander(self._nodes, self.order)  # (n_quad, order+1)

        # Ratio threshold for the self-normalizing energy test: how many times
        # its running mean the spectral-energy innovation must reach to flag a
        # sudden jump. Mapped from ``alpha`` via a chi-squared on the energy's
        # effective degrees of freedom, with a safety margin for the heavier
        # tails of an EWMA-estimated scale.
        dof = self.order + 1
        self._energy_ratio = 1.6 * chi2.ppf(1 - alpha, df=dof) / dof
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)
        self.adapt_r = bool(adapt_r)
        self.adapt_noise = bool(adapt_noise)
        self._v_est = self.R0          # online EWMA of residual variance (adapt_noise)
        self._vn_lambda = 0.05
        self._robust = bool(robust)
        self._huber_c = float(huber_c)
        self.drift_reset = drift_reset
        self.drift_inflation = float(drift_inflation)
        self._r_scale = 1.0  # adaptive multiplier on R when adapt_r

        # Drift detector state (decimated, non-overlapping windows + warmup),
        # mirroring EACFilter so the two are directly comparable.
        self._ewma_lambda = 0.15
        self._warmup_tests = 5
        # Robust self-calibration: one EWMA scale for the scalar spectral-energy
        # innovation (jump test) and one for the mean coefficient (CUSUM). A
        # single scale per statistic -- rather than six per-coefficient scales --
        # keeps the test from inheriting the heavy tails of a noisy variance
        # estimate, exactly as the scalar EACFilter detector does.
        self._s_scale = 0.0   # EWMA of the spectral-energy innovation S
        self._e0_scale2 = 0.0  # EWMA of the mean-coefficient innovation power
        self._n_full = 0
        self._n_tests = 0
        self._g_hi = 0.0
        self._g_lo = 0.0
        self.n_drifts_ = 0
        self.drift_flag_ = False
        self.last_drift_direction_ = 0

        # Most recent one-step forecast residual (innovation); see
        # EACFilter.last_residual_. NaN until the window first fills.
        self.last_residual_ = float("nan")

        self._t: list[float] = []
        self._y: list[float] = []
        self._rbuf: list[tuple] = []   # external-regressor window (aligned with _t)

        # Compile model and per-parameter derivatives once (off the hot path).
        # Regressor symbols are passed positionally before the parameters.
        self._f: Callable[..., Any] = sp.lambdify(
            [t_sym, *reg_syms, *self.params], model, "numpy"
        )
        self._jac: list[Callable[..., Any]] = [
            sp.lambdify([t_sym, *reg_syms, *self.params], sp.diff(model, p), "numpy")
            for p in self.params
        ]

    @property
    def params_(self) -> dict[str, float]:
        """Current parameter estimate as a ``{name: value}`` mapping."""
        return {str(s): float(v) for s, v in zip(self.params, self.p)}

    def _eval(self, func, t_arr, reg_cols=None):
        """Evaluate a compiled callable on the window, broadcasting scalars.
        ``reg_cols`` supplies the external-regressor columns aligned with ``t_arr``
        (passed positionally before the parameters)."""
        if reg_cols is None:
            v = func(t_arr, *self.p)
        else:
            v = func(t_arr, *reg_cols, *self.p)
        if np.ndim(v) == 0:  # constant model/derivative -> broadcast
            v = np.full_like(t_arr, float(v), dtype=float)
        return np.asarray(v, dtype=float)

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

    def _model_spectrum(self, t0, tn, t_arr=None, reg_cols=None, proj=None):
        """Model Legendre spectrum and its parameter Jacobian over the window.

        Without external regressors the model is integrated *exactly* by
        Gauss-Legendre quadrature (a closed-form ``f(t)``, as in batch LSI). With
        regressors the model can only be evaluated at the in-window sample
        positions (the regressors are *measured* signals, not closed-form), so the
        model spectrum is the **discrete Legendre projection at those samples** --
        the same operator ``proj`` used for the data, keeping data and model on one
        footing for the innovation."""
        if self._has_reg:
            fv = self._eval(self._f, t_arr, reg_cols)
            spec = proj @ fv
            h_mat = np.empty((self.order + 1, len(self.p)))
            for jx in range(len(self.p)):
                dv = self._eval(self._jac[jx], t_arr, reg_cols)
                h_mat[:, jx] = proj @ dv
            return spec, h_mat
        t_quad = t0 + (tn - t0) * (self._nodes + 1.0) / 2.0
        fv = self._eval(self._f, t_quad)
        spec = legendre_project(fv, self._qw, self._legvander_q, self._norm)
        h_mat = np.empty((self.order + 1, len(self.p)))
        for jx in range(len(self.p)):
            dv = self._eval(self._jac[jx], t_quad)
            h_mat[:, jx] = legendre_project(
                dv, self._qw, self._legvander_q, self._norm
            )
        return spec, h_mat

    def partial_fit(self, t_new, y_new, regressors=None) -> "LSIFilter":
        """Ingest one ``(t, y[, regressors])`` sample and update in place.

        ``regressors`` (required iff the model declares external regressors) is a
        ``{name: value}`` mapping or a value sequence ordered like ``regressors``.
        """
        self.drift_flag_ = False
        self._t.append(float(t_new))
        self._y.append(float(y_new))
        if self._has_reg:
            self._rbuf.append(self._reg_tuple(regressors))
        cap = self._W_eff if self.adaptive_window else self.W
        while len(self._t) > cap:
            self._t.pop(0)
            self._y.pop(0)
            if self._has_reg:
                self._rbuf.pop(0)
        k = len(self._t)
        if k < self.min_window:
            return self
        full = k >= cap

        t_arr = np.asarray(self._t)
        y_arr = np.asarray(self._y)
        reg_cols = None
        if self._has_reg:
            rb = np.asarray(self._rbuf, dtype=float)
            reg_cols = [rb[:, c] for c in range(rb.shape[1])]
        t0, tn = float(t_arr[0]), float(t_arr[-1])

        # Robust measurement: winsorize the model residual within the window before
        # projecting. Each sample's residual deviation beyond huber_c robust sigmas
        # (MAD) from the window's MEDIAN residual is clipped, so outlier spikes are
        # de-weighted at the sample level (they can no longer dominate the high-order
        # Legendre coefficients), while the median residual -- holding any genuine
        # sustained shift -- passes through, so drift detection still works.
        y_eff = y_arr
        m_win: np.ndarray | None = None
        resid: np.ndarray | None = None
        if self._robust or self.adapt_noise:
            m_win = self._eval(self._f, t_arr, reg_cols)
            resid = y_arr - m_win
        if self._robust:
            # both were assigned above (the guard includes ``self._robust``)
            assert m_win is not None and resid is not None
            med = float(np.median(resid))
            sigma = 1.4826 * float(np.median(np.abs(resid - med)))
            if sigma > 0.0:
                c = self._huber_c * sigma
                y_eff = m_win + (med + np.clip(resid - med, -c, c))

        # Empirical spectrum vs model spectrum (quadrature). The projection is
        # cached only for the true maximum window; at any other length (a growing
        # warm-up or an adaptive window below the cap) recompute it (cheap).
        if k == self.W:
            proj = self._proj
        else:
            proj = np.linalg.pinv(L.legvander(np.linspace(-1.0, 1.0, k), self.order))
        beta_data = proj @ y_eff
        beta_model, h_mat = self._model_spectrum(t0, tn, t_arr, reg_cols, proj)
        e_vec = beta_data - beta_model

        # Robustness guard: reject a non-finite innovation/Jacobian (an unbounded
        # model can overflow) so one bad sample cannot permanently poison the
        # parameter state with NaNs. Keep the last good estimate.
        if not (np.all(np.isfinite(e_vec)) and np.all(np.isfinite(h_mat))):
            return self

        # One-step forecast residual at the newest sample (pre-update params).
        last_reg = None if reg_cols is None else [c[-1:] for c in reg_cols]
        self.last_residual_ = float(
            y_arr[-1] - self._eval(self._f, t_arr[-1:], last_reg)[0]
        )

        # Drift test on the decimated full-window innovation vector. Only run once
        # the window is full -- a partial (growing) window's spectrum is not yet a
        # calibrated baseline for the change detector.
        if full:
            self._n_full += 1
            if self._n_full % cap == 0 and self._drift_step(e_vec):
                return self  # reset happened; skip the update

        # A smaller window averages fewer samples and so is noisier: inflate the
        # measurement noise (trust it proportionally less) so the gain ramps up
        # smoothly and noisy early windows cannot over-kick the estimate. For a
        # fixed window this damps the W/k growing warm-up; for an adaptive window
        # it damps only until a comfortable measurement size W_ref is reached.
        if self.adapt_noise:
            # Dynamic measurement noise: the spectral-coefficient covariance IS the
            # per-sample noise variance propagated through the projection, so
            # ``R_diag = v * diag(proj @ proj.T)`` with ``v`` an online (EWMA)
            # estimate of the residual variance. This self-tunes the gain to the
            # actual noise level -- damping it (smoother) when the stream is noisy or
            # anomaly-ridden and freeing it (responsive) when clean -- with no
            # hand-set ``r``. The raw (un-winsorized) residual is used on purpose so
            # the estimate also senses the energy anomalies leak past the gate; the
            # slow EWMA keeps individual spikes from jerking it. The smaller-window
            # warm-up needs no extra fudge: a short window's projection rows have
            # larger norm, so ``proj_diag`` already inflates ``R`` on its own.
            assert resid is not None  # assigned above (the guard includes adapt_noise)
            v_now = float(np.mean(resid * resid))
            self._v_est = (1.0 - self._vn_lambda) * self._v_est + self._vn_lambda * v_now
            proj_diag = np.einsum("ij,ij->i", proj, proj)
            r_diag = max(self._v_est, 1e-9) * proj_diag
        else:
            r_diag = self._R_diag * self._r_scale
            if self.adaptive_window:
                if k < self._W_ref:
                    r_diag = r_diag * (self._W_ref / k)
            elif not full:
                r_diag = r_diag * (self.W / k)
        s_mat = h_mat @ self.P @ h_mat.T + np.diag(r_diag)
        try:
            gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            return self

        step = gain @ e_vec
        p_new = self.p + step
        P_new = (np.eye(len(self.p)) - gain @ h_mat) @ self.P + self.Q
        if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(P_new))):
            return self  # reject an ill-conditioned (non-finite) update
        self.p = p_new
        self.P = P_new
        # Calibrate the adaptive noise scale only on full windows, so the
        # non-converged warm-up innovations do not corrupt it.
        if self.adapt_r and not self.adapt_noise and full:
            nis = float(e_vec @ (e_vec / r_diag)) / (self.order + 1)
            self._r_scale = 0.95 * self._r_scale + 0.05 * max(nis, 1e-3)
        # Automatic window sizing: keep widening while *more data still moves the
        # estimate* -- the reliable signal that the window is not yet wide enough
        # (the belief covariance is not: a decaying oscillator stays uncertain yet
        # is accurately identified). A model with global parameters keeps shifting
        # as the window grows (so it widens); a locally-observable one stabilizes
        # quickly (so it stops). EWMA of the relative update step, capped at W.
        if self.adaptive_window and self._W_eff < self.W:
            mv = float(np.max(np.abs(step) / (np.abs(self.p) + 1e-9)))
            self._move_ewma = 0.9 * self._move_ewma + 0.1 * mv
            if self._move_ewma > self.window_tol:
                self._W_eff += 1
        return self

    def _drift_step(self, e_vec: np.ndarray) -> bool:
        """Multivariate NIS (sudden jump) + CUSUM on the mean coefficient
        (sustained drift). Resets and returns True on detection."""
        self._n_tests += 1

        # Two scalar statistics, each standardized against the baseline scale
        # built from *previous* windows (so a fresh jump shows up large rather
        # than inflating its own scale), then folded into that scale:
        #   * S = R-weighted spectral energy of the innovation -- omnidirectional
        #     jump detector across all coefficients (curvature, frequency, ...);
        #   * e0 = mean (area-like) coefficient -- a signed, directional channel
        #     for the CUSUM that flags sustained drift up or down.
        r_diag = self._R_diag * self._r_scale
        s_energy = float(e_vec @ (e_vec / r_diag))
        e0 = float(e_vec[0])
        s_ratio = s_energy / self._s_scale if self._s_scale > 0.0 else 0.0
        z0 = e0 / np.sqrt(self._e0_scale2) if self._e0_scale2 > 0.0 else 0.0

        lam = self._ewma_lambda
        self._s_scale = (1.0 - lam) * self._s_scale + lam * s_energy
        self._e0_scale2 = (1.0 - lam) * self._e0_scale2 + lam * (e0 * e0)

        if self._n_tests <= self._warmup_tests:
            return False  # build baselines and let the filter converge first

        self._g_hi = max(0.0, self._g_hi + z0 - self.cusum_k)
        self._g_lo = max(0.0, self._g_lo - z0 - self.cusum_k)
        nis_drift = s_ratio > self._energy_ratio
        cusum_up = self._g_hi > self.cusum_h
        cusum_down = self._g_lo > self.cusum_h
        if nis_drift or cusum_up or cusum_down:
            self._on_drift(up=cusum_up or (not cusum_down and z0 >= 0))
            return True
        return False

    def _on_drift(self, *, up: bool) -> None:
        """Re-arm the filter after a detected drift so it re-adapts quickly."""
        if self.drift_reset == "inflate":
            self.P = self.P * self.drift_inflation
        else:
            self.P = self._p_init.copy()
            self._t, self._y, self._rbuf = [], [], []
        # On a regime change the wide adaptive window now straddles the change and
        # holds stale old-regime data, so collapse it back to min_window: the
        # filter re-acquires the new regime from the freshest samples and re-grows
        # the window as the new parameters become identified (the new regime
        # re-determines the size, so there is nothing to remember from the old one).
        self._W_eff = self.min_window
        self._move_ewma = 1.0
        self._g_hi = 0.0
        self._g_lo = 0.0
        self._s_scale = 0.0
        self._e0_scale2 = 0.0
        self._n_full = 0
        self._n_tests = 0
        self.n_drifts_ += 1
        self.drift_flag_ = True
        self.last_drift_direction_ = 1 if up else -1

    def inflate(self, factor: float | None = None) -> None:
        """Inflate the parameter covariance so new data dominates -- a public
        hook for an external maneuver/change detector to re-arm the filter for
        fast re-adaptation without discarding the current estimate.

        Args:
            factor: Covariance multiplier; defaults to ``drift_inflation``.
        """
        self.P = self.P * (self.drift_inflation if factor is None else float(factor))

    # Convenience alias matching the recursive-filter naming.
    update = partial_fit

    def predict(self, x: np.ndarray, regressors=None) -> np.ndarray:
        """Evaluate the model at the current parameter estimate.

        With external regressors, ``regressors`` supplies their value(s) at ``x``
        (a ``{name: array-or-scalar}`` mapping broadcast to ``x``'s shape, or an
        ``(len(x), n_reg)`` array)."""
        xa = np.asarray(x, dtype=float)
        if not self._has_reg:
            return self._f(xa, *self.p)
        cols = self._predict_cols(xa, regressors)
        return self._f(xa, *cols, *self.p)
