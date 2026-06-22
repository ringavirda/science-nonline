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

from typing import Any, Callable

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
    """

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        p0: InitialGuess = None,
        window_size: int = 50,
        order: int = 5,
        q_diag: list[float] | None = None,
        r: float = 1.0,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        adapt_r: bool = False,
        drift_reset: str = "full",
        drift_inflation: float = 100.0,
    ) -> None:
        """
        Args:
            expr: Model expression, e.g. ``"A * sin(w * t)"``.
            var: Main variable name in ``expr``.
            p0: Initial parameter estimate (defaults to ones).
            window_size: Sliding-window length used for the spectral projection.
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
            drift_reset: On a detected drift, ``"full"`` resets the covariance to
                its large initial value and clears the window; ``"inflate"``
                instead multiplies the covariance by ``drift_inflation`` and
                keeps the current estimate and window (gentler re-adaptation).
            drift_inflation: Covariance inflation factor for
                ``drift_reset="inflate"``.
        """
        t_sym = sp.Symbol(var)
        model = sp.sympify(expr)
        self.params = sorted(
            (s for s in model.free_symbols if s != t_sym), key=str
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

        # Compile model and per-parameter derivatives once (off the hot path).
        self._f: Callable[..., Any] = sp.lambdify(
            [t_sym, *self.params], model, "numpy"
        )
        self._jac: list[Callable[..., Any]] = [
            sp.lambdify([t_sym, *self.params], sp.diff(model, p), "numpy")
            for p in self.params
        ]

    @property
    def params_(self) -> dict[str, float]:
        """Current parameter estimate as a ``{name: value}`` mapping."""
        return {str(s): float(v) for s, v in zip(self.params, self.p)}

    def _eval(self, func: Callable[..., Any], t_arr: np.ndarray) -> np.ndarray:
        """Evaluate a compiled callable on ``t_arr``, broadcasting scalars."""
        v = func(t_arr, *self.p)
        if np.ndim(v) == 0:  # constant model/derivative -> broadcast
            v = np.full_like(t_arr, float(v), dtype=float)
        return np.asarray(v, dtype=float)

    def _model_spectrum(self, t0: float, tn: float) -> tuple[np.ndarray, np.ndarray]:
        """Model Legendre spectrum and its parameter Jacobian over ``[t0, tn]``,
        evaluated by Gauss-Legendre quadrature on the mapped nodes."""
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

    def partial_fit(self, t_new: float, y_new: float) -> "LSIFilter":
        """Ingest one ``(t, y)`` sample and update the estimate in place."""
        self.drift_flag_ = False
        self._t.append(float(t_new))
        self._y.append(float(y_new))
        if len(self._t) > self.W:
            self._t.pop(0)
            self._y.pop(0)
        if len(self._t) < self.W:
            return self

        t_arr = np.asarray(self._t)
        y_arr = np.asarray(self._y)
        t0, tn = float(t_arr[0]), float(t_arr[-1])

        # Empirical spectrum (cached projection) vs model spectrum (quadrature).
        beta_data = self._proj @ y_arr
        beta_model, h_mat = self._model_spectrum(t0, tn)
        e_vec = beta_data - beta_model

        # Robustness guard: reject a non-finite innovation/Jacobian (an unbounded
        # model can overflow) so one bad sample cannot permanently poison the
        # parameter state with NaNs. Keep the last good estimate.
        if not (np.all(np.isfinite(e_vec)) and np.all(np.isfinite(h_mat))):
            return self

        # One-step forecast residual at the newest sample (pre-update params).
        self.last_residual_ = float(y_arr[-1] - self._eval(self._f, t_arr[-1:])[0])

        # Drift test on the decimated full-window innovation vector.
        self._n_full += 1
        if self._n_full % self.W == 0 and self._drift_step(e_vec):
            return self  # reset happened; skip the update

        r_diag = self._R_diag * self._r_scale
        s_mat = h_mat @ self.P @ h_mat.T + np.diag(r_diag)
        try:
            gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            return self

        p_new = self.p + gain @ e_vec
        P_new = (np.eye(len(self.p)) - gain @ h_mat) @ self.P + self.Q
        if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(P_new))):
            return self  # reject an ill-conditioned (non-finite) update
        self.p = p_new
        self.P = P_new
        if self.adapt_r:
            nis = float(e_vec @ (e_vec / r_diag)) / (self.order + 1)
            self._r_scale = 0.95 * self._r_scale + 0.05 * max(nis, 1e-3)
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
            self._t, self._y = [], []
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the model at the current parameter estimate."""
        return self._f(np.asarray(x, dtype=float), *self.p)
