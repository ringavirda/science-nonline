"""Recursive equal-areas filter -- the streaming counterpart of EAC.

A Kalman-style recursive estimator whose "measurement" is the area innovation
(experimental minus model area over a sliding window) and whose measurement
Jacobian is the vector of integrated parameter sensitivities. The window may be
split into ``n_sub`` sub-areas to form a *vector* measurement (more independent
equations per step, improving observability of coupled multi-parameter models);
``n_sub=1`` recovers the original single-area update. The measurement-noise
variance ``R`` can be adapted online (``adapt_r``), which also keeps the vector
measurement well-scaled as the sub-area magnitudes shrink.

Concept drift is detected by two complementary tests on the normalized
innovation ``z = e / sqrt(s_cov)``, both of which reset the covariance so the
filter re-adapts quickly:

* a Normalized Innovation Squared (NIS) chi-squared test on ``z**2`` -- catches
  a single, large, sudden shift (symmetric in sign by construction);
* a two-sided CUSUM on ``z`` -- accumulates evidence for a *sustained* drift and
  flags it in **either** direction (upward via the high arm, downward via the
  low arm), which the instantaneous NIS test misses.

The symbolic model and its derivatives are compiled once in ``__init__``; every
``partial_fit`` call is O(window x params) and contains no SymPy -- so the hot
path is real-time safe.
"""

from typing import Any, Callable

import numpy as np
import sympy as sp
from scipy.stats import chi2

from dtfit._core._kernels import simpson_windows, simpson_windows_rows
from dtfit.types import InitialGuess


class EACFilter:
    """Online equal-areas parameter tracker with drift detection."""

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        p0: InitialGuess = None,
        window_size: int = 50,
        q_diag: list[float] | None = None,
        r: float = 1.0,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        n_sub: int = 1,
        adapt_r: bool = False,
        robust: bool = False,
        huber_c: float = 3.0,
        drift_reset: str = "full",
        drift_inflation: float = 100.0,
    ) -> None:
        """
        Args:
            expr: Model expression, e.g. ``"A * sin(w * t)"``.
            var: Main variable name in ``expr``.
            p0: Initial parameter estimate (defaults to ones).
            window_size: Sliding-window length used for area integration.
            q_diag: Process-noise variances (per parameter); larger values let
                a parameter drift faster. Defaults to 0.01 each.
            r: Measurement-noise variance of the area innovation.
            alpha: Significance level for the NIS sudden-jump test (e.g. 0.001).
            cusum_k: CUSUM slack (reference value), in innovation standard
                deviations; the smallest sustained shift to ignore. ~0.5 detects
                a one-sigma drift. Set to ``inf`` to disable the CUSUM test.
            cusum_h: CUSUM decision threshold, in accumulated standard
                deviations; larger means fewer false alarms but slower
                detection.
            n_sub: Number of sub-window area measurements per step. ``1`` (the
                default) is the original single full-window area. Values ``>1``
                split the window into that many sub-areas, giving a *vector*
                measurement -- more independent equations per sample, hence
                better observability and faster convergence (especially for
                multi-parameter models).
            adapt_r: If True, adapt the measurement-noise variance ``R`` online
                from an EWMA of the squared innovation (Mehra-style), instead of
                trusting the fixed ``r``.
            robust: If True, gate each Kalman update by the normalized innovation:
                a window whose per-dof Mahalanobis innovation exceeds ``huber_c``
                has its measurement-noise inflated (gain shrunk) by a Huber weight,
                so an outlier-corrupted window cannot yank the estimate. The drift
                detector still sees the *raw* innovation, so a genuine regime shift
                is detected and re-armed (the inflated covariance then disables the
                gate during re-adaptation) -- transients are rejected, sustained
                changes are not.
            huber_c: Robust gate threshold in innovation standard deviations
                (per degree of freedom); ~3 keeps clean windows unweighted.
            drift_reset: On a detected drift, ``"full"`` resets the covariance to
                its large initial value and clears the window (original
                behaviour); ``"inflate"`` instead multiplies the covariance by
                ``drift_inflation`` and keeps the current estimate and window, a
                gentler re-adaptation that does not discard hard-won parameter
                information.
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
        self.R = float(r)
        self.W = int(window_size)
        self._nis_threshold = chi2.ppf(1 - alpha, df=1)
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)

        self.n_sub = max(1, int(n_sub))
        self.adapt_r = bool(adapt_r)
        self.robust = bool(robust)
        self._huber_c = float(huber_c)
        self.drift_reset = drift_reset
        self.drift_inflation = float(drift_inflation)
        self._r_ewma = float(r)  # adaptive measurement-noise estimate

        # Drift test: the Kalman update runs every sample, but the drift
        # statistic is evaluated only on *non-overlapping* windows (stride W),
        # because consecutive sliding-window area innovations are heavily
        # autocorrelated and would make a CUSUM false-alarm. Each tested
        # innovation is standardized by its own running scale (EWMA of e**2)
        # rather than the theoretical s_cov, which is hard to calibrate for an
        # area measurement -- so the test fires regardless of the innovation's
        # absolute magnitude. The first few tested windows are a warmup that
        # lets the filter converge before the detector is armed.
        self._ewma_lambda = 0.25
        self._warmup_tests = 3
        self._e_scale2 = 0.0
        self._n_full = 0   # full-window steps seen since the last reset
        self._n_tests = 0  # drift evaluations done since the last reset

        # Two-sided CUSUM accumulators (high arm: upward drift; low arm:
        # downward drift) and observable drift state.
        self._g_hi = 0.0
        self._g_lo = 0.0
        self.n_drifts_ = 0
        self.drift_flag_ = False
        self.last_drift_direction_ = 0  # +1 = up, -1 = down, 0 = none yet

        # Most recent one-step forecast residual y_new - f(t_new; p) using the
        # pre-update parameters -- the filter's *innovation*. Exposed so a
        # coordinating layer (e.g. a FilterBank fusing several axes) can build a
        # multi-stream maneuver detector from the per-stream innovations, which
        # carry a sharper transient than the integrated area statistic. NaN until
        # the window first fills.
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

    def _area_jac(self, j: int, t_arr: np.ndarray) -> np.ndarray:
        d = self._jac[j](t_arr, *self.p)
        if np.isscalar(d):
            d = np.full_like(t_arr, d, dtype=float)
        return d

    def partial_fit(self, t_new: float, y_new: float) -> "EACFilter":
        """Ingest one ``(t, y)`` sample and update the estimate in place."""
        self.drift_flag_ = False  # true only on the exact step a drift fires
        self._t.append(float(t_new))
        self._y.append(float(y_new))
        if len(self._t) > self.W:
            self._t.pop(0)
            self._y.pop(0)
        if len(self._t) < self.W:
            return self

        t_arr = np.ascontiguousarray(self._t, dtype=float)
        y_arr = np.ascontiguousarray(self._y, dtype=float)

        # Vector measurement: split the window into n_sub sub-areas. For each
        # the innovation is (data area - model area) and the measurement
        # Jacobian row is the vector of integrated parameter sensitivities. The
        # model and its sensitivities are evaluated once over the window and
        # integrated per sub-area by the (compiled) Simpson kernel.
        np_params = len(self.p)
        m_func = self._f(t_arr, *self.p)
        if np.ndim(m_func) == 0:  # constant model -> broadcast to the window
            m_func = np.full_like(t_arr, float(m_func))
        m_func = np.ascontiguousarray(m_func, dtype=float)
        sub = self._subwindows(t_arr.size)
        starts = np.array([a for a, _ in sub], dtype=np.intp)
        stops = np.array([b for _, b in sub], dtype=np.intp)

        # Robust measurement: winsorize the model residual within the window before
        # integrating. Each sample's residual deviation beyond huber_c robust sigmas
        # (MAD) from the window's MEDIAN residual is clipped, so outlier spikes are
        # de-weighted at the sample level (the integral can no longer carry them),
        # while the median residual -- which holds any genuine sustained shift --
        # passes through untouched, so drift detection still works.
        y_eff = y_arr
        if self.robust:
            resid = y_arr - m_func
            med = float(np.median(resid))
            sigma = 1.4826 * float(np.median(np.abs(resid - med)))
            if sigma > 0.0:
                c = self._huber_c * sigma
                y_eff = m_func + (med + np.clip(resid - med, -c, c))

        e_vec = simpson_windows(y_eff, t_arr, starts, stops) - simpson_windows(
            m_func, t_arr, starts, stops
        )
        jac_rows = np.vstack(
            [self._area_jac(j, t_arr) for j in range(np_params)]
        )
        h_mat = simpson_windows_rows(jac_rows, t_arr, starts, stops).T

        # Robustness guard: an unbounded nonlinear model (e.g. a decaying
        # exponential whose time-constant wanders toward its singular value) can
        # overflow, making the innovation/Jacobian non-finite. Committing that
        # would poison the parameter state permanently -- every later predict()
        # would return NaN. Reject the sample instead and keep the last good
        # estimate; the next clean sample re-adapts.
        if not (np.all(np.isfinite(e_vec)) and np.all(np.isfinite(h_mat))):
            return self

        # One-step forecast residual at the newest sample (pre-update params).
        self.last_residual_ = float(y_arr[-1] - m_func[-1])

        # The drift detector keeps working on the scalar full-window innovation
        # (sum of the additive sub-areas), so its careful calibration is intact.
        e_full = float(e_vec.sum())
        self._n_full += 1
        if self._n_full % self.W == 0 and self._drift_step(e_full):
            return self  # reset happened; skip the update

        r_eff = self._r_ewma if self.adapt_r else self.R
        s_mat = h_mat @ self.P @ h_mat.T + r_eff * np.eye(len(sub))
        try:
            gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            return self

        p_new = self.p + gain @ e_vec
        P_new = (np.eye(np_params) - gain @ h_mat) @ self.P + self.Q
        if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(P_new))):
            return self  # reject an ill-conditioned (non-finite) update
        self.p = p_new
        self.P = P_new
        if self.adapt_r:
            # Mehra-style: track the per-measurement innovation power.
            self._r_ewma = 0.95 * self._r_ewma + 0.05 * float(
                (e_vec @ e_vec) / len(sub)
            )
        return self

    def _subwindows(self, size: int) -> list[tuple[int, int]]:
        """Contiguous ``[start, stop)`` index spans partitioning the window into
        ``n_sub`` sub-areas, each with at least 2 points for integration."""
        n = min(self.n_sub, max(1, size // 2))
        if n == 1:
            return [(0, size)]
        edges = np.linspace(0, size, n + 1).astype(int)
        return [(int(edges[k]), int(edges[k + 1])) for k in range(n)]

    def _drift_step(self, e: float) -> bool:
        """Run NIS + two-sided CUSUM on one decimated innovation. Resets and
        returns True if drift is detected, else returns False."""
        self._n_tests += 1

        # Standardize against the baseline scale established from *previous*
        # windows (not this one), so a sudden jump shows up as a large z instead
        # of inflating its own scale. Then fold this innovation into the scale.
        z = e / np.sqrt(self._e_scale2) if self._e_scale2 > 0.0 else 0.0
        lam = self._ewma_lambda
        self._e_scale2 = (1.0 - lam) * self._e_scale2 + lam * (e * e)

        if self._n_tests <= self._warmup_tests:
            return False  # build the baseline scale and let the filter converge

        # NIS: a single large sudden shift (symmetric via z**2);
        # CUSUM: a sustained shift up (high arm) or down (low arm).
        self._g_hi = max(0.0, self._g_hi + z - self.cusum_k)
        self._g_lo = max(0.0, self._g_lo - z - self.cusum_k)
        nis_drift = z * z > self._nis_threshold
        cusum_up = self._g_hi > self.cusum_h
        cusum_down = self._g_lo > self.cusum_h
        if nis_drift or cusum_up or cusum_down:
            self._on_drift(up=bool(cusum_up or (nis_drift and z > 0)))
            return True
        return False

    def _on_drift(self, *, up: bool) -> None:
        """Re-arm the filter after a detected drift so it re-adapts quickly.

        ``"full"`` discards the covariance and window (fast but throws away the
        parameter estimate's history); ``"inflate"`` keeps the current estimate
        and window but blows up the covariance by ``drift_inflation`` so new data
        dominates -- a gentler re-adaptation.
        """
        if self.drift_reset == "inflate":
            self.P = self.P * self.drift_inflation
        else:
            self.P = self._p_init.copy()
            self._t, self._y = [], []
        self._g_hi = 0.0
        self._g_lo = 0.0
        self._e_scale2 = 0.0
        self._n_full = 0
        self._n_tests = 0
        self.n_drifts_ += 1
        self.drift_flag_ = True
        self.last_drift_direction_ = 1 if up else -1

    def inflate(self, factor: float | None = None) -> None:
        """Inflate the parameter covariance so new data dominates -- a public
        hook for an *external* maneuver/change detector to re-arm the filter for
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
