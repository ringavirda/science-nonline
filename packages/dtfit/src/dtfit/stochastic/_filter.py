"""Streaming stochastic characterizer -- the **online twin** of
:func:`fit_stochastic`'s second-order stage, built the way dtfit's own streaming
filters are: **per-input incremental estimation, no batch re-fits.**

Where :func:`~dtfit.stochastic.fit_stochastic` characterizes a whole
record in one batch pass (calling ``dtfit.fit_lsi`` on the deterministic
functionals), many stochastic series arrive as a *stream* and the question is:
*track the second-order structure online, and flag the moment it shifts.* That is
the role dtfit's streaming filters (``EACFilter`` / ``LSIFilter`` +
``FusedChiSquareDetector``) play for deterministic models -- and, crucially, they
do it **incrementally per input**, never by re-running a batch fit each sample.
:class:`StochasticFilter` follows the same design:

* it maintains **exponentially-weighted autocovariances** of the level and of the
  absolute deviations -- updated in O(K) **per input**, the running version of the
  ACF (the deterministic functional ``fit_stochastic`` fits in batch);
* every sample it reads the parameters from those autocovariances in **closed
  form**, using **dtfit's own fitting principles** in their streaming (per-input)
  form rather than a batch ``fit_lsi`` / ``fit_eac`` call:

  - **persistence** (AR(1) ``phi``) and **volatility persistence** by dtfit's
    **equal-areas criterion** (the EAC method): an exp-decaying ACF has, over two
    consecutive equal-width windows, an area ratio ``a2/a1 = exp(-g*h)`` that pins
    the decay rate ``g`` independent of amplitude -- the per-input, closed-form
    form of ``fit_eac("exp(-g*k)")`` (it tracks the batch EAC fit to ~1e-2);
  - the **cycle** from the AR(2) characteristic roots of the running
    autocovariances (the lag-1/lag-2 transform quantities).

  So there is no per-sample optimization, no ``fit_lsi`` call, and no batch-fit
  spike (~3 us/sample, flat memory; matching dtfit's own ``LSIFilter``).

The batch estimators this mirrors call ``dtfit.fit_lsi`` / ``fit_eac`` directly
(the ``_*_from_acf`` cores in :mod:`~dtfit.stochastic`); the filter
is their *streaming* counterpart -- the EAC area criterion evaluated incrementally
-- exactly as ``LSIFilter`` is the per-input counterpart of batch ``fit_lsi``.

A two-timescale fused statistic -- the streaming analogue of the fused chi-square
detector -- flags when the structure breaks (a persistence jump, a volatility
regime switch), once per change, with a low false-alarm rate.

**Honest scope.** This is the online twin of the *second-order* stage only.
Long-memory (the spectral Hurst) and the unit-root gate are inherently *batch*
(a periodogram / a regression over the whole record), so they are NOT tracked
here; the filter covers persistence, cycle and volatility -- the structure that
*can* be maintained in O(1).
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["StochasticFilter"]


class StochasticFilter:
    """Online second-order characterizer + regime-change detector for a stream.

    Feed samples one at a time with :meth:`update`; read the running
    characterization (AR(1) phi, cycle, volatility persistence -- each a closed-
    form per-input estimate from the running autocovariances) from
    :attr:`params_` / :meth:`snapshot`, a short forecast from :meth:`predict`, and
    the count / times of detected structural breaks from :attr:`n_flags_` /
    :attr:`flag_times_` / :attr:`last_flag_`.

    Args:
        nlags: Number of autocovariance lags maintained (the memory footprint).
        halflife: EWMA half-life (samples) of the autocovariance tracker -- how
            fast the characterization adapts.
        warmup: Samples to accumulate before the detector is active.
        settle: Samples after ``warmup`` spent calibrating the detector's
            in-control baseline before it may flag (lets the slow EWMA settle).
        z_thresh: Fused-statistic threshold (in standard deviations) for a flag.
    """

    def __init__(self, nlags: int = 24, halflife: float = 150.0,
                 warmup: int = 80, settle: int = 500, z_thresh: float = 4.0):
        self.nlags = int(nlags)
        self.alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
        self.warmup = int(warmup)
        self.settle = int(settle)
        self.z_thresh = float(z_thresh)
        # online autocovariances of the level and of |level - mean|
        self._buf: deque[float] = deque(maxlen=self.nlags)
        self._mean: float | None = None
        self._acov = np.zeros(self.nlags + 1)
        self._vacov = np.zeros(self.nlags + 1)
        self._n = 0
        # change detection: fast / slow EWMA of the characterizing stats and a
        # frozen-when-out-of-control variance of their gap (in-control-baseline
        # control chart -- so a sustained break cannot inflate its own normalizer).
        self._fast = np.zeros(2)
        self._slow = np.zeros(2)
        self._gvar = np.full(2, 1e-4)
        self._det_init = False
        self._afast = 1.0 - 0.5 ** (1.0 / 40.0)
        self._aslow = 1.0 - 0.5 ** (1.0 / 200.0)
        self._agvar = 1.0 - 0.5 ** (1.0 / 200.0)
        self._alarmed = False
        self.n_flags_ = 0
        # bounded ring of recent change-point times (FLAT memory -- the count
        # n_flags_ is unbounded but is a single int; the times ring is capped).
        self.flag_times_: deque[int] = deque(maxlen=512)
        self.last_flag_: int | None = None

    # -- streaming update (per input, O(K), no batch fit) ------------------- #
    def update(self, x: float) -> "StochasticFilter":
        """Ingest one sample; update the running autocovariances and the detector.
        Everything is incremental -- there is no per-sample optimization or
        batch re-fit (the parameters are read in closed form on demand)."""
        x = float(x)
        a = self.alpha
        mean = x if self._mean is None else self._mean + a * (x - self._mean)
        self._mean = mean
        d = x - mean
        ad = abs(d)
        self._acov[0] += a * (d * d - self._acov[0])
        self._vacov[0] += a * (ad * ad - self._vacov[0])
        # vectorized EWMA autocovariance update over the lag buffer (O(K), no
        # Python-level loop -- the per-sample hot path stays flat and fast)
        m = len(self._buf)
        if m:
            buf = np.fromiter(self._buf, dtype=float, count=m)   # most-recent-first
            dl = buf - self._mean
            self._acov[1:m + 1] += a * (d * dl - self._acov[1:m + 1])
            self._vacov[1:m + 1] += a * (ad * np.abs(dl) - self._vacov[1:m + 1])
        self._buf.appendleft(x)
        self._n += 1
        self._detect()
        return self

    def partial_fit(self, xs) -> "StochasticFilter":
        """Ingest a batch of samples (house-style alias for a loop of update)."""
        for x in np.asarray(xs, dtype=float).ravel():
            self.update(x)
        return self

    # -- per-input closed-form estimators from the running autocov ---------- #
    def _eac_decay(self, acov: np.ndarray) -> float:
        """Decay persistence by dtfit's **equal-areas criterion (EAC)**, evaluated
        per-input on the running ACF: for an exp-decaying ACF the ratio of two
        consecutive equal-width area windows is ``exp(-g*h)``, so the decay rate
        ``g`` (hence persistence ``exp(-g)``) is read off the areas in closed form
        -- the streaming form of ``fit_eac("exp(-g*k)")``, amplitude-free and
        robust to per-lag noise (it integrates rather than reading one lag).

        Only lags clearly above the **white-noise band** (``~2/sqrt(n_eff)``) count
        as signal: a fast-decay (low-persistence) ACF has a noisy tail hovering
        near zero, and a fixed tiny threshold lets that noise spuriously trigger
        the area integration -- the dominant cause of a jittery estimate early in a
        weakly-persistent stream. The integration is used only when there are
        enough signal lags (slow decay); for a fast / moderate decay the lag-1
        autocorrelation is the *exact* AR(1) persistence and less noisy than
        integrating a couple of near-noise lags."""
        c0 = acov[0]
        if c0 <= 1e-12:
            return float("nan")
        rho = acov / c0
        nlags = rho.size - 1
        neff = min(float(self._n), 1.0 / self.alpha)
        band = max(2.0 / np.sqrt(max(neff, 1.0)), 0.05)
        cut = nlags
        for k in range(1, nlags + 1):           # integrate the one-sided decay only
            if rho[k] <= band:
                cut = k
                break
        if cut < 6:                             # too few signal lags -> exact lag-1
            return float(np.clip(rho[1], 0.0, 0.999))
        h = cut // 2
        a1 = float(np.trapezoid(rho[:h + 1]))            # area over [0, h]
        a2 = float(np.trapezoid(rho[h:2 * h + 1]))       # area over [h, 2h]
        if a1 <= 0.0 or a2 <= 0.0 or a2 >= a1:
            return float(np.clip(rho[1], 0.0, 0.999))
        g = -np.log(a2 / a1) / h
        return float(np.clip(np.exp(-abs(g)), 0.0, 0.999))

    def _ar2_yule_walker(self) -> tuple[float, float]:
        """AR(2) coefficients ``(phi1, phi2)`` from the lag-1/lag-2 autocorrelations
        (Yule-Walker). Complex (oscillatory) roots iff ``phi2 < 0`` and the
        discriminant is negative -- that is the streaming cycle test."""
        c0 = self._acov[0]
        if c0 <= 1e-12:
            return 0.0, 0.0
        r1, r2 = self._acov[1] / c0, self._acov[2] / c0
        den = 1.0 - r1 * r1
        if abs(den) < 1e-9:
            return float(np.clip(r1, -0.999, 0.999)), 0.0
        return r1 * (1.0 - r2) / den, (r2 - r1 * r1) / den

    def _cycle_period(self) -> float:
        """Dominant cycle period from the AR(2) complex roots (or NaN if the
        lag-1/lag-2 structure is not oscillatory)."""
        phi1, phi2 = self._ar2_yule_walker()
        if phi2 >= 0.0:
            return float("nan")
        r = np.sqrt(-phi2)               # root modulus
        c = phi1 / (2.0 * r)             # cos(angular frequency)
        if abs(c) >= 1.0:
            return float("nan")
        per = float(2.0 * np.pi / np.arccos(c))
        # Same resolvable-cycle window as snapshot()'s "cyclical" label: at least
        # ~4 samples per period, and at least one full period inside the lag
        # window. Outside it the period is unresolved -> NaN (so params_ and the
        # regime label never disagree).
        return per if 4.0 <= per <= self.nlags else float("nan")

    # -- detector ----------------------------------------------------------- #
    def _rho1(self) -> float:
        c0 = self._acov[0]
        return self._acov[1] / c0 if c0 > 1e-12 else 0.0

    def _detect(self) -> None:
        if self._n < self.warmup:
            return
        # characterizing statistic: lag-1 autocorrelation (persistence) and the
        # log volatility level -- a fused (persistence, volatility) descriptor.
        stat = np.array([self._rho1(), 0.5 * np.log(self._acov[0] + 1e-12)])
        if not self._det_init:
            self._fast[:] = stat
            self._slow[:] = stat
            self._det_init = True
            return
        self._fast += self._afast * (stat - self._fast)
        self._slow += self._aslow * (stat - self._slow)
        gap = self._fast - self._slow
        z2 = float(np.sum(gap ** 2 / (self._gvar + 1e-9)))
        calibrating = self._n < self.warmup + self.settle
        in_control = z2 < (0.5 * self.z_thresh) ** 2
        if calibrating or in_control:
            self._gvar += self._agvar * (gap ** 2 - self._gvar)
        if in_control:
            self._alarmed = False               # re-arm once back in-control
        if calibrating:
            return
        # rising-edge latch -> one flag per change, not one per out-of-control sample
        if z2 > self.z_thresh ** 2 and not self._alarmed:
            self.n_flags_ += 1
            self.flag_times_.append(self._n)
            self.last_flag_ = self._n
            self._alarmed = True

    # -- readout (closed form from the current autocov) --------------------- #
    @property
    def params_(self) -> dict[str, float]:
        """Current second-order characterization -- each a per-input closed-form
        estimate from the running autocovariances (no fit, no cache): persistence
        and volatility by the EAC equal-areas criterion, the cycle by the AR(2)
        roots."""
        return {
            "n": float(self._n),
            "level": float(self._mean) if self._mean is not None else float("nan"),
            "sigma": float(np.sqrt(max(self._acov[0], 0.0))),
            "ar1_phi": self._eac_decay(self._acov),
            "cycle_period": self._cycle_period(),
            "vol_persistence": self._eac_decay(self._vacov),
        }

    def snapshot(self) -> dict[str, object]:
        """:attr:`params_` plus a coarse online ``regime`` label."""
        p = self.params_
        per = p["cycle_period"]
        if p["sigma"] < 1e-9:
            regime = "white noise"
        elif np.isfinite(per):  # _cycle_period already restricts to 4..nlags
            regime = "cyclical"
        elif p["ar1_phi"] > 0.2:
            regime = "mean-reverting"
        elif np.isfinite(p["vol_persistence"]) and p["vol_persistence"] > 0.3:
            regime = "vol-clustering"
        else:
            regime = "white noise"
        return {**p, "regime": regime}

    def predict(self, h: int) -> np.ndarray:
        """Forecast ``h`` steps by AR(1) mean reversion at the current snapshot."""
        p = self.params_
        phi = p["ar1_phi"] if np.isfinite(p["ar1_phi"]) else 0.0
        mu = p["level"]
        last = self._buf[0] if self._buf else mu
        return mu + phi ** np.arange(1, h + 1, dtype=float) * (last - mu)
