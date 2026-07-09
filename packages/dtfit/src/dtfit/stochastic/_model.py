"""Stochastic-series adaptations -- using dtfit's deterministic fitters on the
*deterministic functionals* of a random process.

dtfit fits a deterministic ``y = f(t; theta)``; a genuinely random series
(economic / financial data, near a martingale) has no such ``f`` -- fitting a
curve to the path is meaningless (the stable forecaster already guards against
it by falling back to persistence). The way to *use* dtfit on such data is to
point it not at the random **path** but at a deterministic **summary** of the
process, whose functional form is known and happens to be exactly the shapes
dtfit excels at:

* the **autocovariance function** of an ARMA / OU process is a sum of damped
  exponentials / damped cosines  ->  exponential-decay / damped-cosine fit;
* the **spectral density** is rational in ``e^{i w}`` and, for a long-memory
  process, a **power law** ``S(f) ~ c f^{-2d}`` near zero  ->  power-law fit;
* the **aggregated-variance** curve of a self-similar process is a power law
  ``Var(block mean at scale m) ~ c m^{2H-2}``  ->  power-law fit;
* the **conditional mean** (trend + cycle) is a structural curve  ->  the
  stable LSI trend + oscillatory-recipe cycle, leaving the stochastic residual.

So each estimator below recovers a *parameter of a stochastic model* (Hurst /
fractional-integration order, mean-reversion speed, volatility persistence,
cycle period, trend) by feeding a deterministic functional of the data to
:func:`dtfit.fit_lsi` / :func:`dtfit.fit_eac`.

The merged entry point :func:`fit_stochastic` composes these routes behind
significance gates into a single :class:`StochasticModel` that characterizes,
forecasts, and *generates* an arbitrary series; the streaming counterpart is
:class:`~dtfit.stochastic.StochasticFilter`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dtfit._pandas import as_series, capture_index, extend_index, to_1d_array
from dtfit.methods import fit_lsi
from ._estimators import (
    sample_acf, hurst_spectral, ar1_reversion, garch_persistence,
    ar_order, fit_ar,
)
from ._stats import (
    _is_nonstationary, _ols_line, _cycle_strength, _fit_seasonal, _seasonal_design,
)
from ._forecast import (
    _bind_forecaster, _fc_rw, _fc_drift, _fc_meanrev, _fc_trend,
    _make_seasonal_fc, _make_seasonal_fc_anchored, _resolve_forecaster,
)
from ._simulate import (
    _sim_ar1, _sim_long_memory, _sim_garch, make_innovations,
)

__all__ = ["StochasticModel", "fit_stochastic"]


# --------------------------------------------------------------------------- #
# THE MERGED SOLUTION -- one entry point that characterizes an arbitrary
# stochastic series across all the routes above and forecasts it.
# --------------------------------------------------------------------------- #
@dataclass
class StochasticModel:
    """A unified second-order characterization of a stochastic series.

    Produced by :func:`fit_stochastic`. Records which structural components were
    *detected* (each behind a significance gate, so white noise yields none), the
    recovered parameter of each, the primary ``regime`` label for the level
    dynamics, and a :meth:`forecast` that composes the deterministic mean with the
    stochastic mean-reversion and emits regime-appropriate prediction intervals.
    """

    n: int
    level: float
    # deterministic mean (1st order)
    trend_slope: float
    has_trend: bool
    cycle_period: float   # in SAMPLES (index steps of y), whatever the t units
    cycle_amp: float
    has_cycle: bool
    n_harmonics: int
    seasonal: bool
    # stochastic structure (2nd order)
    hurst: float
    has_long_memory: bool
    ar1_phi: float
    has_mean_reversion: bool
    vol_persistence: float
    has_vol_clustering: bool
    sigma: float          # one-step innovation std (level)
    sigma_walk: float     # std of first differences (random-walk scale)
    components: tuple[str, ...]
    regime: str
    forecaster_name: str   # the candidate model chosen by backtest selection
    _forecaster: Callable[[int], np.ndarray] = field(
        repr=False, default=lambda h: np.zeros(h))
    # deterministic mean (trend + multi-harmonic seasonal) as a function of the
    # SAMPLE INDEX 0..n-1 (it maps the index onto the fitted time axis
    # internally) -- captured so :meth:`simulate` can regenerate the structured
    # part of the series. The default (flat zero) is used by the unit-root
    # branch, whose mean is the integrated random walk, not a function of ``t``.
    _mean_fn: Callable[[np.ndarray], np.ndarray] = field(
        repr=False,
        default=lambda t: np.zeros_like(np.asarray(t, dtype=float)))
    # pandas index of the training Series/DataFrame (``None`` for ndarray input),
    # remembered so :meth:`forecast` can label the horizon with the continuing
    # future index ("pandas in -> pandas out"). Never consulted on the ndarray
    # path, so ndarray fits stay bit-identical.
    _index: Any = field(repr=False, default=None)

    def fingerprint(self) -> dict[str, object]:
        """The detected structure as a flat ``{name: value}`` dict (for tables)."""
        return {
            "regime": self.regime,
            "components": ", ".join(self.components),
            "trend slope": self.trend_slope if self.has_trend else float("nan"),
            "cycle period": self.cycle_period if self.has_cycle else float("nan"),
            "Hurst H": self.hurst if self.has_long_memory else float("nan"),
            "AR(1) phi": self.ar1_phi if self.has_mean_reversion else float("nan"),
            "vol persistence": self.vol_persistence if self.has_vol_clustering
            else float("nan"),
            "forecaster": self.forecaster_name,
        }

    def summary(self) -> str:
        lines = [f"StochasticModel  regime={self.regime!r}  n={self.n}",
                 f"  components: {', '.join(self.components)}"]
        if self.has_trend:
            lines.append(f"  trend       slope = {self.trend_slope:.4g}")
        if self.has_cycle:
            kind = "seasonal" if self.seasonal else "cycle"
            lines.append(f"  {kind:<11} period = {self.cycle_period:.4g} "
                         f"({self.n_harmonics} harmonic(s))")
        if self.has_long_memory:
            lines.append(f"  long memory H = {self.hurst:.3f} (d = {self.hurst - 0.5:.3f})")
        if self.has_mean_reversion:
            tau = -1.0 / np.log(self.ar1_phi) if 0 < self.ar1_phi < 1 else np.inf
            lines.append(f"  mean revert phi = {self.ar1_phi:.3f} (tau = {tau:.1f})")
        if self.has_vol_clustering:
            lines.append(f"  volatility  persistence = {self.vol_persistence:.3f}")
        lines.append(f"  innovation sigma = {self.sigma:.4g}")
        lines.append(f"  forecaster: {self.forecaster_name}")
        return "\n".join(lines)

    def forecast(self, h: int, *, return_conf_int: bool = False,
                 alpha: float = 0.05, dist: str = "normal", df: float = 7.0):
        """Forecast ``h`` steps with the **backtest-selected** forecaster
        (:attr:`forecaster_name`). ``h`` is in **samples** (index steps of the
        fitted series), whatever the units of the time axis ``t`` the model was
        fit with. With ``return_conf_int`` also returns
        ``(lower, upper)`` bands whose growth matches that forecaster: bounded for
        mean reversion, ``h^(2H)`` for long memory, ``~sqrt(h)`` for a random
        walk / drift, and ~**constant** for a trend-stationary (trend/seasonal)
        forecast. ``dist="t"`` widens the band with a Student-t (``df``) quantile
        instead of the Gaussian one, for fat-tailed innovations.

        When the model was fit on a pandas ``Series`` the point forecast (and,
        with ``return_conf_int``, the two bands) come back as pandas ``Series``
        labelled by the length-``h`` future index continuing the training index
        (a ``DatetimeIndex`` at its frequency, an integer-like index by its step);
        an ndarray-fit model returns exactly the ndarray / tuple of ndarrays it
        did before."""
        steps = np.arange(1, h + 1, dtype=float)
        point = np.asarray(self._forecaster(h), dtype=float)
        # Future index continuing the training index when the model was fit on a
        # pandas Series (``None`` -> plain ndarray, so ndarray-fit models and a
        # pandas-free env stay bit-identical).
        fidx = extend_index(self._index, h) if self._index is not None else None
        if not return_conf_int:
            return as_series(point, fidx)
        from scipy.stats import norm, t as student_t
        # Key the band growth off the SELECTED forecaster, not detected flags: a
        # trend+seasonal forecast whose residual is stationary must not fan out
        # just because a long-memory *component* was also flagged.
        name = self.forecaster_name
        if name.startswith("mean-reversion") and 0.0 < self.ar1_phi < 1.0:
            var = (self.sigma ** 2) * (1.0 - self.ar1_phi ** (2 * steps)) \
                / (1.0 - self.ar1_phi ** 2)
        elif name.startswith(("random walk", "drift")):
            if self.has_long_memory and 0.5 < self.hurst < 1.0:
                # Long memory forecasts as a random walk, but its h-step
                # forecast-error variance grows as h^(2H) (2H > 1), which the
                # plain sigma^2 * h random-walk band under-covers. Widen it to
                # h^(2H) so the band matches the documented long-memory growth.
                var = (self.sigma_walk ** 2) * steps ** (2.0 * self.hurst)
            else:
                var = (self.sigma_walk ** 2) * steps
        else:
            # Deterministic-mean forecasters (trend / seasonal / trend+seasonal,
            # incl. the "(anchored)" variants): the residual around the fitted
            # mean is (trend-)STATIONARY, so its h-step forecast-error variance is
            # ~constant sigma^2 -- NOT the random-walk sigma^2 * h that fans the
            # bands out and massively over-covers. (Parameter-estimation leverage
            # adds a slow extra growth; it needs the fit covariance and is omitted
            # here -- documented rather than faked with an h-linear term.)
            var = (self.sigma ** 2) * np.ones_like(steps)
        if dist == "t":
            z = float(student_t.ppf(1.0 - alpha / 2.0, df))
        elif dist == "normal":
            z = float(norm.ppf(1.0 - alpha / 2.0))
        else:
            raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
        sd = np.sqrt(np.clip(var, 0.0, None))
        return (as_series(point, fidx), as_series(point - z * sd, fidx),
                as_series(point + z * sd, fidx))

    def simulate(self, n: int | None = None, *, seed: int | None = None,
                 rng: np.random.Generator | None = None,
                 dist: str = "normal", df: float = 7.0) -> np.ndarray:
        """Draw a fresh **realization** of length ``n`` from the fitted model.

        This is what makes :class:`StochasticModel` a *generative* model -- the
        tunable simulator of "typical stochastic data" -- not merely a
        forecaster. It composes the detected deterministic mean (trend +
        multi-harmonic seasonal) with a stochastic residual drawn to match the
        detected **second-order regime**:

        * **unit-root** -> an integrated random walk (drift + innovations,
          optionally with GARCH volatility clustering on the increments);
        * **mean-reverting** -> a stationary AR(1) residual at the recovered phi;
        * **long-memory** -> fractional Gaussian noise at the recovered Hurst H
          (spectral synthesis, ``S(f) ~ |f|^{1-2H}``);
        * **vol-clustering** -> a GARCH(1,1) residual at the recovered
          persistence;
        * **white noise** -> i.i.d. Gaussian innovations.

        Re-fitting the simulated path recovers the same regime and parameters
        (the round-trip test), which is the honest validation that the model is a
        faithful generator of the process it claims to characterize.

        Args:
            n: Length of the realization (defaults to the fitted ``n``).
            seed / rng: Randomness control (``rng`` takes precedence).
            dist / df: Innovation distribution -- ``"normal"`` (default) or a
                unit-variance Student-t (``"t"``, ``df`` degrees of freedom) for a
                fat-tailed generator (financial-style tail risk).
        """
        n = int(self.n if n is None else n)
        if rng is None:
            rng = np.random.default_rng(seed)
        noise = make_innovations(dist, df)
        sig = (self.sigma if np.isfinite(self.sigma) and self.sigma > 0.0
               else (self.sigma_walk if np.isfinite(self.sigma_walk)
                     and self.sigma_walk > 0.0 else 1.0))
        # unit-root level: integrate drift + innovations (the mean is the walk).
        if "unit-root" in self.components:
            if self.has_vol_clustering and np.isfinite(self.vol_persistence):
                steps = _sim_garch(n, self.vol_persistence, sig, rng, noise=noise)
            else:
                steps = noise(rng, n) * sig
            drift = self.trend_slope if np.isfinite(self.trend_slope) else 0.0
            return self.level + np.cumsum(drift + steps)
        # stationary / trend-stationary: deterministic mean + a regime residual.
        t = np.arange(n, dtype=float)
        mean = np.asarray(self._mean_fn(t), dtype=float)
        if mean.shape != t.shape:
            mean = np.full(n, float(self.level))
        # Residual regime in the SAME dominance order the regime label uses
        # (long memory subsumes a short-range AR(1), so it is drawn first -- a
        # long-memory series also trips the mean-reversion flag, but simulating it
        # as a bare AR(1) would erase the long memory and break the round-trip).
        if self.has_long_memory and np.isfinite(self.hurst):
            resid = _sim_long_memory(n, self.hurst, sig, rng, noise=noise)
        elif self.has_mean_reversion and 0.0 < self.ar1_phi < 1.0:
            resid = _sim_ar1(n, self.ar1_phi, sig, rng, noise=noise)
        elif self.has_vol_clustering and np.isfinite(self.vol_persistence):
            resid = _sim_garch(n, self.vol_persistence, sig, rng, noise=noise)
        else:
            resid = noise(rng, n) * sig
        return mean + resid


def _ar_whiten(e: np.ndarray, *, max_order: int = 5) -> np.ndarray:
    """Innovations of an AR(p) fit to ``e`` (order chosen by AIC); the de-meaned
    series if ``p == 0``. Used to distinguish a genuine higher-order AR from long
    memory: an AR(p) is white after AR(p) whitening, long memory is not."""
    p = ar_order(e, max_order=max_order)
    ec = np.asarray(e, dtype=float) - float(np.mean(e))
    if p < 1 or ec.size <= p:
        return ec
    phi = np.asarray(fit_ar(e, order=p)["phi"], dtype=float)
    inn = ec[p:].copy()
    for j in range(1, p + 1):
        inn = inn - phi[j - 1] * ec[p - j:ec.size - j]
    return inn


def fit_stochastic(
    y: np.ndarray,
    t: np.ndarray | None = None,
    *,
    period: float | None = None,
    max_harmonics: int = 4,
    forecaster: object = "auto",
    trend_t: float = 3.0,
    cycle_strength: float = 0.08,
    min_cycles: float = 2.5,
    lm_hurst: float = 0.68,
    mr_phi: float = 0.15,
    vol_persist: float = 0.60,
) -> StochasticModel:
    """**The merged stochastic-series solution.** Characterize an arbitrary series
    across every route at once and return a single coherent model.

    The routes are not run blindly; they are composed in the order the second-order
    theory dictates, each behind a significance gate:

    1. **deterministic mean** -- an LSI trend (kept only if its slope is
       significant, ``|t| > trend_t``) and an LSI cycle (kept only if a genuine
       interior spectral peak carries ``> cycle_strength`` of the power and repeats
       ``>= min_cycles`` times);
    2. **whiten** -- fit an AR(1) to the residual; its innovations are tested for
    3. **long memory** -- the spectral Hurst of the *innovations* (so a
       near-unit-root AR(1), whose innovations are white, is *not* mislabelled as
       long memory; declared when ``H > lm_hurst``);
    4. **mean reversion** -- the AR(1) coefficient, kept when
       ``mr_phi < phi < 0.99`` and the lag-1 ACF is significant;
    5. **volatility clustering** -- the persistence of the ``|residual|`` ACF, kept
       when significant and ``> vol_persist``.

    A series with none of these gates open is reported as ``regime="white noise /
    random walk"`` with no components -- the honest "no structure" verdict.

    **Forecasting** is by backtest model selection: a regime-informed set of
    candidate forecasters (random walk, drift, mean reversion, local-slope trend,
    multi-harmonic seasonal continuation) is rolling-origin backtested and the
    **RMSE-optimal** one is chosen (defaulting to the random walk when nothing
    beats it). The choice is recorded in :attr:`StochasticModel.forecaster_name`.
    A series too short to backtest (``n <= 50``) cannot be selected over: the
    first candidate (the random walk) is used, a :class:`UserWarning` is
    emitted, and the recorded name is suffixed with ``" (short-series
    fallback)"`` so the fallback stays visible on the fitted model.

    **Units.** The seasonal period is detected from the spectrum in **sample
    units** (index steps of ``y``); when a custom time axis ``t`` is supplied
    (seconds, years, any spacing) it is converted internally to ``t`` units via
    the median spacing of ``t`` before the seasonal model is fit, so the fitted
    frequency is correct on any axis. :attr:`StochasticModel.cycle_period` is
    reported in **samples**, and the forecast horizon ``h`` of
    :meth:`StochasticModel.forecast` is likewise in **samples**.

    Args:
        y: The series. ``t`` defaults to ``0..n-1`` (uniform spacing); a custom
            ``t`` only rescales the time axis (see **Units** above) -- detection
            gates and forecasts are unchanged by the units of ``t``.
        period: A seasonal period to use, in **samples** (index steps of ``y``,
            NOT ``t`` units; it is converted internally like the detected one),
            else it is detected from the spectrum.
        max_harmonics: Cap on the Fourier harmonics of the seasonal component
            (the actual count is chosen by BIC).
        forecaster: How to forecast. ``"auto"`` (default) backtest-selects the
            RMSE-optimal candidate; a built-in name (one of :data:`FORECASTERS`)
            forces that model; a callable ``(train, h) -> array`` is used
            directly; a list of names / callables / ``(name, fn)`` pairs is a
            custom candidate set to backtest-select among.
        trend_t, cycle_strength, min_cycles, lm_hurst, mr_phi, vol_persist:
            Detection gates (see above).

    Returns:
        A :class:`StochasticModel` with the detected components, parameters and a
        backtest-selected :meth:`~StochasticModel.forecast`.
    """
    # Remember the training index (a pandas Series/DataFrame carries one) so the
    # forecast can be labelled with the continuing future index; then coerce to a
    # 1-D float ndarray. ``to_1d_array`` is bit-identical to ``np.asarray(...,
    # float)`` for a 1-D ndarray / list, so the ndarray path is unchanged.
    index = capture_index(y)
    y = to_1d_array(y, "y")
    n = y.size
    t = np.arange(n, dtype=float) if t is None else to_1d_array(t, "t")
    # Median sample spacing of the (possibly custom) time axis: periods are
    # detected / supplied in SAMPLE units but the seasonal model is fit over
    # ``t``, so they must be converted to t units (samples * dt) first. dt = 1
    # for the default axis, so the conversion is then the identity.
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1.0
    level = float(y.mean())
    sigma_walk = float(np.std(np.diff(y))) if n > 1 else 0.0
    band = 2.0 / np.sqrt(max(n, 1))

    # -- Stage 0: unit-root gate. An I(1) level (random walk) must be DIFFERENCED;
    # fitting a trend / cycle / long memory to its wandering level is the classic
    # spurious regression. Characterize the increments instead -- which is also
    # where a financial series carries its structure (returns = the increments).
    # EXEMPTION: a strongly cyclical series also has AR roots near the unit circle
    # -- but at the cycle frequency, not at f=0 -- so ADF mis-flags it; a genuine
    # interior spectral peak means the persistence is a *cycle*, not a random
    # walk, so keep it for the stationary branch (which detects the cycle).
    # The exemption is deliberately STRICTER than the general cycle gate: a true
    # cycle repeats many times (peak well away from f=0), whereas a random walk's
    # one-off low-frequency wander also shows a spectral peak. Require >= 5
    # repetitions and a clear peak so FX/RW levels are still differenced.
    if period is not None:
        pre_cyclical = True   # the caller declared a seasonal period -> keep it
    else:
        pre_period, pre_strength = _cycle_strength(y) if n >= 8 else (float("nan"), 0.0)
        pre_cyclical = (pre_strength > 0.12 and np.isfinite(pre_period)
                        and 4 <= pre_period <= n / 5.0)
    if _is_nonstationary(y) and not pre_cyclical:
        w = np.diff(y)
        drift = float(w.mean())
        drift_sig = abs(drift) > 2.0 * (np.std(w) / np.sqrt(max(w.size, 1)))
        vol = float("nan")
        has_vol = False
        try:
            aw = np.abs(w - w.mean())
            if w.size > 2 and float(sample_acf(aw, 1)[1]) > band:
                vol = float(garch_persistence(w, use="abs")["persistence"])
                has_vol = vol > vol_persist
        except Exception as exc:
            warnings.warn(
                f"stochastic stage unit-root vol-clustering failed: {exc}",
                UserWarning, stacklevel=2)
        comps = (["unit-root"] + (["drift"] if drift_sig else [])
                 + (["vol-clustering"] if has_vol else []))
        regime = "random walk + drift" if drift_sig else "random walk"
        # An I(1) level is forecast by persistence or drift -- NOT by reversion to
        # a sample mean, which is meaningless for a non-stationary level (it pulls
        # an interest rate back to an outdated long-run average -- a garbage
        # forecast). The caller can still force any model via ``forecaster=``.
        ur_per = float(period) if period is not None else _cycle_strength(w)[0]
        fname, ffn = _resolve_forecaster(
            forecaster,
            [("random walk", _fc_rw), ("drift", _fc_drift)],
            per=ur_per, max_harmonics=max_harmonics, y=y)
        return StochasticModel(
            n=n, level=level, trend_slope=drift if drift_sig else 0.0,
            has_trend=False,
            cycle_period=float("nan"), cycle_amp=float("nan"), has_cycle=False,
            n_harmonics=0, seasonal=False,
            hurst=float("nan"), has_long_memory=False,
            ar1_phi=float("nan"), has_mean_reversion=False,
            vol_persistence=vol, has_vol_clustering=has_vol,
            sigma=float(np.std(w)), sigma_walk=sigma_walk,
            components=tuple(comps), regime=regime, forecaster_name=fname,
            _forecaster=_bind_forecaster(ffn, y), _index=index,
        )

    # -- Stage 1: deterministic mean (series is stationary or trend-stationary).
    slope, intercept, tstat = _ols_line(t, y)
    resid_lin = y - (intercept + slope * t)
    sst = float((y - y.mean()) @ (y - y.mean())) + 1e-12
    trend_r2 = 1.0 - float(resid_lin @ resid_lin) / sst
    # the trend must be significant AND explain real variance (a persistent but
    # stationary AR(1) can show a spuriously "significant" OLS slope that explains
    # almost no variance -- the r2 gate rejects it).
    has_trend = (tstat > trend_t) and (trend_r2 > 0.10)
    if has_trend:
        rt = fit_lsi(t, y, "a0 + a1*x", "x", k_star=1, filter_data=True)
        a0, a1 = float(rt.coeffs[0]), float(rt.coeffs[1])
        trend = a0 + a1 * t
        slope = a1
        mean_a0, mean_a1 = a0, a1
    else:
        trend = np.full(n, level)
        slope = 0.0
        mean_a0, mean_a1 = level, 0.0
    d1 = y - trend

    # 1b. seasonal / cyclical component -- a MULTI-HARMONIC Fourier model at the
    # detected (or caller-given) period, fit by fast linear least squares. The
    # harmonics capture non-sinusoidal seasonal shapes (the CO2 sawtooth, the
    # sunspot pulse) a single sinusoid cannot, which both reports the cycle right
    # and forecasts it far better; the period is FFT-detected (robust) or supplied.
    if period is not None:
        per, strength = float(period), 1.0
        has_cycle = 4 <= per <= n / 2.0
    else:
        per, strength = _cycle_strength(d1)
        has_cycle = (strength > cycle_strength and np.isfinite(per)
                     and 4 <= per <= n / min_cycles)
    cyc_amp = float("nan")
    n_harm = 0
    seas_r2 = 0.0
    # ``per`` is in SAMPLE units (FFT bins / the documented ``period=`` unit);
    # the seasonal model is fit over the supplied ``t``, so convert to t units
    # first -- otherwise any non-unit spacing fits the wrong frequency.
    per_t = per * dt
    if has_cycle:
        n_harm, s_coef = _fit_seasonal(t, d1, per_t, max_harmonics)
        x_seas = _seasonal_design(t, per_t, n_harm)
        cyc = x_seas @ s_coef
        cyc_amp = float(np.hypot(s_coef[0], s_coef[1]))   # fundamental amplitude
        seas_r2 = 1.0 - float(np.var(d1 - cyc)) / (float(np.var(d1)) + 1e-12)
    else:
        cyc = np.zeros(n)
        s_coef = None
    e = d1 - cyc  # stochastic residual
    # "seasonal" (a strong, clean repeating pattern) vs "cyclical" (a weaker,
    # stochastic cycle): the caller-given period, or a component that explains a
    # good share of the de-trended variance, is treated as seasonal.
    is_seasonal = has_cycle and (period is not None or seas_r2 > 0.30)

    # 2-4. whiten with AR(1), then test long memory on the innovations
    try:
        phi = float(ar1_reversion(e)["phi"])
    except Exception as exc:
        warnings.warn(f"stochastic stage AR(1) failed: {exc}",
                      UserWarning, stacklevel=2)
        phi = 0.0
    acf1 = float(sample_acf(e, 1)[1]) if n > 2 else 0.0
    has_mr = (mr_phi < phi < 0.99) and (abs(acf1) > band)
    innov = e[1:] - phi * e[:-1] if has_mr else (e - e.mean())
    sigma = float(np.std(innov)) if innov.size else float(np.std(e))

    # Long memory: read the Hurst of the RAW residual (its true long-range
    # strength) and gate it with a finite-order-AR **veto**. Testing the Hurst of
    # the AR(1)-whitened innovations -- the old approach -- both under-detects
    # strong long memory (AR(1) whitening removes part of it) and does not stop a
    # higher-order AR(2)/AR(3) whose AR(1) residual still looks long-range. The
    # veto instead whitens with a LOW-order AR (cap 3): a genuine finite-order
    # AR(1..3) -- including a near-unit-root AR(1) -- is driven back to WHITE
    # (Hurst ~0.5), while a hyperbolic-ACF ARFIMA survives it (no finite AR
    # captures a hyperbolic ACF). So the veto cleanly separates a mean-reverting
    # autoregression from true long memory, at any AR order.
    hurst = float("nan")
    has_lm = False
    try:
        hurst = float(hurst_spectral(e - e.mean())["H"])
        has_lm = hurst > lm_hurst
        if has_lm and ar_order(e, max_order=3) >= 1:
            inn_p = _ar_whiten(e, max_order=3)
            veto_h = 0.5 + 0.5 * (lm_hurst - 0.5)  # midpoint 0.5 <-> the LM gate
            if inn_p.size > 128 and float(hurst_spectral(inn_p)["H"]) <= veto_h:
                has_lm = False
                has_mr = True  # a stationary finite-order AR reverts to its mean
    except Exception as exc:
        warnings.warn(f"stochastic stage Hurst/long-memory failed: {exc}",
                      UserWarning, stacklevel=2)

    # 5. volatility clustering on the WHITENED residual: a persistent AR(1) level
    # has trivially autocorrelated |values|, so testing the raw residual gives a
    # false positive; genuine ARCH-type clustering survives whitening.
    vol = float("nan")
    has_vol = False
    try:
        aw = np.abs(innov - innov.mean())
        if innov.size > 2 and float(sample_acf(aw, 1)[1]) > band:
            vol = float(garch_persistence(innov, use="abs")["persistence"])
            has_vol = vol > vol_persist
    except Exception as exc:
        warnings.warn(f"stochastic stage GARCH/vol-clustering failed: {exc}",
                      UserWarning, stacklevel=2)

    comps = []
    if has_trend:
        comps.append("trend")
    if has_cycle:
        comps.append("seasonal" if is_seasonal else "cycle")
    if has_lm:
        comps.append("long-memory")
    if has_mr:
        comps.append("mean-reversion")
    if has_vol:
        comps.append("vol-clustering")
    if has_trend and has_cycle:
        regime = "trend+seasonal" if is_seasonal else "trend+cycle"
    elif has_cycle:
        regime = "seasonal" if is_seasonal else "cyclical"
    elif has_trend:
        regime = "trend"
    elif has_lm:
        regime = "long-memory"
    elif has_mr:
        regime = "mean-reverting"
    elif has_vol:
        regime = "white-mean / vol-clustering"
    else:
        regime = "white noise / random walk"

    # Forecast by RMSE-optimal backtest selection over a REGIME-APPROPRIATE
    # candidate set (vs the random walk fallback). The set is scoped to the
    # detected structure so the chosen forecaster *reflects* it -- a cyclical
    # series picks its seasonal continuation vs RW (it is not handed a
    # mean-reverting level model that would forecast a flat line and erase the
    # cycle), a trending series picks trend/drift, a stationary mean-reverting one
    # picks mean reversion (toward its well-defined mean). The caller overrides via
    # ``forecaster=``.
    # When a deterministic structure (cycle/seasonal/trend) is detected, prefer to
    # forecast WITH it -- show the structure -- unless the backtest finds it
    # clearly worse than RW (a lenient margin, since a per-fold backtest under-
    # rates a phase-sensitive cyclical forecast that is actually RMSE-competitive
    # on the full record). Level regimes stay strict (RMSE-optimal, parsimony->RW).
    candidates: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]]
    sel_margin = 0.98
    if has_cycle:
        seasonal_name = "trend+seasonal" if has_trend else "seasonal"
        # two seasonal forecasters: the unbiased fitted extrapolation (wins on a
        # noisy seasonal series -- no last-value bias) and the anchored one (wins
        # on a clean trend+seasonal series like CO2). The backtest picks per series.
        candidates = [("random walk", _fc_rw),
                      (seasonal_name,
                       _make_seasonal_fc(per, max_harmonics, has_trend)),
                      (seasonal_name + " (anchored)",
                       _make_seasonal_fc_anchored(per, max_harmonics, has_trend))]
        sel_margin = 1.15
    elif has_trend:
        # Pure-trend regime keeps the conservative LINEAR trend candidate: a
        # backtest on a short record (the Nile, n=100) mis-selects a curved fit
        # that over-extrapolates a spurious level-shift "trend". The curvature-
        # aware dtfit extrapolation is used in the SEASONAL path instead (a
        # trend+seasonal series like CO2 has a genuine accelerating trend).
        candidates = [("random walk", _fc_rw), ("drift", _fc_drift),
                      ("trend", _fc_trend)]
        sel_margin = 1.15
    elif has_mr:
        candidates = [("random walk", _fc_rw), ("mean-reversion", _fc_meanrev)]
    else:                                   # long-memory / white noise
        candidates = [("random walk", _fc_rw)]
    fname, ffn = _resolve_forecaster(forecaster, candidates, per=per,
                                     max_harmonics=max_harmonics, y=y,
                                     margin=sel_margin)

    # Deterministic mean over the SAMPLE INDEX (trend + multi-harmonic
    # seasonal), captured for StochasticModel.simulate to regenerate the
    # structured part. The coefficients were fit against the supplied ``t``, so
    # the index is mapped onto that axis (t[0] + dt * index) before evaluating
    # -- for the default axis this is the identity. Bind the fitted
    # coefficients as defaults (no late binding).
    def _mean_fn(tt, _a0=mean_a0, _a1=mean_a1, _hc=bool(has_cycle),
                 _p=float(per_t) if has_cycle else float("nan"),
                 _nh=int(n_harm), _co=s_coef,
                 _t0=float(t[0]) if n else 0.0, _dt=dt):
        tt = _t0 + _dt * np.asarray(tt, dtype=float)
        m = _a0 + _a1 * tt
        if _hc and _co is not None:
            m = m + _seasonal_design(tt, _p, _nh) @ _co
        return m

    return StochasticModel(
        n=n, level=level, trend_slope=float(slope), has_trend=bool(has_trend),
        cycle_period=float(per) if has_cycle else float("nan"),
        cycle_amp=float(cyc_amp), has_cycle=bool(has_cycle),
        n_harmonics=int(n_harm), seasonal=bool(is_seasonal),
        hurst=hurst, has_long_memory=bool(has_lm),
        ar1_phi=float(phi), has_mean_reversion=bool(has_mr),
        vol_persistence=vol, has_vol_clustering=bool(has_vol),
        sigma=float(sigma), sigma_walk=float(sigma_walk),
        components=tuple(comps) if comps else ("none",), regime=regime,
        forecaster_name=fname, _forecaster=_bind_forecaster(ffn, y),
        _mean_fn=_mean_fn, _index=index,
    )
