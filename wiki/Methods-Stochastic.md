# Stochastic-series characterization -- mathematical reference

dtfit fits a deterministic `y = f(t; theta)`. A genuinely random series (economic
/ financial data, near a martingale) has no such `f` -- fitting a curve to the path
is meaningless. The way to *use* dtfit on it is to point the fitters not at the
random **path** but at a deterministic **functional** of the process, whose form is
known and happens to be exactly the shapes dtfit excels at. Each route then recovers
a *parameter of a stochastic model* from that functional.

API: [api/stochastic.md](API-Stochastic). Validation against ground truth and the
classical estimators: [the stochastic-series domain report](Domain-Stochastic-Series).

---

## The functionals and their fits

For a second-order stationary process, the deterministic objects below have known
closed forms; the named dtfit fitter recovers the parameter.

| process feature | functional | its form | dtfit fit |
|---|---|---|---|
| **mean reversion** (OU / AR(1)) | autocovariance `rho(k)` | `phi^k = exp(-k/tau)` | exponential to the ACF ([`ar1_reversion`](API-Stochastic#estimators)) |
| **stochastic cycle** (AR(2), complex roots) | autocovariance `rho(k)` | `r^k cos(w k + p)` | damped cosine to the ACF, oscillatory recipe ([`cycle_period`](API-Stochastic#estimators)) |
| **volatility clustering** (GARCH(1,1)) | ACF of `\|returns\|` / returns^2 | `~ (alpha+beta)^k` | exponential to that ACF ([`garch_persistence`](API-Stochastic#estimators)) |
| **long memory** (ARFIMA, `d = H - 1/2`) | spectral density near 0 | power law `S(f) ~ c f^{-2d}` | LSI slope of the log-periodogram, GPH ([`hurst_spectral`](API-Stochastic#estimators)) |
| **self-similarity** | aggregated-variance curve | power law `Var(block mean at m) ~ c m^{2H-2}` | power-law fit ([`hurst_aggvar`](API-Stochastic#estimators)) |
| **conditional mean** | trend + cycle | structural curve | LSI trend + LSI cycle, leaving a stochastic residual ([`decompose_trend_cycle`](API-Stochastic#estimators)) |

The autocovariance is itself an *integral* functional (a lagged second moment) and
is computed in `O(n log n)` by the Wiener-Khinchin FFT; the spectrum is the
periodogram. So the work the fitters do here is the same weighted-spectral
([LSI](Methods-LSI)) and area ([EAC](Methods-EAC)) matching used everywhere else --
applied to the functional rather than the path.

---

## The merged solution: gated composition

[`fit_stochastic`](API-Stochastic#fit_stochastic) composes the routes in the order
the second-order theory dictates, each behind a significance gate, so the model
claims only the structure that is really there:

```
   y
   |
 (0) unit-root gate (ADF, ct, AIC)  --I(1)-->  difference; report random walk [+ drift]
   | stationary / trend-stationary
 (1) deterministic mean: LSI trend (|t|>trend_t, R^2 gate)
                         + multi-harmonic Fourier seasonal/cycle (spectral-peak gate)
   | residual
 (2) whiten with AR(1)
   |
 (3) long memory on the INNOVATIONS (spectral Hurst > lm_hurst)
 (4) mean reversion (AR(1) phi)   (5) volatility clustering (whitened |residual|)
```

Two disambiguations matter. Long memory is tested on the **whitened innovations**,
so a near-unit-root AR(1) (whose innovations are white) is not mislabelled as long
memory. Volatility is tested on the **whitened residual**, so a persistent level
(whose `|values|` are trivially autocorrelated) is not a false positive. The
**unit-root gate** is the load-bearing guard -- without it a random walk's wandering
level draws a spurious trend / cycle / long memory (the classic spurious
regression). It is a vendored augmented Dickey-Fuller test (constant+trend
regression, AIC lag selection) reproducing `statsmodels.adfuller` to machine
precision in pure NumPy, with a strict cyclical exemption so a genuine interior
spectral peak (a real cycle, near the unit circle but at `f > 0`) is kept for the
stationary branch instead of being differenced.

---

## Forecasting: backtest model selection

Rather than trusting one structural forecast, `fit_stochastic` rolling-origin
backtests a **regime-informed candidate set** -- random walk, drift, mean reversion,
a curvature-aware LSI trend, and two multi-harmonic seasonal continuations (an
unbiased fitted extrapolation and an anchored one) -- and keeps the RMSE-optimal
one, defaulting to the random walk when nothing beats it. So it beats persistence
wherever some model genuinely can (a drift for GDP, mean reversion for a rate, a
seasonal cycle for CO2) and ties it on a near-martingale, never losing badly. The
seasonal forecast extrapolates the *fitted* trend + seasonal (not the noisy last
value, which would carry that residual forward as a bias) for a noisy series, while
a clean strong-trend series keeps the anchored variant -- the backtest decides which.

The confidence band is keyed off the **selected** forecaster rather than the
detected flags, so its growth matches the point forecast: **bounded** for mean
reversion, `~sqrt(h)` for a random walk, and the long-memory **`h^(2H)`** band
(`2H > 1`) when the long-memory forecaster is chosen -- a wider envelope than the
plain `sigma^2 * h` random-walk band, which would under-cover a long-memory path.

---

## The generative model

[`StochasticModel.simulate`](API-Stochastic#model) makes the characterization a
*generator*: it composes the detected deterministic mean (trend + multi-harmonic
seasonal) with a stochastic residual drawn to match the detected regime -- a
stationary AR(1) for mean reversion, **ARFIMA(0, d, 0)** with `d = H - 1/2` for long
memory, a GARCH(1,1) path for volatility clustering, an integrated walk (with drift)
for a unit root, white noise otherwise. The honest test is the round-trip: fit a
series, simulate from the fitted model, re-fit the simulation -- a faithful generator
recovers its own regime.

---

## The streaming twin

[`StochasticFilter`](API-Stochastic#filter) is the per-input online version of the
second-order stage, built the way dtfit's other streaming filters are -- incremental,
no batch re-fit. It maintains **EWMA autocovariances** of the level and of
`|level - mean|` in `O(K)` per sample (the running ACF), then reads the parameters
in closed form using dtfit's own principles in streaming form:

- **persistence** (AR(1) `phi`) and **volatility persistence** by the **EAC
  equal-areas criterion** -- for an exp-decaying ACF the ratio of two consecutive
  equal-width area windows is `exp(-g h)`, which pins the decay rate `g` (hence the
  persistence `exp(-g)`) amplitude-free; the streaming form of
  `fit_eac("exp(-g*k)")`. Only lags above the white-noise band `~2/sqrt(n_eff)` count
  as signal, so a fast-decay ACF's noisy tail does not trigger the integration;
- the **cycle** from the AR(2) characteristic roots of the running autocovariances.

A two-timescale **fused statistic** (a fast/slow EWMA of the persistence and log
volatility, normalized by a frozen in-control gap variance) flags a structural break
once per change, at a low false-alarm rate -- the streaming counterpart of the
[`FusedChiSquareDetector`](API-Streaming#fused). Memory and per-sample cost are flat
(independent of the stream length), matching the characteristics of
[`EACFilter` / `LSIFilter`](Methods-Equal-Areas-Filter).

One API note: `StochasticFilter.partial_fit(xs)` ingests a **batch** of samples (a
house-style alias for a loop over `update`), unlike the single-sample
`partial_fit(t, y)` of [`EACFilter` / `LSIFilter`](Methods-Equal-Areas-Filter). Use
`update(x)` for the true one-at-a-time path.
