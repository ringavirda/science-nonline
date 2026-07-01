# API: batch fitting

The core batch fitters and their support functions. All return a
[`FittingResult`](API-Types) unless noted. Conceptual background:
[../guides/methods-explained.md](Guides-Methods-Explained); proofs:
[../methods/](Methods).

- [`fit_lsi`](#fit_lsi) -- Least-Squares Integral (accurate, general default)
- [`fit_eac`](#fit_eac) -- Equal-Areas Criterion (robust, fast); `window_mode="curvature"` for curvature-placed windows
- [`ensemble_fit`](#ensemble_fit) -- overlapping-window robust ensemble (outliers)
- [`fit_dsb`](#fit_dsb) -- Differential Spectra Balance (symbolic reference)
- [`find_degree`](#find_degree) -- polynomial degree selection (DSB support)
- [`fft_frequency_seed`](#fft_frequency_seed) -- frequency seed for oscillatory fits

---

<a name="fit_lsi"></a>
## `fit_lsi`

```python
fit_lsi(data_x, data_y, expr, var, *,
        k_star=None, alpha=0.0, filter_data=True,
        bounds=None, p0=None,
        oscillatory=False, freq_param=None, random_state=0,
        robust=False, huber_c=3.0, nan_policy="raise") -> FittingResult
```

Fit `expr` to `(data_x, data_y)` by integral least-squares in the reconditioned
(Legendre) differential-transformation scheme. The accurate, general-purpose
batch fitter.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `data_x`, `data_y` | array | -- | observed samples (1-D) |
| `expr` | str | -- | model, e.g. `"a0 + a1*exp(a2*x)"` |
| `var` | str | -- | main variable name in `expr` |
| `k_star` | int \| `"auto"` \| None | `None` | number of Legendre spectral coefficients to match; `None` uses Legendre order 5 (auto-raised under the oscillatory recipe); `"auto"` selects it by BIC of the data fit |
| `alpha` | float | `0.0` | extra `exp(-alpha*j)` down-weight on high orders, on top of the built-in `1/(2j+1)`; usually leave at 0 |
| `filter_data` | bool | `True` | apply a Savitzky-Golay pre-filter to `y` |
| `bounds` | list[(lo, hi)] \| None | `None` | per-parameter bounds; **when given, a global search (differential evolution) runs before local refinement** |
| `p0` | array \| None | `None` | initial guess (defaults to ones) |
| `oscillatory` | bool | `False` | apply the oscillatory recipe (smoothing off, order raised to resolve the cycle) |
| `freq_param` | str \| None | `None` | name of the angular-frequency parameter; seeds it from the data's FFT peak and **implies `oscillatory=True`** |
| `random_state` | int \| None | `0` | seed for the deterministic global / differential-evolution search; `None` uses the global RNG |
| `robust` | bool | `False` | robustify the **empirical spectrum** via IRLS: winsorize each sample's residual to the current model (within `huber_c` sigmas) before re-projecting, so an outlier sample cannot distort the Legendre coefficients. Forces `filter_data=False`. The robust-integral lever for LSI |
| `huber_c` | float | `3.0` | winsorization threshold in residual sigmas for `robust=True` |
| `nan_policy` | str | `"raise"` | `"raise"` rejects non-finite samples; `"omit"` drops NaN/inf `(x, y)` pairs before fitting (gappy telemetry) |

**Notes**

- Inputs are validated: `data_x`/`data_y` must be 1-D, equal-length, finite, and
  carry enough samples, or a `ValueError` is raised.
- The empirical spectrum is a Maclaurin-type fit, so LSI needs a **modest dynamic
  range** -- normalize a wide domain (e.g. to `[0, 1.5]`) and scale `y` to O(1)
  first.
- The oscillatory recipe matters: a sinusoid recovers to <1% with it vs ~50%
  without. Pass `freq_param="w"` (or `oscillatory=True`) for any cyclic model.
- Returns a `FittingResult` **with a covariance** (so `stderr`,
  `confidence_intervals`, and prediction bands are available).

**Example**

```python
res = fit_lsi(x, y, "A*sin(w*x + p)", "x", freq_param="w")   # oscillatory recipe
print({k: round(v, 3) for k, v in res.params.items()})
```

---

<a name="fit_eac"></a>
## `fit_eac`

```python
fit_eac(data_x, data_y, expr, var, *,
        active_ratio=0.8, n_windows=None, window_mode="uniform",
        bounds=None, loss="linear", f_scale=None, robust=False,
        huber_c=3.0, p0=None, nan_policy="raise") -> FittingResult
```

Fit `expr` by matching **integral areas** of model and data over windows. The
most noise-robust and fastest batch method; best for few-parameter (2-4)
transient/saturating shapes.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `data_x`, `data_y` | array | -- | observed samples |
| `expr`, `var` | str | -- | model and main variable |
| `active_ratio` | float | `0.8` | leading fraction of the data used for window placement. **Set to `1.0` for a saturating shape** whose asymptote lives in the tail (e.g. `arctan`), or `a` is biased |
| `n_windows` | int \| None | `None` | number of area equations; defaults to `2 x n_params` (overdetermined, for noise averaging). Must be >= `n_params`; clamped so each window keeps >= 3 samples |
| `window_mode` | str | `"uniform"` | window placement: `"uniform"` (default, evenly spaced edges) or `"curvature"` (curvature-adaptive edges -- narrow where the signal bends, wide where it's smooth; best for localized transients/peaks and rational-saturating shapes) |
| `bounds` | (lower, upper) \| None | `None` | parameter bounds (scipy `least_squares` form); switches to a trust-region solver |
| `loss` | str | `"linear"` | least-squares loss; `"soft_l1"`/`"cauchy"`/`"huber"` for outlier robustness |
| `f_scale` | float \| None | `None` | soft margin of the robust `loss`. **Auto-scaled by default**: a quick linear-loss seed fit is run and `f_scale` is set to a robust scale (`1.4826*MAD`) of that fit's window-area residuals, so a robust `loss` actually engages instead of sitting in its quadratic regime. (The historical fixed default `1.0` was far larger than typical window-area residuals and silently disabled the robustness.) Pass an explicit value to override. Ignored when `loss="linear"` |
| `robust` | bool | `False` | robustify the **integrand itself** via IRLS: winsorize each *sample's* residual to the current model (within `huber_c` sigmas) and re-integrate, so an outlier sample cannot distort a window's area. Finer-grained than `loss=` (which down-weights whole window areas) and composes with it -- the "robust integral" lever; no `f_scale` tuning |
| `huber_c` | float | `3.0` | winsorization threshold in residual sigmas for `robust=True` |
| `p0` | array \| None | `None` | initial guess (defaults to ones); a wrong-length `p0` raises a `ValueError` naming the expected parameter order |
| `nan_policy` | str | `"raise"` | `"raise"` rejects non-finite samples; `"omit"` drops NaN/inf `(x, y)` pairs before fitting (gappy sensor/GPS telemetry) |

**Notes**

- Needs at least `2 x n_params` samples.
- Inputs are validated: `data_x`/`data_y` must be 1-D, equal-length, finite, and
  carry enough samples, or a `ValueError` is raised.
- Returns a covariance when the system is overdetermined (the default).
- Because the robust loss is applied to *integrated windows*, it can only
  down-weight whole contaminated windows -- give it **enough windows** that an
  outlier stays localized for the robustness to bite. See the worked discussion in
  [example 02](Example-02-Fitting-Methods).

**Example**

```python
res = fit_eac(x, y, "a*atan(w*x)", "x",
              active_ratio=1.0, n_windows=60, loss="soft_l1", f_scale=0.05)
```

---

<a name="fit_eac_adaptive"></a>
## Curvature-adaptive windows (`fit_eac(..., window_mode="curvature")`)

The former standalone `fit_eac_adaptive` is gone; its behavior is now the
`window_mode="curvature"` path of [`fit_eac`](#fit_eac). Passing
`window_mode="curvature"` places window edges by **curvature** -- narrow where the
signal bends, wide where it's smooth -- so each window carries roughly equal
information. It is the best estimator for localized transients/peaks and
rational-saturating shapes (Michaelis-Menten / Hill / `arctan`).

```python
res = fit_eac(data_x, data_y, expr, var,
              n_windows=None, window_mode="curvature", p0=None)
```

The default `window_mode="uniform"` reproduces the original evenly-spaced EAC
placement. Returns a `FittingResult` with covariance.

---

<a name="ensemble_fit"></a>
## `ensemble_fit`

```python
ensemble_fit(data_x, data_y, expr, var, *,
             method="eac", n_windows=8, overlap=0.5,
             aggregate="median", p0=None, **kwargs) -> EnsembleResult
```

Fit the model on many **overlapping subwindows** and aggregate the per-window
coefficients robustly -- bagging over the time axis. The **median** of the
per-window estimates rejects windows corrupted by outliers, and the inter-window
spread is a cheap empirical uncertainty band.

**Use it for outlier-contaminated data.** The median-of-windows aggregation
rejects whole corrupted windows without the per-problem `f_scale` tuning that
[`fit_eac(loss="soft_l1")`](#fit_eac) needs -- and stays stable where that robust
loss can diverge. On clean (Gaussian-noise) data prefer a single whole-record
fit: the ensemble trades a little accuracy there for the outlier robustness, so
it is a **specialised tool, not the default path**.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `data_x`, `data_y` | array | -- | observed samples |
| `expr`, `var` | str | -- | model and main variable |
| `method` | str | `"eac"` | underlying batch fitter, `"eac"` or `"lsi"` |
| `n_windows` | int | `8` | target number of overlapping subwindows |
| `overlap` | float | `0.5` | fractional overlap between consecutive windows (`0..0.9`) |
| `aggregate` | str | `"median"` | `"median"` (robust) or `"mean"` |
| `p0` | array \| None | `None` | initial guess forwarded to each window fit |
| `**kwargs` | -- | -- | extra args forwarded to the underlying fitter (e.g. `bounds`) |

**Returns** an [`EnsembleResult`](#ensembleresult) -- a [`FittingResult`](API-Types)
(so `params`, `predict`, `stderr`, `to_dict` all work) that additionally carries
the per-window `members` and their `spread` (which also fills the covariance).

**Example**

```python
from dtfit import ensemble_fit

res = ensemble_fit(x, y, "a*exp(-b*x)", "x", method="eac", p0=[1.0, 1.0])
print(res.params, res.spread)   # robust estimate + per-parameter spread
```

<a name="ensembleresult"></a>
### `EnsembleResult`

Subclass of [`FittingResult`](API-Types) returned by `ensemble_fit`. Extra
attributes: `spread` (per-parameter inter-window standard deviation) and
`members` (`(n_windows_fitted, n_params)` raw per-window coefficients). The spread
populates the covariance diagonal, so `stderr()` returns it and
`predict(return_std=True)` reports the ensemble's uncertainty.

---

<a name="fit_dsb"></a>
## `fit_dsb`

```python
fit_dsb(coeffs_poly, expr, var, *, rank=None, p0=None) -> FittingResult
```

Symbolic **reference** method: balances the model's Maclaurin spectrum against a
polynomial's, order by order, and solves symbolically. **Not for noisy production
data** -- use LSI/EAC. Note it takes *polynomial coefficients*, not raw `(x, y)`:
build them with [`find_degree`](#find_degree) + `np.polyfit`, or use
`NonlineRegressor(method="dsb")` which does the pre-fit for you.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `coeffs_poly` | array | -- | polynomial coefficients in **ascending** order (`coeffs_poly[k]` = coefficient of `var**k` = the data's order-`k` Maclaurin coefficient). `np.polyfit` returns descending -- reverse it with `[::-1]` |
| `expr`, `var` | str | -- | model and main variable |
| `rank` | int \| None | `None` | number of balance equations (Maclaurin orders); default uses all available polynomial coefficients |
| `p0` | array \| None | `None` | initial guess for the numeric refinement/fallback |

Raises if the polynomial carries fewer coefficients than the model has parameters
(the balance would be underdefined -- fit a higher-degree polynomial).

**Example**

```python
from dtfit import fit_dsb, find_degree
import numpy as np

deg = find_degree(x, y)              # BIC-selected degree
pc = np.polyfit(x, y, deg)[::-1]     # ascending = the data's Maclaurin spectrum
res = fit_dsb(pc, "a*exp(b*x)", "x")
```

---

<a name="find_degree"></a>
## `find_degree`

```python
find_degree(data_x, data_y, method="bic", max_degree=12) -> int
```

Select a polynomial degree for `(data_x, data_y)` by information criterion (the
DSB pre-fit support primitive). Returns the degree in `0..max_degree` minimizing
`"bic"` (default) or `"aic"` -- a parsimony vs fit trade-off. Warns (via the logger)
if it hits `max_degree`.

---

<a name="fft_frequency_seed"></a>
## `fft_frequency_seed`

```python
fft_frequency_seed(x, y) -> float
```

Dominant **angular** frequency of `y` over uniform grid `x` -- the peak of the
mean-removed real FFT, returned as `2*pi*f`. This is the seed
[`fit_lsi`](#fit_lsi)'s oscillatory recipe uses for `freq_param`; a sinusoid's
frequency can't be recovered without it. Assumes (near-)uniform sampling; the
spacing is read from `x[1] - x[0]`.

```python
from dtfit import fft_frequency_seed
w0 = fft_frequency_seed(x, y)   # ~= angular frequency of the dominant cycle
```

---

### Also exported from `dtfit.methods`

`model_params(f_sym, t)` and `taylor_coeffs(f_sym, t, order)` are the symbolic
helpers the scheme is built on (free-parameter extraction and Maclaurin
coefficients). They're available via `from dtfit.methods import model_params,
taylor_coeffs` for advanced/extension use; most users won't need them.
