# API: experimental adaptations

Signatures and usage for the adaptations in trial in `dtfit-experimental`.
These are **experimental APIs and may change** until promoted. For the ideas
behind them and what each experiment reads see [Experimental](Experimental);
for where they sit in the lineage see
[lineage and variants](Guides-Lineage-and-Variants).

```python
from dtfit_experimental import (
    FourierBasis, ChebyshevBasis, LaguerreBasis,     # image bases
    EdgeBlockBasis, SegmentBasis,                    # windows with explicit edges
    aggregated_image, fit_aggregated,                # data known as window totals
    fit_aligned, detect_jumps, AlignedFit,           # jumps at unknown epochs
    weak_operators, fit_logistic, fit_michaelis_menten,
    fit_damped_oscillator, fit_lotka_volterra_prey, seed_nlls,   # weak-form rate laws
    LocalTimeFilter, shift_matrix,                   # a tracker in local time
    InformationFilter,                               # information-form fusion
    available_backends, resolve_backend, Backend,    # array backends
)
from dtfit_experimental.streaming import FilterBank, FusedChiSquareDetector
```

- [image bases](#bases) -- `FourierBasis`, `ChebyshevBasis`, `LaguerreBasis`
- [windows with explicit edges](#edges) -- `EdgeBlockBasis`, `SegmentBasis`
- [window totals](#aggregated) -- `aggregated_image`, `fit_aggregated`
- [window alignment](#aligned) -- `fit_aligned`, `detect_jumps`, `AlignedFit`
- [weak-form rate laws](#weak) -- `weak_operators`, the four fitters, `seed_nlls`
- [`LocalTimeFilter`](#localtime) and `shift_matrix`
- [`InformationFilter`](#informationfilter)
- [several streams](#bank) -- `FilterBank`, `FusedChiSquareDetector`
- [array backends](#backends) -- `available_backends`, `resolve_backend`, `Backend`

---

<a name="bases"></a>
## Image bases

```python
FourierBasis(order) -> None
ChebyshevBasis(order) -> None
LaguerreBasis(order, scale=6.0) -> None
```

A basis object is passed to `dtfit.fit` (or `Original.image`) in place of a
basis name. Each exposes `evaluate(u) -> Phi` of shape `(len(u), n_coef)` for
`u` in `[-1, 1]`, `n_coef`, and `to_dict()`.

| class | functions | `n_coef` | for |
|---|---|---|---|
| `FourierBasis(order)` | `1, cos(k*ph), sin(k*ph)` for `k = 1 .. order`, `ph = pi*(u + 1)` | `2*order + 1` | periodic or seasonal records; the fundamental is one cycle across the domain, a shorter period is carried by the higher harmonics |
| `ChebyshevBasis(order)` | `T_0 .. T_order` | `order + 1` | the span of Legendre on a smooth signal, with a Gram on the sample grid about half as ill-conditioned at high order |
| `LaguerreBasis(order, scale=6.0)` | `L_j(t) exp(-t/2)`, `t = (u + 1)/2 * scale` | `order + 1` | decays and relaxations; `scale` sets how many decay constants fill the window |

```python
import numpy as np
import dtfit
from dtfit_experimental import FourierBasis

t = np.linspace(0, 10, 400)
y = 0.3 + 1.5 * np.sin(1.885 * t + 0.4) + np.random.default_rng(0).normal(0, 0.05, t.size)
res = dtfit.fit("c + A*sin(w*t + p)", dtfit.Original(t, y), "t",
                basis=FourierBasis(6), p0=[1.0, 0.0, 0.0, 1.8])   # A, c, p, w
print(res.params)
```

---

<a name="edges"></a>
## Windows with explicit edges

```python
EdgeBlockBasis(edges, keep=None) -> None
SegmentBasis(edges, orders) -> None
```

| arg | meaning |
|---|---|
| `edges` | edges on the unit variable `u`, strictly increasing, from `-1` to `+1` (to 1e-12); `K + 1` edges for `K` windows or segments |
| `keep` | boolean mask of length `K` selecting the windows that become basis functions; `None` keeps all, at least one must be kept |
| `orders` | Legendre degree of each segment, length `K`, each `>= 1` |

`EdgeBlockBasis` is the block basis of `dtfit` with the equal-window
constraint dropped. The windows are half-open `[lo, hi)`, the last one closed
at `+1`; a `u` inside a window that is not kept gives an all-zero row.
`edges_on(domain)` and `windows_on(domain)` return the edges and the kept
windows in data units. `SegmentBasis` puts Legendre polynomials on each
segment and zero outside it, so the Gram is block diagonal; a segment with no
sample in it keeps its columns and leaves them rank deficient. Both raise
`ValueError` on edges that are not strictly increasing from `-1` to `+1` and
on a `keep` or `orders` of the wrong length.

```python
from dtfit_experimental import EdgeBlockBasis

x = np.linspace(0, 10, 1000)
y = 1.0 + 0.3 * x + 0.6 * (x >= 3.7) + np.random.default_rng(1).normal(0, 0.1, x.size)

def stepped(x, a, b, d):
    return a + b * x + d * (x >= 3.7)

edges = np.unique(np.concatenate([np.linspace(0, 10, 17), [3.7]]))   # one edge on the epoch
res = dtfit.fit(stepped, dtfit.Original(x, y),
                basis=EdgeBlockBasis(2 * edges / 10 - 1), p0=[1.0, 0.0, 0.0])
print(res.params)
```

---

<a name="aggregated"></a>
## Data known as window totals

```python
aggregated_image(edges, totals, counts, *, x=None) -> Image
fit_aggregated(model, edges, totals, counts, *, x=None, **fit_kwargs) -> FittingResult
```

| arg | meaning |
|---|---|
| `model` | a SymPy expression string, a `sympy.Expr` or a callable `f(x, *params)`, as `dtfit.fit` takes |
| `edges` | window edges in data units, strictly increasing, length `K + 1` |
| `totals` | sum of the samples per window, length `K`; the total of a zero-count window is ignored |
| `counts` | samples per window, length `K`, non-negative integers; zero-count windows are left out of the basis |
| `x` | the sample positions when known; `None` places `counts[k]` positions at the midpoints of equal sub-intervals of window `k`, which differs from the fit on the true positions at second order in the sub-interval width (about 1 percent at 20 samples per window) |
| `**fit_kwargs` | passed to `dtfit.fit` (`p0`, `bounds`, `var`, `param_names`, `absolute_sigma`); `sigma`, `robust` and `basis="auto"` raise `TypeError` |

`aggregated_image` returns an [`Image`](API-Types) in an `EdgeBlockBasis`
over the windows with a positive count: `S` the totals, `G` the diagonal of
the counts, `n` the sample count. For a continuously integrating sensor pass
quadrature node counts proportional to the window lengths as `counts` and
`totals = means * counts`.

`fit_aggregated` is `dtfit.fit` on that image with the result put on the
window scale: `rss` is the weighted window-sum residual, `n_obs` is the
number of windows with a positive count, and the covariance is rescaled by
`(n - p) / (K - p)` unless `absolute_sigma=True`. `cov` is `None` when
`K == p`. `ValueError` on malformed edges, totals, counts or positions.

```python
from dtfit_experimental import fit_aggregated

xa = np.linspace(0, 4, 480)
ya = 0.5 + 2.0 * np.exp(0.5 * xa) + np.random.default_rng(2).normal(0, 0.05, xa.size)
edges = np.linspace(0, 4, 25)                      # 24 windows of 20 samples
k = np.minimum(np.searchsorted(edges, xa, side="right") - 1, 23)
totals = np.bincount(k, weights=ya, minlength=24)
counts = np.bincount(k, minlength=24)

res = fit_aggregated("a0 + a1*exp(a2*x)", edges, totals, counts, var="x")
print(res.params, res.n_obs)                       # ~ 0.5, 2.0, 0.5 from 24 numbers
```

---

<a name="aligned"></a>
## Jumps at unknown epochs

```python
fit_aligned(model, original, *, n_windows, fine=8, span=8, threshold=5.0,
            n_min=2, self_scale=False, clip=None, max_jumps=None, max_iter=3,
            p0=None, param_names=None, bounds=None, **fit_kwargs) -> AlignedFit
detect_jumps(counts, sums_x, resid_sums, edges, *, span=8, threshold=5.0,
             n_min=2, exclude=(), self_scale=False) -> list[tuple[int, float, float, float]]
```

`fit_aligned` fits `f(x, *params)` plus one level shift `d_j * (x >= e_j)`
per detected epoch. The model is fitted without shifts on
`fine * n_windows` fine windows, the residual window sums go to
`detect_jumps`, the fine windows are merged to about `n_windows` with the
flagged ones left out, and the model with its shifts is refitted in the
resulting `EdgeBlockBasis`; detection repeats on the new residual until it
finds nothing, at most `max_iter` rounds.

| arg | default | meaning |
|---|---|---|
| `model` | -- | a callable `f(x, *params)`; parameter names are introspected unless `param_names` is given |
| `original` | -- | the record; its `domain` sets the window range |
| `n_windows` | -- | coarse windows aimed for, `>= 1` |
| `fine` | `8` | fine windows per coarse window; the epoch resolution is one fine window |
| `span` | `8` | fine windows a side in the detector's local lines, `>= 3` |
| `threshold` | `5.0` | detection level of `abs(z)`, `> 0` |
| `n_min` | `2` | fewest samples a fine window must hold to take part in detection |
| `self_scale` | `False` | divide the statistic by its own robust spread over the record, for coloured noise; never makes a detection easier |
| `clip` | `None` | robust sigma (1.4826 MAD, per fine window) beyond which a sample is left out before anything else |
| `max_jumps` | `None` | cap on the number of epochs |
| `p0`, `bounds` | `None` | for the model's own parameters; the shifts are seeded from the detector and unbounded |

`detect_jumps` is the detector alone: a skip-one local-linear test on the
fine window means of the residual. It returns `(window, epoch, amplitude, z)`
per detection, strongest first, with the `span` windows around each
suppressed; empty when nothing reaches `threshold`.

<a name="alignedfit"></a>
### `AlignedFit`

| attribute | meaning |
|---|---|
| `result` | the [`FittingResult`](API-Types) of the model plus the shifts |
| `epochs` | detected epochs in data units, ascending; empty when nothing was detected |
| `steps` | fitted shift amplitudes, in the order of `epochs` |
| `z` | the detection statistic of each epoch |
| `basis` | the `EdgeBlockBasis` the final fit ran in |
| `n_dropped` | samples left out by `clip` |
| `flagged` | fine-window indices left out of the refit |

Finding no jump is a normal outcome: the epochs are empty and the result is
the plain fit on `n_windows` equal windows.

```python
from dtfit_experimental import fit_aligned

def trend(x, a, b):
    return a + b * x

out = fit_aligned(trend, dtfit.Original(x, y), n_windows=32, p0=[1.0, 0.0])
print(out.epochs, out.steps)                       # ~ [3.7], [0.6]
```

---

<a name="weak"></a>
## Weak-form rate laws

```python
weak_operators(t, n_test=12, order=2) -> (I0, I1, I2)
fit_logistic(t, y, n_test=12, *, method="ols") -> dict
fit_michaelis_menten(t, y, n_test=12, *, method="ols") -> dict
fit_damped_oscillator(t, y, n_test=12, *, method="ols") -> dict
fit_lotka_volterra_prey(t, x, n_test=24, *, method="ols") -> dict
seed_nlls(rhs, y0, t, y, p0, *, names=None, rtol=1e-08, atol=1e-10, max_nfev=400) -> dict
```

`weak_operators` returns three callables on the grid `t` (strictly
increasing, may be non-uniform). Each maps a sampled `g(t)` to its `n_test`
projections against `phi_k(u) = (1 - u**2)**order * P_k(u)`: `I0(g)` is
`g` against `phi`, `I1(g)` the first derivative of `g` against `phi`, `I2(g)`
the second, both read from the data without differentiating it.

| fitter | law | returns |
|---|---|---|
| `fit_logistic` | `y' = r y (1 - y/K)` | `{"r", "K"}` |
| `fit_michaelis_menten` | `y' = -Vm y / (Km + y)` | `{"Vm", "Km"}` |
| `fit_damped_oscillator` | `y'' + 2 zeta omega y' + omega**2 y = 0` | `{"omega", "zeta"}` |
| `fit_lotka_volterra_prey` | Lotka-Volterra from the prey series alone (strictly positive) | `{"alpha", "gamma", "delta"}`; `beta` scales the unobserved predator and is not identifiable |

`seed_nlls` is the solved fit: it integrates `rhs(t, state, *params)` with
`scipy.integrate.solve_ivp` at each trial parameter set and fits the first
state component to `y`, started from `p0` -- the weak estimate. `y0` is held
fixed. `ValueError` on mismatched lengths; `RuntimeError` when the
integration fails or the least squares does not converge.

```python
from dtfit_experimental import fit_logistic, seed_nlls

tt = np.linspace(0, 10, 200)
yl = 10.0 / (1 + 19.0 * np.exp(-0.9 * tt)) + np.random.default_rng(3).normal(0, 0.05, tt.size)
weak = fit_logistic(tt, yl)                        # no start, no ODE solve
print(weak)                                        # ~ r 0.9, K 10

def rhs(t, s, r, K):
    return [r * s[0] * (1 - s[0] / K)]

print(seed_nlls(rhs, [0.5], tt, yl, np.array([weak["r"], weak["K"]]), names=["r", "K"]))
```

---

<a name="localtime"></a>
## `LocalTimeFilter`

```python
LocalTimeFilter(degree=2, *, q_rate=0.1, p0=None, **filter_options) -> None
shift_matrix(degree, d) -> np.ndarray
```

A [`dtfit.ImageFilter`](Methods-Legendre-Filter) on a polynomial trend whose
coefficients refer to the newest sample's time: `c0` is the value, `c1` the
first derivative and `c<j>` is `1/j!` of the `j`-th derivative there.

| arg | default | meaning |
|---|---|---|
| `degree` | `2` | degree of the polynomial, 1 to 9 |
| `q_rate` | `0.1` | process-noise variance added to each coefficient per unit of time between samples; one value or one per coefficient. A gap of `d` adds `q_rate * d` |
| `p0` | `None` | the coefficients at the first sample, default ones |
| `**filter_options` | -- | any other keyword of `dtfit.ImageFilter` (`basis`, `order`, `window_size`, `min_window`, `adaptive_window`, `noise_var`, `robust`, `drift_reset`, ...); `q_diag`, `regressors`, `stream` and `param_names` raise `ValueError` |

| method / attribute | meaning |
|---|---|
| `partial_fit(t_new, y_new)` / `update(...)` | move the origin onto `t_new`, then ingest the sample; `ValueError` on a `t_new` earlier than the previous one; a non-finite sample is skipped with a `RuntimeWarning` |
| `predict(x)`, `predict_cov(x)` | the polynomial and its variance at absolute times `x` |
| `coast(x, order=1)`, `coast_cov(x, order=1)` | dead-reckon from the newest sample to absolute times `x` |
| `params_` | `{name: value}` of the coefficients at `t_ref_` |
| `t_ref_` | time of the newest ingested sample; `None` before the first |
| `inflate(factor=None)`, `rearm()` | the hooks for an external change detector, as on `ImageFilter` |
| `result()` | a batch fit on the current window, coefficients at `t_ref_` |

`shift_matrix(degree, d)` is the `(degree + 1, degree + 1)` upper-triangular
map of a polynomial's coefficients under a shift `d` of its origin:
`T[i, j] = C(j, i) * d**(j - i)`; `T(0)` is the identity and `T(-d)` inverts
`T(d)`.

```python
from dtfit_experimental import LocalTimeFilter

f = LocalTimeFilter(2, q_rate=0.1, window_size=20)
ts = 1.0e4 + 0.2 * np.arange(200)                  # the clock's zero is far away
s = ts - ts[0]
pos = 3.0 + 1.5 * s + 0.05 * s ** 2 + np.random.default_rng(4).normal(0, 0.1, ts.size)
for ti, pi in zip(ts, pos):
    f.partial_fit(ti, pi)
print(f.params_["c1"], f.predict(ts[-1] + 1.0))    # the speed now, the position in 1 s
```

---

<a name="informationfilter"></a>
## `InformationFilter` -- information-form fusion

A recursive *linear*-Gaussian estimator of `theta` in `z = h . theta + noise`
that keeps the information matrix `Y = P^-1` and vector `yv = P^-1 theta`.
Absorbing a measurement is an addition (`Y += H^T R^-1 H`,
`yv += H^T R^-1 z`, no inverse), so independent estimators **fuse by adding
information**, in any order; the readout inverts only the small `n x n`
matrix. It shares no code with `dtfit.ImageFilter`, which runs the covariance
form.

```python
InformationFilter(n_params, *, prior_precision=1e-06, forgetting=1.0) -> None
```

| arg | default | meaning |
|---|---|---|
| `n_params` | -- | state dimension `n` |
| `prior_precision` | `1e-6` | diagonal of the initial information matrix `Y0 = prior_precision * I` (a weak prior; `0` is uninformative and leaves `Y` singular until enough measurements arrive) |
| `forgetting` | `1.0` | exponential forgetting in `(0, 1]` (`1` = none); each step down-weights the accumulated information before adding the new measurement |

| method / attribute | meaning |
|---|---|
| `partial_fit(h, z, r=1.0)` | absorb one measurement with noise variance `r`; `h` is a length-`n` row for scalar `z`, or an `(m, n)` matrix for a vector `z` of length `m` (then `r` is a scalar or length `m`). Returns `self` |
| `fuse(other)` | add another estimator's information into this one **in place**; the shared prior is subtracted once. Reordering reproduces the single-pass estimate to rounding. Both operands must have run with `forgetting == 1`. Returns `self` |
| `theta_` | current estimate, from the solve `Y theta = yv` (alias `p`) |
| `cov_` | parameter covariance `P = Y^-1` (alias `P`) |

```python
from dtfit_experimental import InformationFilter

# A line z = a0 + a1*x streamed as rows h = [1, x], split across two sensors
# and fused into the state one estimator seeing all the data reaches.
xs = np.linspace(0, 1, 200)
z = 0.5 + 2.0 * xs + np.random.default_rng(0).normal(0, 0.05, xs.size)
fa, fb = InformationFilter(2), InformationFilter(2)
for i in range(xs.size):
    (fa if i % 2 == 0 else fb).partial_fit([1.0, xs[i]], z[i], r=0.04)
fused = fa.fuse(fb)
print(np.round(fused.theta_, 3))          # ~ [0.5, 2.0]
```

---

<a name="bank"></a>
## Several streams -- `FilterBank`, `FusedChiSquareDetector`

```python
FilterBank(filters) -> None
FilterBank.from_model(expr, var, n_streams, *, filter_cls=ImageFilter, **kwargs) -> FilterBank
FusedChiSquareDetector(bank, *, alpha=0.0001, inflate=4.0, rearm=False,
                       warmup=None, cooldown=None) -> None
```

`FilterBank` drives K independent streaming filters in lockstep:
`partial_fit(t, y, n_jobs=1)` ingests one sample per stream (`t` shared or
per stream), `run(t_seq, Y, n_jobs=1, track=False, backend="thread")` drives
a whole `(n_steps, K)` block and returns `params`, `n_drifts` and optionally
`track`, `predict(x)`, `params_`, `params_array()`, `drift_flags_`.
`backend="process"` needs a bank built by `from_model`.

`FusedChiSquareDetector` sums each filter's `nis_` into one
`chi2(sum n_coef)` statistic. `update(t, y)` ingests one sample per stream
and returns `True` when the step raises a flag; `statistic_`, `flags_` and
`n_flags_` hold the record.

| arg | default | meaning |
|---|---|---|
| `alpha` | `1e-4` | per-step false-alarm probability; the threshold is `chi2.ppf(1 - alpha, df=sum(n_coef))` |
| `inflate` | `4.0` | covariance re-arm factor applied to every filter on a detection (`<= 1` disables it); ignored when `rearm=True` |
| `rearm` | `False` | on a detection call every filter's `rearm()`, which applies its own `drift_reset` and collapses its adaptive window |
| `warmup` | `None` | steps before detecting, default `3 * window` |
| `cooldown` | `None` | steps to suppress detection after a flag, default one `window` |

The idea and the stable per-filter equivalent are on
[the filter bank page](Methods-Filter-Bank).

```python
from dtfit_experimental.streaming import FilterBank

bank = FilterBank.from_model("A*exp(-b*t)", "t", 3, basis="legendre",
                             order=4, window_size=40, p0=[1.0, 0.5])
det = bank.fused_detector(alpha=1e-4, rearm=True)
tt = np.linspace(0, 8, 400)
Yb = np.column_stack([a * np.exp(-0.4 * tt) for a in (1.0, 0.7, 0.4)])
Yb += np.random.default_rng(5).normal(0, 0.01, Yb.shape)
Yb[250:] *= 1.5                                    # a fault on every axis at step 250
flags = [i for i in range(tt.size) if det.update(tt[i], Yb[i])]
print(flags[:1])
```

---

<a name="backends"></a>
## Array backends -- `available_backends`, `resolve_backend`, `Backend`

The channel GEMM behind `ImageStream(channels=B, backend=...)` in stable
`dtfit` is one matrix product that runs unchanged on CPU or GPU -- only
*where the arrays live* changes. `ImageStream` selects that backend by name
(`"numpy"`, `"cupy"` or `"torch"`); `available_backends()` lists the names
usable here.

```python
available_backends() -> list[str]
resolve_backend(name="auto", *, dtype="float64") -> Backend
```

- **`available_backends()`** -- always `"numpy"`, plus `"cupy"` / `"torch"`
  when they import *and* a GPU is present.
- **`resolve_backend(name="auto", *, dtype="float64")`** -- build a `Backend`
  by name; `"auto"` prefers a GPU when present, else NumPy.
- **`Backend`** -- moves arrays on and off a device (`asarray` / `to_host`);
  the projection arithmetic stays generic (`@`, `*`, `.T`).

```python
from dtfit_experimental import available_backends
print(available_backends())          # e.g. ['numpy'] or ['numpy', 'torch']

from dtfit import ImageStream
xs = np.linspace(0, 4, 300)
Ys = np.column_stack([xs, 2 * xs, xs ** 2])
stream = ImageStream("legendre", 6, domain=(0, 4), channels=Ys.shape[1],
                     backend="numpy")
stream.update(xs, Ys)
images = stream.images()             # one running Image per channel
```

> The channel GEMM has low arithmetic intensity, so a GPU pays off only when
> the data is already resident on the device; the transfer is measured on
> [the big-data page](Domain-Big-Data).
