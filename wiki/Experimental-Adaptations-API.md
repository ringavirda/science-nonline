# API: experimental adaptations

Full signatures and usage for the structural adaptations still in trial in
`dtfit-experimental`, plus the array-backend helpers. These are **experimental
APIs and may change** until promoted. For the *ideas* behind them see
[README.md](Experimental); for where they sit in the lineage see
[../guides/lineage-and-variants.md](Guides-Lineage-and-Variants).

> **#3 the overlapping-window ensemble has been promoted** to stable `dtfit` --
> `from dtfit import ensemble_fit, EnsembleResult`. See
> [../methods/ensemble.md](Methods-Ensemble) and
> [../api/fitting.md#ensemble_fit](API-Fitting#ensemble_fit).

```python
from dtfit_experimental import (
    fit_lsi_basis,                 # #2 pluggable orthogonal basis
    fit_joint, JointResult,        # #4 joint shared-parameter multi-channel fit
    boosted_fit, BoostedModel,     # #5 stage-wise residual boosting
    available_backends, resolve_backend, Backend,   # array backends
)
```

- [`fit_lsi_basis`](#fit_lsi_basis) -- #2 pluggable basis
- [`fit_joint`](#fit_joint) / [`JointResult`](#jointresult) -- #4 joint fit
- [`boosted_fit`](#boosted_fit) / [`BoostedModel`](#boostedmodel) -- #5 boosting
- [array backends](#backends) -- `available_backends`, `resolve_backend`, `Backend`

---

<a name="fit_lsi_basis"></a>
## `fit_lsi_basis` -- #2 pluggable orthogonal basis

**What it is.** LSI matches fingerprints on the Legendre basis, which is general
but not always *efficient*: a periodic wiggle needs many high Legendre orders to
express, whereas a **Fourier** basis captures it in a couple of harmonics, and a
pure decay is natural in a **Laguerre** basis. This keeps LSI's exact criterion
(the diagonal-weighted spectral match) but lets you pick the basis to match the
signal -- fewer coefficients, better conditioning.

```python
fit_lsi_basis(data_x, data_y, expr, var, *,
              basis="fourier", order=5, filter_data=True,
              period=None, bounds=None, p0=None) -> FittingResult
```

| arg | default | meaning |
|---|---|---|
| `data_x`, `data_y` | -- | observed samples |
| `expr`, `var` | -- | model expression and main variable |
| `basis` | `"fourier"` | `"legendre"` \| `"chebyshev"` \| `"fourier"` \| `"laguerre"` |
| `order` | `5` | spectral order (number of harmonics `K` for Fourier) |
| `filter_data` | `True` | Savitzky-Golay pre-smoothing before projection |
| `period` | `None` | fundamental period for Fourier (defaults to the domain length) |
| `bounds` | `None` | per-parameter `(min, max)` bounds |
| `p0` | `None` | initial guess |

Returns a [`FittingResult`](API-Types) (coefficients, callable model,
covariance).

```python
from dtfit_experimental import fit_lsi_basis
res = fit_lsi_basis(x, y, "A*sin(w*x + p)", "x", basis="fourier", order=4)
```

---

<a name="fit_joint"></a>
## `fit_joint` -- #4 joint shared-parameter fit

**What it is.** Several channels often share structure -- a common frequency across
x/y/z axes, a common growth rate across regions, a common time constant across a
plant's outputs. Fitting them independently wastes that coupling. `fit_joint`
stacks **all** channels' EAC area equations into one system with **shared**
parameters (estimated jointly from every channel) and **per-channel private**
parameters, solved in one pass. More equations per shared unknown means you
observe it better than any single channel could.

```python
fit_joint(channels, expr, var, shared, *,
          n_windows=6, active_ratio=1.0,
          p0_shared=None, p0_private=None,
          bounds_shared=None, bounds_private=None) -> JointResult
```

| arg | default | meaning |
|---|---|---|
| `channels` | -- | list of `(x, y)` arrays, one per channel |
| `expr`, `var` | -- | model expression and main variable |
| `shared` | -- | names of parameters tied across all channels; the rest are per-channel |
| `n_windows` | `6` | area windows per channel |
| `active_ratio` | `1.0` | leading fraction of each channel used for windows |
| `p0_shared` / `p0_private` | `None` | initial guesses (default ones) |
| `bounds_shared` / `bounds_private` | `None` | per-parameter bounds; **when given, a global search (differential evolution) precedes the local refine** -- needed for a multimodal shared parameter such as a free frequency |

<a name="jointresult"></a>
### `JointResult`

| attribute | meaning |
|---|---|
| `shared` | `{name: value}` shared-parameter estimates |
| `private` | list of `{name: value}` per-channel private estimates |
| `expr`, `var` | the model |
| `predict(channel, x)` | evaluate channel `channel`'s fitted model at `x` |

```python
from dtfit_experimental import fit_joint
jr = fit_joint([(t, ax), (t, ay), (t, az)], "A*sin(w*t + p)", "t",
               shared=["w"])          # one frequency, per-axis amplitude/phase
print(jr.shared["w"], jr.private[0])
```

---

<a name="boosted_fit"></a>
## `boosted_fit` -- #5 stage-wise residual boosting

**What it is.** One parametric form may not capture both a trend and a cycle.
Boosting stages the methods: fit stage 1 to the data, subtract its prediction, fit
stage 2 to the residual, and so on. The composite is the **sum** of the stages --
e.g. an LSI trend plus an EAC-fitted oscillatory residual -- more expressive than
either alone, while each stage stays a cheap, well-posed fit. (It works because
the fingerprint transform is linear, so the fingerprint of a sum is the sum of
fingerprints.)

```python
boosted_fit(data_x, data_y, stages) -> BoostedModel
```

| arg | meaning |
|---|---|
| `data_x`, `data_y` | observed samples |
| `stages` | ordered list of stage specs, each a dict with `expr`, `var`, `method` (`"lsi"`/`"eac"`), plus any extra fitter kwargs (`p0`, `bounds`, ...) |

<a name="boostedmodel"></a>
### `BoostedModel`

| attribute | meaning |
|---|---|
| `stage_models` | the per-stage fitted callables |
| `stage_specs` | per-stage `{expr, var, method, coeffs}` (the fitted record) |
| `predict(x)` | sum of all staged contributions |

```python
from dtfit_experimental import boosted_fit
bm = boosted_fit(x, y, [
    {"expr": "a0 + a1*x", "var": "x", "method": "lsi"},      # trend
    {"expr": "A*sin(w*x + p)", "var": "x", "method": "eac"}, # cycle on the residual
])
y_hat = bm.predict(x)
```

---

<a name="backends"></a>
## Array backends -- `available_backends`, `resolve_backend`, `Backend`

The GEMM-batched projection (the promoted `fit_lsi_batched` / `project_spectra` in
stable `dtfit`) is a single matrix product that runs unchanged on CPU or GPU -- only
*where the arrays live* changes. These helpers expose that choice.

```python
available_backends() -> list[str]
resolve_backend(name="auto", *, dtype="float64") -> Backend
```

- **`available_backends()`** -- the backends usable here: always `"numpy"`, plus
  `"cupy"` / `"torch"` when they import *and* a GPU is present.
- **`resolve_backend(name="auto", *, dtype="float64")`** -- build a `Backend` by
  name; `"auto"` prefers a GPU when present, else NumPy.
- **`Backend`** -- a thin object that only moves arrays on/off a device
  (`asarray` / `to_host`); the projection arithmetic stays generic (`@`, `*`,
  `.T`), which is why GPU support is a backend *choice*, not a rewrite.

```python
from dtfit_experimental import available_backends, resolve_backend
print(available_backends())          # e.g. ['numpy'] or ['numpy', 'torch']
bk = resolve_backend("auto")         # GPU if present, else numpy

from dtfit import fit_lsi_batched
fit_lsi_batched(x, Y, "a*exp(b*x)", "x", backend=bk)   # pass it straight in
```

> The batched projection has low arithmetic intensity (it's a thin GEMM), so a GPU
> pays off only when the data is already resident on the device -- but the code path
> is identical either way. See the big-data domain study in
> [README.md](Experimental#suite).
