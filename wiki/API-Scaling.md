# API: scaling backends

Run the batch methods at scale. The fitting *math* is unchanged -- these are
alternative *execution backends* built on the same additive integral projection.
Pick by your situation ([../guides/choosing-a-method.md Sec.3](Guides-Choosing-a-Method)):

| situation | tool |
|---|---|
| many **independent** fits | [`fit_many`](#fit_many) |
| many **channels on a shared grid** | [`fit_lsi_batched`](#fit_lsi_batched) / [`project_spectra`](#project_spectra) (in `dtfit.scale`) |
| a dataset **too big for memory**, one pass | [`PartitionedLSI`](#partitioned) / [`PartitionedEAC`](#partitioned) |
| **distributed** workers, then combine | the same accumulators via `.merge()` |
| many channels **and** streaming | [`PartitionedBatchLSI`](#partitionedbatch) |

The online (one-sample-at-a-time) counterparts are in
[streaming.md](API-Streaming).

---

<a name="fit_many"></a>
## `fit_many`

```python
fit_many(problems, *, n_jobs=-1, backend="loky", verbose=0) -> list[FittingResult]
```

Fit many **independent** problems in parallel (different series and/or models).

| arg | default | meaning |
|---|---|---|
| `problems` | -- | a sequence of [`FittingProblem`](#fittingproblem) specs |
| `n_jobs` | `-1` | workers (`-1` = all cores; `1` = serial, no pool) |
| `backend` | `"loky"` | `"loky"` (processes), `"threading"` (rides GIL-released kernels), or `"multiprocessing"` |
| `verbose` | `0` | forwarded to `joblib.Parallel` |

Returns [`FittingResult`](API-Types) objects **in input order**, each carrying
`.label` and `.error`. A failed problem has its `error` set (and empty `coeffs`)
rather than aborting the batch.

```python
from dtfit import fit_many, FittingProblem
problems = [FittingProblem(x, y, "a*exp(b*t)", "t", label=name)
            for name, (x, y) in series.items()]
results = fit_many(problems, n_jobs=-1)
for r in results:
    print(r.label, r.error or r.coeffs)
```

<a name="fittingproblem"></a>
### `FittingProblem`

A picklable spec for one fit (a dataclass):

| field | default | meaning |
|---|---|---|
| `x`, `y` | -- | observed samples |
| `expr`, `var` | -- | model and main variable |
| `method` | `"lsi"` | `"lsi"` or `"eac"` |
| `kwargs` | `{}` | method-specific keywords (`p0`, `bounds`, ...) |
| `label` | `None` | tag carried through to the result (channel name, etc.) |

### Return type

`fit_many` (and the batched / partitioned fitters) return plain
[`FittingResult`](API-Types) objects -- batch and single fits share the **same**
type. There is no separate batch type: `FittingResult` is itself picklable (it
drops its lazily-built callable on pickling and rebuilds it from `expr`/`coeffs`
on the caller side), so it survives a process-pool round trip and carries:

- `coeffs`, `expr`, `var`, `cov`, `label`, `error` (set instead of `coeffs` when
  the fit raised).
- `model` -- the fitted callable, rebuilt lazily from `expr`/`coeffs`.
- `predict(x) -> ndarray` -- evaluate the model (broadcasts scalars).

Because they are real `FittingResult`s, the uncertainty helpers (`stderr`,
`confidence_intervals`, prediction bands) are available when a covariance was
produced.

> The old `BatchFittingResult` alias has been **removed**; use `FittingResult`.

---

<a name="project_spectra"></a>
## `project_spectra`

Not top-level -- import from the `scale` submodule:

```python
from dtfit.scale import project_spectra

project_spectra(x, Y, *, order=6, basis="legendre", backend="auto", **basis_kwargs) -> ndarray
```

Empirical spectra of `B` channels sharing grid `x`, in **one GEMM** (matrix
multiply). `Y` is `(n, B)` (a column per channel) or `(n,)` for one channel;
returns `(B, n_coef)` (or `(n_coef,)` for a single channel). `backend` is a name
(`"auto"`/`"numpy"`/`"cupy"`/`"torch"`) or a `Backend`. This is just the
fingerprint extraction -- useful on its own when you want the spectra, not a fit.

<a name="fit_lsi_batched"></a>
## `fit_lsi_batched`

```python
fit_lsi_batched(x, Y, expr, var, *, order=6, basis="legendre",
                backend="auto", p0=None, bounds=None, **basis_kwargs)
    -> FittingResult | list[FittingResult]
```

Fit **one LSI model per channel** of `Y` (shared grid `x`), with all channels'
empirical spectra computed in a single GEMM via `backend`; each channel's small
spectral-match solve then runs on the host. `Y` is `(n, B)` or `(n,)`; returns a
list of [`FittingResult`](API-Types) (or one for a single channel).

```python
from dtfit import fit_lsi_batched
results = fit_lsi_batched(x, Y, "a*exp(b*x)", "x", order=6, backend="auto")
# Y is (n_samples, n_channels); one FittingResult per channel
```

Available backends depend on what's installed (NumPy always; CuPy/Torch if
present) -- just pass `backend="auto"` and it resolves the best installed one. To
enumerate them explicitly, `available_backends` lives in `dtfit._core._backend`
(an internal module -- not part of the stable top-level surface).

---

<a name="partitioned"></a>
## `PartitionedLSI` / `PartitionedEAC`

Streaming / distributed estimators via an **additive** reduce -- fit a dataset too
big for memory in one pass, or fan it across workers and combine.

```python
PartitionedLSI(expr, var, *, domain, order=6, basis="legendre")
PartitionedEAC(expr, var, *, domain, n_windows=8)   # area windows instead of a basis
```

These are **approximate** map-reduce estimators, not bit-exact re-implementations
of the batch fitters:

- `PartitionedEAC` places its `n_windows` windows **uniformly in the domain
  value** (not by sample index as batch [`fit_eac`](API-Fitting#fit_eac)), folds
  each chunk's exact edge-split trapezoid data areas, and matches the model with a
  Simpson quadrature over each window. It therefore recovers batch EAC's
  parameters only **asymptotically** on dense, roughly uniform data -- it does
  **not** return bit-identical parameters.
- `PartitionedLSI`'s reduce is exact relative to its own trapezoid projection, but
  parity with [`fit_lsi`](API-Fitting#fit_lsi) (Legendre least-squares projection
  + Savitzky-Golay pre-filter + robust/auto order selection) is likewise
  **asymptotic**, not bit-exact.

The pattern (identical for both):

```python
from dtfit import PartitionedLSI
acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
for x_chunk, y_chunk in stream:      # one pass, fixed memory
    acc.update(x_chunk, y_chunk)     # fold this chunk's partial integrals in
result = acc.fit(p0=[1.0, 1.0])      # FittingResult
```

| method | meaning |
|---|---|
| `update(x_chunk, y_chunk) -> self` | fold one chunk's partial projection/area integrals into the accumulator. **Feed chunks in domain order** -- consecutive updates are made *exactly* additive by carrying the previous chunk's last sample across the boundary |
| `merge(other) -> self` | associative reduce: combine another accumulator's partial sums (the distributed step). Exact when partitions share boundary samples |
| `spectrum()` (LSI) | the reduced empirical spectrum (whole-domain coefficients) |
| `fit(*, p0=None) -> FittingResult` | solve the spectral / area match against the accumulated sums |

`domain` is the `(min, max)` of the variable over the *whole* stream (needed up
front so every chunk projects onto the same basis).

Every scale fitter now populates `converged` and `x_range` on its returned
[`FittingResult`](API-Types) (`x_range` is the fitted `domain`), so the
extrapolation guard that warns when you `predict` outside the fitted support works
on these results too -- as it does for the batch fitters.

<a name="partitionedbatch"></a>
## `PartitionedBatchLSI`

```python
PartitionedBatchLSI(expr, var, *, domain, n_channels, order=6,
                    basis="legendre", backend="auto", **basis_kwargs)
```

The fused **multi-channel** streaming estimator -- `PartitionedLSI`'s reduce for
`B` channels at once (each chunk is `(n, B)`). `n_channels` (`= B`) is a
**required keyword-only** argument: it sizes the `(B, n_coef)` accumulator up
front, and `update` raises `ValueError` if a chunk's channel count disagrees.
`backend` selects the per-chunk GEMM backend (`"auto"`/`"numpy"`/`"cupy"`/`"torch"`
or a `Backend`; see [`project_spectra`](#project_spectra)).

| method | meaning |
|---|---|
| `update(x_chunk, Y_chunk) -> self` | fold a multi-channel chunk (`Y_chunk` is `(n, B)`; feed chunks in domain order) |
| `merge(other) -> self` | combine accumulators (distributed) |
| `spectra()` | the reduced per-channel spectra, `(B, n_coef)` |
| `fit(*, p0=None, bounds=None)` | solve every channel's spectral match -> `list[FittingResult]`; `bounds` are the per-parameter `(min, max)` pairs forwarded to each channel's bounded solve |

Use it when you have **both** many channels and a stream too big for memory.
