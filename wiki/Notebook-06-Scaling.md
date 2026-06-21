# Scaling - parallel, batched, partitioned

Per-problem independence makes dtfit embarrassingly parallel. Three tools:

- `fit_many` (+ `FittingProblem`, `BatchFittingResult`) - fan independent fits
  across CPU cores.
- `fit_lsi_batched` / `project_spectra` - many channels' projections in one
  GEMM, on a pluggable backend (`numpy` / `cupy` / `torch`).
- `PartitionedLSI` / `PartitionedEDA` / `PartitionedBatchLSI` - one-pass /
  distributed (map-reduce) estimators for streams too big for memory.


```python
%matplotlib inline
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Fitting at extreme parameter trials can overflow exp() harmlessly; keep the
# guide output clean.
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
rng = np.random.default_rng(0)
```

## `fit_many` - independent fits across cores

Each `FittingProblem` is a self-contained, picklable spec. A failed fit is
captured as `error` rather than aborting the batch. Use `n_jobs=-1` for all cores.


```python
from dtfit import fit_many, FittingProblem

problems = []
for i, b in enumerate([0.4, 0.6, 0.8, 1.0, 1.2]):
    x = np.linspace(0, 3, 200)
    y = (1 + 0.2 * i) * np.exp(b * x) + rng.normal(0, 0.05, x.size)
    problems.append(FittingProblem(
        x=x, y=y, expr="a*exp(b*t)", var="t",
        method="lsi", kwargs={"p0": [1.0, 1.0]}, label=f"ch{i}"))

results = fit_many(problems, n_jobs=1)
for r in results:
    msg = r.error if r.error else f"coeffs={np.round(r.coeffs, 3)}"
    print(f"{r.label}: {msg}")
```

    ch0: coeffs=[1.005 0.398]
    ch1: coeffs=[1.195 0.601]
    ch2: coeffs=[1.404 0.799]
    ch3: coeffs=[1.601 1.   ]
    ch4: coeffs=[1.798 1.2  ]
    

## `project_spectra` / `fit_lsi_batched` - one GEMM for many channels

All channels share a grid `x`; their empirical spectra are computed together,
then each small spectral-match solve runs on the host.


```python
from dtfit import project_spectra, fit_lsi_batched

x = np.linspace(0, 3, 300)
b_true = [0.4, 0.6, 0.8, 1.0]
Y = np.column_stack([np.exp(b * x) + rng.normal(0, 0.03, x.size) for b in b_true])

spectra = project_spectra(x, Y, order=6)            # (B, n_coef), one GEMM
print("spectra shape:", spectra.shape)

fits = fit_lsi_batched(x, Y, "a*exp(b*t)", "t", order=6, p0=[1.0, 1.0])
print("recovered b:", [round(f.params["b"], 3) for f in fits])
print("true b     :", b_true)
```

    spectra shape: (4, 7)
    recovered b: [0.4, 0.6, 0.8, 1.0]
    true b     : [0.4, 0.6, 0.8, 1.0]
    

## `PartitionedLSI` - one pass, fixed memory

Fold chunks of a stream into an additive projection accumulator, then fit once
at the end. Consecutive `update` calls are made *exactly* additive (equal to a
single whole-domain projection).


```python
from dtfit import PartitionedLSI

acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 5), order=6)
for x_chunk in np.array_split(np.linspace(0, 5, 5000), 10):
    y_chunk = 1.3 * np.exp(0.7 * x_chunk) + rng.normal(0, 0.05, x_chunk.size)
    acc.update(x_chunk, y_chunk)

res = acc.fit(p0=[1.0, 1.0])
print("one-pass fit:", {k: round(v, 3) for k, v in res.params.items()},
      f"(n_samples={acc.n_samples})")
```

    one-pass fit: {'a': 1.3, 'b': 0.7} (n_samples=5000)
    

## Map-reduce with `merge`

Workers each accumulate over a shard, then the partials are reduced with
`merge` - the distributed estimator.


```python
def shard_accumulator(x_shard):
    a = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 5), order=6)
    y = 1.3 * np.exp(0.7 * x_shard) + rng.normal(0, 0.05, x_shard.size)
    return a.update(x_shard, y)

shards = np.array_split(np.linspace(0, 5, 5000), 4)
partials = [shard_accumulator(s) for s in shards]      # the "map" (parallelizable)
reduced = partials[0]
for a in partials[1:]:
    reduced = reduced.merge(a)                          # the "reduce"

print("map-reduce fit:", {k: round(v, 3) for k, v in reduced.fit(p0=[1.0, 1.0]).params.items()})
```

    map-reduce fit: {'a': 1.297, 'b': 0.7}
    

The backend is pluggable: `project_spectra(..., backend="auto")` uses a GPU
(`cupy` / `torch`) when one is available, else NumPy. `PartitionedBatchLSI`
combines the multi-channel batching with the one-pass reduce.
