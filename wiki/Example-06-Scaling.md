# Example 06 Scaling

Scaling out -- parallel, batched, partitioned.

Per-problem independence makes dtfit embarrassingly parallel. Three tools:

- fit_many (+ FittingProblem)  -- fan independent fits across CPU cores.
- fit_lsi_batched / project_spectra -- many channels' projections in one GEMM on
  a pluggable backend (numpy / cupy / torch). project_spectra is the low-level
  primitive and lives under dtfit.scale.
- PartitionedLSI / PartitionedEAC / PartitionedBatchLSI -- one-pass / distributed
  (map-reduce) estimators for streams too big for memory.

Run headless:   python examples/06_scaling.py

Source: [`packages/dtfit/examples/06_scaling.py`](https://github.com/ringavirda/science-pylab/blob/main/packages/dtfit/examples/06_scaling.py)

```python
import numpy as np

from dtfit import fit_many, FittingProblem, fit_lsi_batched, PartitionedLSI
from dtfit.scale import project_spectra


def parallel_fits(rng) -> None:
    # Each FittingProblem is a self-contained, picklable spec. A failed fit is
    # captured as .error rather than aborting the batch (use n_jobs=-1 for all
    # cores). fit_many returns a FittingResult per problem, tagged with .label.
    problems = []
    for i, b in enumerate([0.4, 0.6, 0.8, 1.0, 1.2]):
        x = np.linspace(0, 3, 200)
        y = (1 + 0.2 * i) * np.exp(b * x) + rng.normal(0, 0.05, x.size)
        problems.append(FittingProblem(
            x=x, y=y, expr="a*exp(b*t)", var="t",
            method="lsi", kwargs={"p0": [1.0, 1.0]}, label="ch{}".format(i)))
    print("== fit_many: independent fits ==")
    for r in fit_many(problems, n_jobs=1):
        msg = r.error if r.error else "coeffs={}".format(np.round(r.coeffs, 3))
        print("  {}: {}".format(r.label, msg))


def batched(rng) -> None:
    # All channels share grid x; their empirical spectra are computed together in
    # one GEMM, then each small spectral-match solve runs on the host.
    x = np.linspace(0, 3, 300)
    b_true = [0.4, 0.6, 0.8, 1.0]
    Y = np.column_stack([np.exp(b * x) + rng.normal(0, 0.03, x.size) for b in b_true])
    spectra = project_spectra(x, Y, order=6)           # (B, n_coef), one GEMM
    # Multi-channel Y returns one FittingResult per channel; a single channel
    # would return one result, so normalize to a list.
    fits = fit_lsi_batched(x, Y, "a*exp(b*t)", "t", order=6, p0=[1.0, 1.0])
    fits = fits if isinstance(fits, list) else [fits]
    print("\n== project_spectra / fit_lsi_batched ==")
    print("spectra shape:", spectra.shape)
    print("recovered b  :", [round(f.params["b"], 3) for f in fits])
    print("true b       :", b_true)


def one_pass(rng) -> None:
    # Fold chunks of a stream into an additive projection accumulator, then fit
    # once. Consecutive update() calls are made exactly additive (equal to a
    # single whole-domain projection).
    acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 5), order=6)
    for x_chunk in np.array_split(np.linspace(0, 5, 5000), 10):
        y_chunk = 1.3 * np.exp(0.7 * x_chunk) + rng.normal(0, 0.05, x_chunk.size)
        acc.update(x_chunk, y_chunk)
    res = acc.fit(p0=[1.0, 1.0])
    print("\n== PartitionedLSI: one pass, fixed memory ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()},
          " n_samples:", acc.n_samples)


def map_reduce(rng) -> None:
    # Workers each accumulate over a shard, then the partials are reduced with
    # merge() -- the distributed estimator.
    def shard(x_shard):
        a = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 5), order=6)
        y = 1.3 * np.exp(0.7 * x_shard) + rng.normal(0, 0.05, x_shard.size)
        return a.update(x_shard, y)

    shards = np.array_split(np.linspace(0, 5, 5000), 4)
    partials = [shard(s) for s in shards]              # the "map" (parallelizable)
    reduced = partials[0]
    for a in partials[1:]:
        reduced = reduced.merge(a)                      # the "reduce"
    res = reduced.fit(p0=[1.0, 1.0])
    print("\n== map-reduce with merge() ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()})


def main() -> None:
    rng = np.random.default_rng(0)
    parallel_fits(rng)
    batched(rng)
    one_pass(rng)
    map_reduce(rng)


if __name__ == "__main__":
    main()
```

## Output (`python examples/06_scaling.py`)

```text
== fit_many: independent fits ==
  ch0: coeffs=[1.005 0.398]
  ch1: coeffs=[1.195 0.601]
  ch2: coeffs=[1.404 0.799]
  ch3: coeffs=[1.601 1.   ]
  ch4: coeffs=[1.798 1.2  ]

== project_spectra / fit_lsi_batched ==
spectra shape: (4, 7)
recovered b  : [0.4, 0.6, 0.8, 1.0]
true b       : [0.4, 0.6, 0.8, 1.0]

== PartitionedLSI: one pass, fixed memory ==
params: {'a': 1.3, 'b': 0.7}  n_samples: 5000

== map-reduce with merge() ==
params: {'a': 1.297, 'b': 0.7}
```
