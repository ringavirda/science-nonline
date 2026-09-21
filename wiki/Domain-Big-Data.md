# Domain -- Big-data processing

Fit a model to a record that does not fit in memory, that arrives in shards on
several machines, or that is one of thousands fitted at once -- and get the
answer the resident fit would have given.

**Notebook:**
[22_scale](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/technology/22_scale.ipynb).
The same properties on two public archives: [Archive showcase](Domain-Image-Showcase).

## Routes and baselines

- **dtfit:** `Image.merge` over chunk images, `ImageStream` as the one-pass
  accumulator, `fit_many` over processes and threads, the channel projection
  `Phi^T Y` as one GEMM on numpy and on cupy.
- **Baselines:** the resident whole-record fit, a serial loop, one BLAS thread,
  polynomial surrogates of degree 2 to 8.

## What is measured

**The reduce is exact.** Eight chunk images merged in either order reproduce
the whole-data fit to 4.92e-14 relative in the coefficients and 2.07e-15 in the
parameters; reordering the merge moves the coefficients by 1.16e-12. A fit
missing one of eight shards moves the two parameters by 0.01 and 0.02 of a
standard error: a lost shard costs data, not an error.

**Numerics.** On an integrand chosen to be hard the float64 accumulation stays
at 1.9e-15 at every chunk count. Naive float32 sits at 1.3e-06; a Kahan float32
sum across chunks reaches 5.2e-08 at 1024 chunks, 24 times better, and does
nothing at one chunk.

**Memory.** On a million-sample 32-channel panel `ImageStream` peaks at 80.1
MiB against 335.7 MiB resident, and between the two record lengths tried its
peak moves by -4.5 MiB while the resident one grows by 400 percent.

**Many fits.** 200 fits take 1.93 s in a serial loop, 0.52 s on 4 processes
(3.5 to 3.8 times) and 0.20 s on 12.

**The surrogate trap.** A polynomial surrogate matches or beats the structured
fit in sample at 4 of 6 degrees and extrapolates 3.3 to 9431 times worse on
the held-out half, on a record that holds one exponential decay.

## Where dtfit loses

- Chunking does not pay in time: imaging eight chunks of a million samples and
  merging takes 0.074 s against 0.045 s whole. The reduce buys exactness under
  splitting, not speed against the resident route.
- On a short record `ImageStream` costs more memory than the resident fit (84.6
  against 67.1 MiB); its fixed cost is what buys the flat growth.
- `fit_many` on a thread pool is 1.11 times slower than the serial loop. The
  GIL owns that; nothing in the image route is a compiled kernel, and
  throughput comes from the process pool.
- With the host transfer counted, the cupy projection is slower than one BLAS
  thread at every width (1.14 to 1.95 times); warm on the device it takes 0.20
  of the numpy time at the widest. The projection is a device win only once
  the data is already there.
- The legacy map-reduce accumulators of `dtfit-legacy` reduce 16 to 48 times
  faster than the image merge
  ([notebook 17](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/17_evolution_matrix.ipynb));
  what they reduce to is a different estimator, 1e-4 and 1e-7 away from the
  whole-data fit.

One machine, synthetic panels, wall-clock numbers with the spread the notebook
prints beside them.
