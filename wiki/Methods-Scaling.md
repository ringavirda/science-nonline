# Scaling -- map-reduce, GEMM-batched & parallel batch fitting

> Numeric **batch-at-scale** backends. Source:
> [`scale/_partitioned.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/scale/_partitioned.py),
> [`scale/_batched.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/scale/_batched.py),
> [`scale/_parallel.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/scale/_parallel.py).
> `PartitionedLSI`, `PartitionedEAC`, `PartitionedBatchLSI`, `fit_lsi_batched`,
> `fit_many` (top-level); `project_spectra` via `from dtfit.scale import
> project_spectra`. API: [../api/scaling.md](API-Scaling).

The fitting *math* of [LSI](Methods-LSI)/[EAC](Methods-EAC) is unchanged here; these are
alternative **execution backends** that run those methods on data too big for
memory, spread across workers, or spanning thousands of channels. They are exact --
not approximations -- because the empirical spectrum has two structural properties:
it is **additive over the domain** and **linear across channels**.

## The two structural properties

### Additive over the domain -> map-reduce / streaming

The LSI empirical coefficient is an integral,

$$
\beta_j \;=\; \frac{2j+1}{H}\int_{x_0}^{x_N} y(x)\,P_j\big(u(x)\big)\,dx
\;=\; \frac{2j+1}{H}\,s_j,
\qquad s_j = \int y\,P_j\,dx .
$$

An integral over the whole domain is the **sum** of integrals over a partition of
it:

$$
s_j \;=\; \sum_{p} \int_{\text{chunk}_p} y\,P_j\,dx .
$$

So a stream can be reduced **chunk by chunk** in fixed $O(\text{order})$ memory,
and independent workers can each accumulate a partial $\mathbf s$ and **`merge`**
by addition -- an associative, order-independent reduce. The same holds for
[EAC](Methods-EAC): a window's area is additive over the samples that fall in it.

**Exactness at chunk boundaries.** The trapezoid rule needs the interval
*connecting* two chunks. Each `update` therefore carries the previous chunk's last
sample into the next, so the connecting interval is integrated exactly and the
partial sums add up to a single whole-domain projection -- provided chunks are fed
in domain order (and, for `merge`, that partitions share boundary samples).

### Linear across channels -> one GEMM

Folding the trapezoid weights into a weight vector $w$ ($\int y\,dx \approx \sum_i
w_i y_i$) turns each projection into a single matrix product. For $B$ channels
sharing the sampling grid $x$, stacked as columns of $Y\in\mathbb R^{n\times B}$,
**all** their empirical integrals are one GEMM:

$$
S \;=\; D^{\top}\,(w \odot Y) \;\in\; \mathbb R^{(L+1)\times B},
\qquad D_{ij} = P_j(u(x_i)) .
$$

(In practice the weights are folded into the *small* $(n,L{+}1)$ design,
$S = (w\odot D)^{\top}Y$, so the only large array touched is $Y$ -- read once into
the matmul.) Because it is a plain GEMM, it runs on multithreaded BLAS on the CPU
or on cuBLAS/torch on a GPU by swapping only **where the arrays live** (see the
pluggable [Backend](API-Scaling)); the projection has low arithmetic
intensity, so a GPU pays off mainly when the data is already resident, but the code
path is identical.

## The estimators

### `PartitionedLSI` / `PartitionedEAC` -- one-pass / distributed (map-reduce)

Accumulate, then solve. `PartitionedLSI` accumulates the additive projection
integrals $\mathbf s$; `PartitionedEAC` accumulates per-window areas. Both expose
`update(x_chunk, y_chunk)` (fold a chunk), `merge(other)` (combine workers), and
`fit(p0=...)` (solve the spectral / area match -- LSI's `solve_spectral`, EAC's
midpoint-area least squares). Fixed memory, exact, one pass.

```
acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
for x_chunk, y_chunk in stream:   # one pass, O(order) memory
    acc.update(x_chunk, y_chunk)
result = acc.fit(p0=[1.0, 1.0])
```

`PartitionedEAC` matches the model's window areas to the reduced data areas; it
uses a **midpoint-rule** model area per window ($f(\text{center})\times\text{width}$)
so the model side is cheap and consistent across solver iterations, while the data
areas were accumulated by trapezoid during the reduce.

### `project_spectra` / `fit_lsi_batched` -- GEMM-batched multi-channel

`project_spectra(x, Y)` returns the $(B, L{+}1)$ empirical spectra of $B$ channels
in a single GEMM. `fit_lsi_batched(x, Y, expr, var)` then solves each channel's
small spectral match on the host (it is `len(params)`-dimensional and negligible),
returning one [`FittingResult`](API-Types) per channel. Maximal throughput,
$O(N\cdot B)$ memory (data resident), GPU-able via `backend=`.

### `PartitionedBatchLSI` -- fused map-reduce **and** GEMM

The two levers combined: the **volume** partition of `PartitionedLSI` (a stream of
arbitrary length reduced in fixed $O(B\cdot\text{order})$ memory) *and* the
**channel** batch of `project_spectra` (each chunk's $(B,L{+}1)$ partial integrals
computed in one GEMM). The fusion is exact because the projection is linear across
channels and additive over the domain: each chunk's partial integrals are folded
in (with the same boundary-sample carrying), `merge` reduces across workers, and
`fit` solves every channel. Flat memory over volume *and* one matmul over channels
-- the estimator for **many channels and a stream too big for memory**.

### `fit_many` -- process/thread fan-out of independent fits

Orthogonal to the above: for **many independent problems** (different series and/or
models), `fit_many(problems, n_jobs=...)` fans the fits across a `joblib` pool
(`"loky"` processes, `"threading"`, or `"multiprocessing"`). Each
[`FittingProblem`](API-Scaling#fittingproblem) is picklable and a failed fit
is captured per-problem (its `error` set) rather than aborting the batch; results
come back in input order as ordinary picklable
[`FittingResult`](API-Types)s, each carrying the problem's `.label` (and `.error`
when it failed). `BatchFittingResult` remains a back-compat alias of
`FittingResult`.

## Optimizations and guards

- **Exact additive reduce** -- boundary-sample carrying makes chunked `update`s and
  `merge`s equal to a single whole-domain projection (validated in the big-data
  domain study across order-independent, variable-chunk and missing-data reduces).
- **Weights folded into the small design** -- `(w*D)^T.Y` avoids materializing an
  $(n,B)$ temporary; the only large array touched is $Y$.
- **Pluggable backend** -- the GEMM dispatches to NumPy/BLAS, CuPy or Torch by name
  (`"auto"` prefers a GPU); the math is written once with `@`/`*`/`.T`.
- **Numerical stability at scale** -- the additive reduction is the concern that
  bites at $10^8$-$10^9$ samples; the domain study profiles naive float32 vs
  float64 vs compensated (Kahan) summation.
- **Per-problem error capture** (`fit_many`) -- one failing problem does not abort
  the batch.

## Worked example

**Left:** a `PartitionedLSI` reduce over 8 chunks recovers the *identical* fit to
whole-batch LSI -- the growth rate matches to `max|Deltacoef| ~= 5x10^-^7`, the only
difference being trapezoid boundary terms; the map-reduce is exact, not
approximate. **Right:** `fit_lsi_batched` recovers the per-channel growth rate of
**300 channels in one GEMM**, each landing on the truth diagonal.

![Partitioned exactness and GEMM-batched multi-channel recovery](figures/scaling.png)

## Where it is best applied

| situation | backend |
|---|---|
| stream too big for memory, one pass | `PartitionedLSI` / `PartitionedEAC` |
| distributed workers, then combine | the same accumulators via `.merge()` |
| many channels on a shared grid | `project_spectra` / `fit_lsi_batched` |
| many channels **and** a big stream | `PartitionedBatchLSI` |
| many independent series/models | `fit_many` |

**Trade-off.** Streaming/partitioned estimators trade peak throughput for
**bounded memory**; the GEMM-batched path trades memory (data resident) for
**maximal throughput**. Both return the same parameters as the in-memory
[LSI](Methods-LSI)/[EAC](Methods-EAC). For real-time *online* tracking (as opposed to batch
at scale) use the [streaming filters](Methods-Legendre-Filter).
