# dtfit documentation

`dtfit` fits models that are **nonlinear in their parameters** -- exponentials,
sinusoids, logistic curves, saturating and peak shapes -- to noisy data, and
recovers the *physical parameters* (a growth rate, a frequency, an asymptote),
not just an opaque curve. It does this through **differential (non-Taylor)
transformations**: instead of comparing the model to the data sample-by-sample,
it compares their *images* -- a fixed-size statistic of projections onto a
basis. The image is additive over samples, so one fit runs in one shot, chunk
by chunk, across workers, or one sample at a time.

This folder is the documentation. Pick the door that matches what you need:

| If you want to... | Read |
|---|---|
| **Understand the ideas** from scratch, no heavy math assumed | [guides/](Guides) -- plain-language explanations of every method, with the proofs built up gently |
| **See the full map** -- every method, version, variant and adaptation | [guides/lineage-and-variants.md](Guides-Lineage-and-Variants) -- the complete atlas of where each approach came from and how it relates |
| **How the method evolved** -- the stages from spectrum balance to the image, each one runnable | [Lineage](Lineage) -- where every historical stage lives in the `dtfit-legacy` package |
| **Look up a function or class** -- signatures, arguments, return types | [api/](API) -- complete reference for the public `dtfit` API |
| **See the rigorous math** -- the formal derivations and proofs | [methods/](Methods) -- the mathematical reference, one file per method |
| **Learn by running code** -- copy-paste examples | [examples/](Examples) -- quickstart -> methods -> models -> sklearn -> streaming -> scaling -> diagnostics |
| **Understand the research** -- the experimental adaptations and how they were validated | [experimental/](Experimental) -- the `dtfit-experimental` package, the adaptations in trial, and every baseline the methods are compared against |
| **See what is measured** -- every claim with its numbers, wins and losses | [Experiments](Experiments) -- the experiment notebooks with a summary of each, and [Domains](Domains) -- the same results by application |

## The shortest possible introduction

There are **three core fitting methods**, plus a streaming one. They are all the
same idea (match the images of the model and the data) applied differently:

- **LSI** -- the batch fit in the Legendre basis: accurate and
  general-purpose, at parity with pointwise least squares. *Start here.*
- **EAC** -- the batch fit in the block basis of equal windows: local, the
  basis for records with jumps at known epochs and for data known only as
  per-window totals.
- **DSB** -- a symbolic *reference* method, used to derive and check the others;
  not for production.
- **ImageFilter** -- the streaming version: feed one sample at a time, track
  parameters that change over time, and detect when the system changes regime.
  `ImageFilter(basis="legendre")` and `ImageFilter(basis="block")` fix its basis to Legendre and block.

On top of those sit convenience layers: a [scikit-learn estimator](API-Estimator)
(`NonlineRegressor`), a [model catalog](API-Models) so you pick a *shape*
instead of writing a formula, [one-call "just fit it" entry points](API-Auto)
(`fit(..., basis="auto")`, `auto_forecast`), and [scaling backends](API-Scaling) for
big or multi-channel data.

For genuinely **random** data (economic / financial series) there is a dedicated
[stochastic-series solution](API-Stochastic): it fits the deterministic
*functionals* of the process (its autocovariance, spectrum, trend/cycle) to
characterize, forecast, and even generate it (`fit_stochastic`, `Stochastic`), with
a streaming twin (`StochasticFilter`) that tracks the structure live.

New here? Open [guides/README.md](Guides).
