# dtfit documentation

`dtfit` fits models that are **nonlinear in their parameters** — exponentials,
sinusoids, logistic curves, saturating and peak shapes — to noisy data, and
recovers the *physical parameters* (a growth rate, a frequency, an asymptote),
not just an opaque curve. It does this through **differential (non-Taylor)
transformations**: instead of comparing the model to the data sample-by-sample,
it compares *integral fingerprints* of the two, which is what makes it robust to
noise.

This folder is the documentation. Pick the door that matches what you need:

| If you want to… | Read |
|---|---|
| **Understand the ideas** from scratch, no heavy math assumed | [guides/](guides/) — plain-language explanations of every method, with the proofs built up gently |
| **See the full map** — every method, version, variant and adaptation | [guides/lineage-and-variants.md](guides/lineage-and-variants.md) — the complete atlas of where each approach came from and how it relates |
| **Look up a function or class** — signatures, arguments, return types | [api/](api/) — complete reference for the public `dtfit` API |
| **See the rigorous math** — the formal derivations and proofs | [methods/](methods/) — the mathematical reference, one file per method |
| **Learn by running code** — copy-paste notebooks | [guides/notebooks/](guides/notebooks/) — quickstart → methods → models → sklearn → streaming → scaling → diagnostics |
| **Understand the research** — the experimental adaptations and how they were validated | [experimental/](experimental/) — the `dtfit-experimental` package, the experiment suite, and every baseline it is compared against |

## The shortest possible introduction

There are **three core fitting methods**, plus a streaming one. They are all the
same idea (match integral fingerprints) applied differently:

- **LSI** — the accurate, general-purpose batch fitter. *Start here.*
- **EDA** — the fast, most noise-robust batch fitter, best for few-parameter
  transient/saturating shapes.
- **DSB** — a symbolic *reference* method, used to derive and check the others;
  not for production.
- **EDAFilter / LSIFilter** — the streaming versions: feed one sample at a time,
  track parameters that change over time, and detect when the system changes
  regime.

On top of those sit convenience layers: a [scikit-learn estimator](api/estimator.md)
(`NonlineRegressor`), a [model catalog](api/models.md) so you pick a *shape*
instead of writing a formula, [one-call "just fit it" entry points](api/auto.md)
(`auto_estimate`, `auto_forecast`), and [scaling backends](api/scaling.md) for
big or multi-channel data.

New here? Open [guides/README.md](guides/README.md).
