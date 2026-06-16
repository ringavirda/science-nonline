# dtfit methods — mathematical reference

This folder documents each fitting method shipped in `dtfit`, with its
mathematical grounding in **differential (non-Taylor) transformations**, the
full algorithm, the optimizations and numerical guards in the implementation,
and where each method is best applied. Every claim is backed by a figure and a
comparison table generated from real code on model *and* real data — see
[reproducing the figures](#reproducing-the-figures-and-tables).

| method | doc | tier | runtime path | role |
|--------|-----|------|--------------|------|
| **DSB** — Differential Spectra Balance | [dsb.md](dsb.md) | reference | symbolic (offline) | analytical ground truth / derivation |
| **LSI** — Least-Squares Integral | [lsi.md](lsi.md) | batch | numeric (offline) | accurate batch fit & model selection |
| **EDA** — Equal Differential Areas | [eda.md](eda.md) | batch | numeric (offline) | noise-robust batch fit |
| **EqualAreasFilter** — recursive EDA | [equal_areas_filter.md](equal_areas_filter.md) | streaming | numeric (online) | real-time tracking + drift detection |

The three production methods (LSI, EDA, EqualAreasFilter) are numeric successors
to the symbolic methods. DSB is kept as the analytical reference. The split
follows the dissertation's hard requirement that the **runtime path carry no
unbounded symbolic solve** — SymPy is allowed only once, offline, at
derivation/init time.

---

## The common foundation: differential transformations

All four methods operate on the **differential spectrum** of a signal rather
than on its samples directly. For an analytic function $x(t)$ the (scaled)
differential transform about $t_0=0$ is the sequence of *discretes*

$$
X(k) \;=\; \frac{H^{k}}{k!}\,\left.\frac{d^{k}x}{dt^{k}}\right|_{t=0},
\qquad k = 0,1,2,\dots
$$

where $H$ is a scale constant (in this library, the observation interval
$H = t_{\max}-t_{\min}$). The inverse transform reconstructs the function:

$$
x(t) \;=\; \sum_{k=0}^{\infty} X(k)\,\Big(\tfrac{t}{H}\Big)^{k}.
$$

So $X(k)$ is the $k$-th Maclaurin coefficient of $x$, rescaled by $H^k$. The
transform is a **linear bijection** between an analytic function and its
spectrum on the radius of convergence: two analytic functions are equal **iff**
their spectra agree discrete-by-discrete. This is the identity every method
below leans on — matching spectra (DSB, LSI), matching integrals of spectra
(EDA, the filter), recovers the function and hence its parameters.

### The non-Taylor base (and why a table is no longer needed)

The classical motivation of the scheme is that elementary functions have
**closed-form discretes** — you never truncate an infinite Taylor series for
them:

| term | discrete $X(k)$ |
|------|------------------|
| constant $c$ | $c$ at $k=0$, else $0$ |
| $t^{\,n}$ | $H^{k}$ at $k=n$, else $0$ |
| $e^{w t}$ | $(wH)^{k}/k!$ |
| $\sin(w t)$ | $\dfrac{(wH)^{k}}{k!}\sin\!\frac{\pi k}{2}$ |
| $\cos(w t)$ | $\dfrac{(wH)^{k}}{k!}\cos\!\frac{\pi k}{2}$ |

A transcendental model such as $a\,e^{bt}$ or $a\,\arctan(wt)$ is represented by
the *exact* discrete of its basis function, so the parameters $a,b,w$ appear
analytically in the spectrum and can be solved for or fitted.

**Implementation note.** `dtfit` no longer maintains this table in code. In a
*spectra balance* every equation matches model and data discretes at the same
order $k$, so the $H^{k}$ factor cancels on both sides (see [DSB](dsb.md)) and
the balance reduces to matching plain Maclaurin coefficients $g^{(k)}(0)/k!$.
Those are produced for any expression by generic SymPy differentiation
([`extra/dt/taylor.py`](../../src/dtfit/extra/dt/taylor.py)), which both
reproduces the table above and extends the scheme to *any* differentiable model
(rational, logarithmic, mixed) without a per-function rule. The numeric methods
go further and replace the monomial spectrum with a better-conditioned
orthogonal-polynomial (Legendre) spectrum ([LSI](lsi.md)).

---

## How the four methods relate

```
                differential spectrum  X(k) = (H^k/k!) x^(k)(0)
                                 │
        ┌────────────────────────┼─────────────────────────────┐
        │                        │                              │
   exact balance          weighted integral             integral / area
   F(k;θ)=Z(k)            L2 of spectra                  matching
        │                        │                              │
      DSB                       LSI                            EDA
  (symbolic solve)      (∫ reconstruction error)       (∫ over windows)
                                                               │
                                                       recursive / online
                                                               │
                                                      EqualAreasFilter
```

- **DSB** sets the empirical spectrum (from a polynomial pre-fit) equal to the
  symbolic model spectrum, discrete by discrete, and *solves* the algebraic
  system — exact when well-posed, but symbolic and noise-sensitive.
- **LSI** relaxes the exact balance to a **weighted integral least-squares**
  discrepancy of the two spectra — numeric, noise-tolerant, accurate.
- **EDA** matches **integrals (areas)** of model and data over windows rather
  than spectra — integration smooths noise, so it is the most robust.
- **EqualAreasFilter** runs EDA **recursively**, one sample at a time, with a
  Kalman-style update and drift detection — the real-time path.

---

## Reproducing the figures and tables

All figures in the `figures/` subfolder and every comparison table in these
docs are produced by one script against real downloaded data plus clearly
labelled model (synthetic) data:

```bash
python experiments/download_data.py   # fetch COVID-19 + USD/UAH (once)
python experiments/benchmark.py        # write figures/*.png and print tables
```

Real datasets and the dissertation-domain rationale are documented in
[../../experiments/README.md](../../experiments/README.md). Baselines compared
against are SciPy `curve_fit` (Levenberg–Marquardt nonlinear least squares, the
NLS gold standard), `numpy.polyfit` (a linear-in-parameters surrogate), and the
naive random walk (the standard FX one-step benchmark).
