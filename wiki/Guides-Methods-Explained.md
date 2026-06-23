# The methods explained

Each method below follows the same four-part shape:

1. **The intuition** -- what it's doing, in plain words.
2. **How it works** -- the actual steps.
3. **Why it's correct** -- the proof, built up gently.
4. **Knobs & adaptations** -- what you can tune and the variants that exist.

They all rest on the one idea from [the guide](Guides): *match the integral
fingerprint (differential spectrum) of the model to that of the data.* If you
haven't read that page, read its sections 2-3 first. For the fully formal
treatment of any method, follow the link to [../methods/](Methods).

Quick map:

- [DSB -- the exact, symbolic reference](#dsb)
- [LSI -- the accurate batch fitter](#lsi)
- [EAC -- the robust, fast batch fitter](#eac)
- [The streaming filters -- real-time tracking](#streaming)

---

<a name="dsb"></a>
## DSB -- Differential Spectra Balance (the reference)

Full math: [../methods/dsb.md](Methods-DSB).

### The intuition

DSB is the purest expression of the core idea. It says: *make the model's
fingerprint exactly equal the data's fingerprint, number for number, and solve
for the parameters.* No approximation, no least-squares slack -- an exact balance.

Think of it as a set of scales. On the left pan you put the model's fingerprint
numbers (which contain the unknown parameters as algebra); on the right pan the
data's fingerprint numbers. You balance the pans one level of detail at a time.
Each balanced pan is one equation; with as many equations as unknown parameters,
you solve the system exactly.

### How it works

1. **Summarize the data with a polynomial.** A polynomial (a sum of powers of
   $t$) can be fit to data by *linear* least squares -- easy and stable. Its
   coefficients *are* the data's fingerprint numbers, read off directly.
2. **Write the model's fingerprint** by differentiating the model expression
   symbolically (with SymPy). The unknown parameters appear inside these
   expressions.
3. **Balance** -- set model fingerprint = data fingerprint at each level of
   detail, producing a small system of equations.
4. **Solve symbolically** for the parameters; if there are more equations than
   unknowns, refine the solution numerically.

### Why it's correct

This rests on the faithful-ID property from the guide: two smooth functions are
equal **if and only if** their fingerprints agree at every level. So if the data
truly came from $f(t;\theta^*)$ for some true parameters $\theta^*$, and your
polynomial captured enough levels of the fingerprint, then forcing the model
fingerprint to equal the data fingerprint forces the model to equal the true
function -- which pins down $\theta = \theta^*$. On clean, ideal data this is an
**exact identification**, not an approximation. That is precisely why DSB is the
*reference*: the numeric methods are judged against the answer DSB would give on
perfect data.

A neat simplification the implementation exploits: in the balance, a scaling
factor that appears on *both* pans (the $H^k$ in the formal definition) cancels,
so DSB just matches plain polynomial coefficients -- and that works for **any**
model you can differentiate, not just a hand-coded list of exp/sin/cos.

### Knobs & limits

- `rank` -- how many fingerprint levels (equations) to balance.
- **Why it's not for production:** the symbolic solve has unpredictable runtime,
  and it leans on the polynomial's *high-order* coefficients, which are exactly
  the ones noise corrupts most. On noisy data DSB becomes an unreliable curve fit.
  Use it to derive and validate; use LSI/EAC to actually fit.

---

<a name="lsi"></a>
## LSI -- Least-Squares Integral (the accurate default)

Full math: [../methods/lsi.md](Methods-LSI).

### The intuition

LSI keeps DSB's idea but drops the demand for an *exact* balance. Real data is
noisy, so an exact match is both impossible and undesirable (you'd be matching
the noise). Instead LSI asks for the **best possible** match: make the model's
fingerprint as close as possible to the data's, in the least-squares sense. This
one change turns a brittle symbolic solve into a robust numerical fit.

It also fixes a subtle numerical trap. If you build the fingerprint from plain
powers ($1, t, t^2, t^3, \dots$), the high powers look almost identical to each
other over an interval, which makes the math *ill-conditioned* -- tiny data
changes cause huge parameter swings. LSI swaps the plain powers for **Legendre
polynomials**, a set of "spread-out," mutually-independent shapes that don't step
on each other. With them the problem becomes perfectly stable.

(If "Legendre polynomials" means nothing to you: think of them as a better set of
measuring sticks. Plain powers are like rulers that are almost the same length,
so you can't tell them apart; Legendre polynomials are rulers of clearly
different, independent lengths, so each measures something distinct.)

### How it works

1. **(Optional) lightly smooth** the data to tame noise.
2. **Data fingerprint:** project the data onto Legendre polynomials (a
   well-behaved least-squares fit), giving stable fingerprint coefficients.
3. **Model fingerprint:** project the model onto the same Legendre polynomials,
   computing the integrals *exactly* by Gaussian quadrature (so the model is
   integrated, not crudely approximated). These coefficients contain the unknown
   parameters.
4. **Match:** tune the parameters to minimize the weighted difference between the
   two sets of coefficients. Because the Legendre basis is *orthogonal*, this
   difference is a clean, perfectly-conditioned sum of squared coefficient gaps.
5. **Solve:** plain local optimization (Levenberg-Marquardt) by default, or -- if
   you give parameter `bounds` -- a global search first, so it can't get trapped
   in a wrong local minimum.

### Why it's correct

Start from the most natural goal: make the model curve close to the data curve in
the integrated-squared-error sense,

$$ J(\theta) = \int \big[\,\text{data}(t) - \text{model}(t;\theta)\,\big]^2\,dt . $$

Now expand both curves in Legendre polynomials. Because those polynomials are
**orthogonal** (their cross-integrals vanish), this single integral collapses
into a simple sum:

$$ J(\theta) = \sum_j \frac{H}{2j+1}\,\big(\beta_j^{\text{data}} - \beta_j^{\text{model}}(\theta)\big)^2 . $$

In words: *minimizing the area between the two curves is the same as minimizing
the gap between their fingerprint coefficients, one coefficient at a time.* That
is the whole justification -- and it's why LSI is both faithful to the data
(it minimizes real reconstruction error) and numerically stable (the sum is
diagonal, no ill-conditioned matrix). The full derivation, including why the
plain-power version is a notorious *Hilbert matrix* and the Legendre version
isn't, is in [../methods/lsi.md](Methods-LSI).

### Knobs & adaptations

- `k_star` -- how many fingerprint levels to match (the spectral order). Higher
  resolves finer structure; `"auto"` picks it by an information criterion (BIC).
- `alpha` -- extra down-weighting of high orders (usually unnecessary; the
  orthogonal basis already handles it).
- `filter_data` -- the optional pre-smoothing.
- `bounds` -- supplying per-parameter ranges switches on a **global search**,
  which is what lets LSI fit stubborn exponential/transcendental models without a
  good starting guess.
- **The oscillatory recipe** (`oscillatory=True` / `freq_param=`): smoothing and
  low order *erase* a cycle, so for sinusoids LSI turns smoothing off, raises the
  order to resolve the cycle, and seeds the frequency from the data's FFT peak. A
  sinusoid recovers to under 1% with this recipe versus ~50% without it. See
  [fft_frequency_seed](API-Fitting#fft_frequency_seed).
- **Pluggable basis** (experimental `fit_lsi_basis`): swap Legendre for Fourier
  (natural for periodic signals) or Laguerre (natural for decays). See
  [../experimental/README.md](Experimental).

---

<a name="eac"></a>
## EAC -- Equal-Areas Criterion (the robust, fast one)

Full math: [../methods/eac.md](Methods-EAC).

### The intuition

EAC matches the **simplest** fingerprint of all: the **area under the curve**.

Split the data into a handful of consecutive windows. For each window, measure
the area under the data, and the area under the model. Tune the parameters until
every window's model-area equals its data-area. That's it -- "equal areas."

Why does so crude a quantity work? Because area is an *integral*, and (from the
guide) integrals average out noise. EAC never differentiates the data and never
builds a wobbly high-order polynomial; it only ever sums the data up. That makes
it the **most noise-robust** of the batch methods and, because each window is one
cheap equation, the **fastest**.

### How it works

1. **Pick windows.** Take the informative part of the data and split it into
   $M$ windows (by default $2\times$ the number of parameters, so the system has
   some redundancy to average over).
2. **Data areas.** Integrate the data over each window (Simpson's rule).
3. **Model areas.** Integrate the model over each window -- and, for the optimizer,
   the integral of each parameter-derivative too (the *analytic integrated
   Jacobian*, computed once symbolically, so the solver gets an exact, smooth
   gradient).
4. **Solve** the "model area = data area" equations by least squares.

### Why it's correct

Each window gives one equation: model area on the window equals data area on the
window. Why do a few such equations pin down the parameters? Because an area is
itself a summary of the fingerprint -- integrating the curve over a window is a
particular weighted combination of its fingerprint numbers. So matching areas
over $M$ well-placed windows is matching $M$ independent summaries of the
fingerprint, and once you've matched as many independent summaries as the model
has free parameters, the model is pinned down (this is a *weak-form*, or Galerkin,
identification -- the formal statement is in [../methods/eac.md](Methods-EAC)).

The robustness has a one-line proof: zero-mean noise integrates toward zero,
$\int_W \varepsilon(t)\,dt \to 0$ as the window grows. The data enters EAC *only*
through these integrals, so the noise is gone before the fit even starts.

**Why overdetermine it?** Using more windows than parameters ($2m$ by default)
means the random per-window integration errors partly cancel across windows,
lowering the variance of the estimate -- and it lets EAC report a parameter
**covariance** (uncertainty) from the leftover residuals.

### Knobs & adaptations

- `n_windows` -- number of area equations. More windows localize information (and
  let a robust loss isolate outlier-contaminated windows); too many makes each
  window tiny and noisy. The default $2m$ is a safe start.
- `active_ratio` -- what leading fraction of the data to window. The default `0.8`
  assumes the information lives in an early transient; **set it to `1.0` for a
  saturating shape whose asymptote lives in the tail** (e.g. `arctan`), or the
  fit will be biased.
- `loss` + `f_scale` -- a robust loss (`"soft_l1"`, `"cauchy"`) down-weights
  windows an outlier has corrupted. This is the *single-fit* robustness path;
  for densely contaminated records prefer the `ensemble_fit` path below. Important
  caveat: the loss acts at the *window* level, and only "bites" when `f_scale` is
  set near the size of a clean window's area residual (the default `1.0` usually
  dwarfs them, silently behaving like plain least squares). See the worked
  discussion in [example 02](Example-02-Fitting-Methods).
- `bounds` -- constrained fits (switches to a trust-region solver).
- **Curvature windows** (`fit_eac(..., window_mode="curvature")`): instead of the
  default uniform windows, place window edges by **curvature** -- narrow where the
  signal bends, wide where it's flat -- so each window carries roughly equal
  information. This is the best estimator for localized transients and saturating
  (Michaelis-Menten / Hill) shapes. -> [api/fitting.md#fit_eac](API-Fitting#fit_eac)
- **Overlapping-window ensemble** (`ensemble_fit`): when *outliers* contaminate
  the record, fit many overlapping sub-windows and take the **median** of the
  per-window estimates -- whole corrupted windows are simply outvoted, with no
  `f_scale` tuning, and the inter-window spread is a free uncertainty band. More
  reliable than the robust loss on spiky data; on clean data prefer a single fit.
  -> [../methods/ensemble.md](Methods-Ensemble)

---

<a name="streaming"></a>
## The streaming filters -- real-time tracking

Full math: [../methods/equal_areas_filter.md](Methods-Equal-Areas-Filter).

### The intuition

The batch methods above look at the *whole* dataset at once. But sometimes data
arrives one sample at a time (a sensor, a live feed), the parameters **drift**
over time, and you need an answer *now*, at fixed cost per sample. That's what the
streaming filters do.

The mental model is a **Kalman filter** -- the classic "predict, then correct"
loop used in GPS and control systems:

- You hold a current estimate of the parameters, plus a sense of how unsure you
  are about each.
- Each new sample produces a **surprise** (how far the data was from what your
  current parameters predicted).
- You nudge the parameters to reduce the surprise -- nudging the uncertain ones
  more and the confident ones less -- and update your uncertainty.

The dtfit twist: the "surprise" is **not** a single-point error (which would be
noisy). It's the **area mismatch over a sliding window** (for `EACFilter`) or the
**spectrum mismatch over the window** (for `LSIFilter`). So even the streaming
filters inherit EAC/LSI's integrate-don't-differentiate robustness.

### How it works (per sample)

1. Add the new sample to a sliding window; drop the oldest.
2. Compute the window's area (or spectrum) for the data and for the model ->
   the **innovation** (the surprise) and its sensitivity to each parameter.
3. Apply the Kalman correction: move the parameters by gain x surprise, and
   update the uncertainty. All pure NumPy -- the symbolic work was done once at
   construction, so each update has **bounded cost** and is real-time safe.

### Detecting regime changes (drift)

A single smoothly-updated estimate cannot represent a **sudden structural break**
(a currency un-pegging, a plant fault). So the filter watches the stream of
surprises for two patterns:

- a single **big** surprise (a sudden jump) -- caught by a chi-squared test (NIS);
- a **sustained** lean in one direction (slow drift) -- caught by a two-sided
  CUSUM accumulator.

When either fires, the filter **re-arms** (resets or inflates its uncertainty) so
it re-adapts to the new regime instead of stubbornly averaging across the break.
Careful guards (self-standardizing the surprise, only testing on non-overlapping
windows, a warmup period) keep it from false-alarming on ordinary noise -- these
are detailed in [../methods/equal_areas_filter.md](Methods-Equal-Areas-Filter).

### Knobs & adaptations

- **Presets** -- rather than set the knobs below by hand, start from a curated
  classmethod: `EACFilter.tracking(...)` / `LSIFilter.tracking(...)` favors fast
  re-adaptation, and `.robust(...)` favors stability under outliers/dropouts. The
  individual knobs below still override anything a preset sets.
- `window_size` -- smoothing vs responsiveness (bigger = smoother, slower to react).
- `q_diag` -- how fast you allow each parameter to drift.
- `r` -- how much you trust each measurement.
- `cusum_k` / `cusum_h` -- drift-detector sensitivity vs false-alarm rate.
- `n_sub` (EACFilter) / `order` (LSIFilter) -- split the window into more
  sub-measurements for better observability of coupled multi-parameter models.
- **`LSIFilter` vs `EACFilter`:** use `LSIFilter` (spectrum measurement) for
  **oscillatory** plants -- the area criterion partly cancels oscillations, so the
  spectrum is the right fingerprint there; use the cheaper `EACFilter` for
  monotone/saturating signals.
- **`FilterBank` + `FusedChiSquareDetector`** -- run many streams in lockstep and
  pool their surprises to catch a fault that's too weak in any single stream but
  strong across all of them. -> [api/streaming.md](API-Streaming)

---

## Side-by-side summary

| | DSB | LSI | EAC | EACFilter / LSIFilter |
|---|---|---|---|---|
| **Matches** | exact fingerprint | fingerprint (least-squares, Legendre) | integrated areas / windows | area / spectrum, one sample at a time |
| **Mode** | symbolic, offline | batch, offline | batch, offline | streaming, online |
| **Best at** | derivation / reference | accurate general fitting | robust, fast, few-parameter | real-time tracking + drift detection |
| **Noise** | fragile | tolerant | most robust | robust (integral measurement) |
| **Use for production?** | no | yes (default) | yes | yes (real-time) |

Where to go next:

- **Which one for my data?** -> [choosing-a-method.md](Guides-Choosing-a-Method)
- **Exact signatures and arguments** -> [../api/](API)
- **The full proofs** -> [../methods/](Methods)
- **The experimental adaptations and how they were validated** ->
  [../experimental/](Experimental)
