# The experimental package -- adaptations and validation

`dtfit-experimental` is a **separate package** on top of stable `dtfit`. It
holds the adaptations in trial, the study tier the experiments stand on, and
the experiment notebooks that carry every measured claim of the project.
Nothing here ships inside the published `dtfit` wheel, so the public API stays
lean.

This page covers:

- [the promotion model](#promotion) -- how an adaptation graduates from
  experimental to stable;
- [the adaptations in trial](#adaptations) -- what each one is, the math it
  rests on, and what its experiment reads;
- [the experiments](#suite) -- the notebooks and the tier of helpers under
  them;
- and, on a companion page, [every baseline](Experimental-Baselines) the
  methods are compared against and **why each was selected**.

Source: [`packages/dtfit-experimental/`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental).

---

<a name="promotion"></a>
## 1. The promotion model -- why separate packages

The project separates **what is proven** from **what is being tried**:

- `dtfit` (stable) -- the image core, the streaming filter, the stochastic
  tier, the model catalog. The lean, public, supported API.
- `dtfit-experimental` -- adaptations in trial, the study tier, the experiment
  notebooks and the datasets. It *depends on* `dtfit`; it is never depended on
  *by* it.
- `dtfit-hardware` -- the embedded rig; it depends on `dtfit-experimental`.
- `dtfit-legacy` -- the historical stages of the method (the source-form
  criteria, the integral fitters, the recursive filters, the map-reduce
  accumulators), kept for the
  [evolution matrix](Experiments-Method#17----the-evolution-matrix). Only
  the study tier imports it.

An adaptation is held to one rule, applied in
[notebook 19](Experiments-Method#19----adaptations-in-trial): a win over the
plain route on its own signal class, outside the seed scatter, or parity where
no plain route exists. An adaptation that clears it across more than one
application is **physically moved** into `dtfit` and imported from there --
there is no re-export shim, so the dependency only ever points one way.

**In stable `dtfit` by this route:**

| adaptation | in `dtfit` as |
|---|---|
| the oscillatory recipe | `fit(..., oscillatory=True)` or `fit(..., freq_param="w")`, `dtfit.image.fft_frequency_seed` |
| one-pass and distributed reduction | `ImageStream`, `Image.merge` |
| batched multi-channel projection | `ImageStream(channels=B)` |

---

<a name="adaptations"></a>
## 2. The adaptations in trial

Every adaptation rests on the *same* statistic the core uses: the image
`S = Phi^T y`, `G = Phi^T Phi` of a record in a basis `Phi`, additive over
samples and fitted by `dtfit.fit`. An adaptation changes the basis, the way
the image is formed, or the way a filter carries it; none changes the
criterion. Signatures are on the
[adaptations API page](Experimental-Adaptations-API).

### Image bases -- `FourierBasis`, `ChebyshevBasis`, `LaguerreBasis`

**Intuition.** The coefficients of an image are spent best in a basis that
matches the signal: a periodic record is a few harmonics but many polynomial
orders, a transient is natural in decaying functions. Each class is a basis
object passed as `dtfit.fit(..., basis=FourierBasis(K))`; the fit, the
covariance and the merge are those of the core.

**The math it rests on.** The image fit needs no orthogonality: `G` carries
the correlation of the basis functions on the sample grid exactly, so any
family with `evaluate(u)` and `n_coef` is a valid basis.

**What is measured.** On a periodic signal the Fourier basis reaches a 1e-6
reconstruction at 9 coefficients against Legendre's 26. On a decay the
Laguerre basis needs 9 against Legendre's 10, which is parity.

### Windows with explicit edges -- `EdgeBlockBasis`, `SegmentBasis`

**Intuition.** The block basis of the core places equal windows. A record
whose level jumps at a known epoch is better served by a window edge on the
epoch, so that no window averages across it. `EdgeBlockBasis` takes the edges
as given and lets windows be left out; `SegmentBasis` puts a Legendre
polynomial on each segment between the edges.

**The math it rests on.** Indicators of disjoint windows give a diagonal
Gram of sample counts whatever the edges are; polynomials on disjoint
segments give a block-diagonal one.

**What is measured.** With one edge on each epoch the jump terms reach an
efficiency of 0.989 to 0.991 at 32 windows where Legendre is at 0.895; on
120 GPS station series the offsets are recovered to 0.032 standard errors
against 0.393 for Legendre and 0.435 for equal windows.
On a gapped record the block basis loses: its Gram stays under 200 where
Legendre's passes 1e12, and it still reads 1.05 to 3.19 times the pointwise
RMSE where Legendre reads 1.00.

### Data known as window totals -- `aggregated_image`, `fit_aggregated`

**Intuition.** A meter that reports one total per interval, or an archive
that keeps only window means, has already formed the block image: the totals
are `S` and the counts are the diagonal of `G`. Fitting the model's window
sums to them is the equal-areas criterion in its direct form. Reading each
mean as a sample at the window centre is a different, biased fit.

**The math it rests on.** The window totals are the exact sufficient
statistic of the record in the window basis; the model is projected onto the
same windows on the sample grid.

**What is measured.** The fit on the totals removes the midpoint reading's
bias of -6.1 to +6.6 percent (under 0.053 percent after). On temperature
records read as 3 h and 6 h means it returns the hourly amplitudes to 1.0001
to 1.0048 where the midpoint reading returns 0.906 to 0.978.

### Jumps at unknown epochs -- `fit_aligned`, `detect_jumps`

**Intuition.** When the epochs are not known the windows are aligned to them
from the data. A fine block image carries the residual of a first fit; a
skip-one local-linear test on the fine window means finds the windows that
hold a jump; those windows are left out, the rest are merged to the coarse
count, and the model is refitted with one level shift per epoch.

**The math it rests on.** The detector reads the fine image alone -- counts,
position sums and residual sums per window -- so detection costs one pass
over the data; coarsening a block image is a 0/1 aggregation of its windows.

**What is measured.** On generated records the jump terms of `fit_aligned`
reach 0.955 to 0.969 of the accuracy of the fit that is told the epochs, in
every case where detection finds all of them; at jumps of two noise standard
deviations detection recalls 0.46 and decides the result. On the station
archive detection decides the answer and the basis does not.

### Weak-form rate laws -- `weak_operators` and the four fitters

**Intuition.** A rate law that is linear in its constants can be identified
without solving it: multiply by test functions that vanish at the ends of the
record, integrate by parts, and the derivatives move from the noisy data onto
the smooth test functions. What remains is a linear system in the constants.
No starting guess, no ODE solve.

**The math it rests on.** The test functions are
`(1 - u^2)^order * P_k(u)` with `P_k` the Legendre polynomials; the boundary
terms of the integration by parts vanish with the window factor.

**What is measured.** A call takes about 1 ms against 31 to 205 ms for the
solved fit. It is less accurate than the solved fit from a good start, by
1.69 to 4.39 times, and ahead of finite-difference regression at every law
and noise level. Used as the start of the solved fit (`seed_nlls`) it reaches
the good-start accuracy to within 0.005 percentage points, and on
Michaelis-Menten removes the 97 to 100 percent failure rate of a start at
three times the truth. It breaks on a stiff pair: 88 percent error on roots
-1 and -10 where the solved fit is at 1.3 percent.

### A tracker in the time of its newest sample -- `LocalTimeFilter`

**Intuition.** A polynomial tracker on absolute time stamps conditions its
window on the clock's zero. `LocalTimeFilter` moves the polynomial's origin
onto every new sample, so the coefficients are the value, the rate and the
curvature *now*, and the process noise is a rate per unit of time that
carries across sampling rates.

**The math it rests on.** A shift of a polynomial's origin is an exact linear
map of its coefficients (`shift_matrix`), applied to the state and its
covariance between samples.

**What is measured.** On the rig's drives a 10000 s shift of the time stamps
moves the absolute-time tracker's two-sample RMSE from 1.73 m to 8.91 m; with
the window held it moves no score of the re-centred one, and with the adaptive
window the 5 Hz drive reads 1.40 m or 1.44 m by the shift. The re-centred tracker takes the 1 Hz
drive from 5.85 m to 4.14 m at two samples; it is 1 to 7 percent behind the
absolute-time tracker on the 5 Hz drive and a fifth behind it on the walk.

### Information-form fusion -- `InformationFilter`

**Intuition.** A recursive *linear* estimator that keeps the inverse
covariance: absorbing a measurement is an addition, and two estimators that
saw different data fuse by adding their information, in any order.

**What is measured.** Fusion agrees with a single undivided pass to 7e-15
relative, and with the merged-image `dtfit.fit` route to 8e-9. It has no
plain route to lose to, so parity keeps it.

### Several streams -- `FilterBank`, `FusedChiSquareDetector`

`dtfit_experimental.streaming` drives several `dtfit.ImageFilter` instances in
lockstep and pools their innovations into one chi-square test; see
[the filter bank page](Methods-Filter-Bank). On the three-axis fault of
[notebook 21](Domain-Embedded-Control) a test on every sample flags in 1 step
against 19 for the once-a-window test, and per-axis tests on every sample
match the pooled test at every fault size tried.

---

<a name="suite"></a>
## 3. The experiments -- how claims are measured

The experiments are Jupyter notebooks under
[`packages/dtfit-experimental/experiments/`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments),
in three groups: `method/` (what the image is and where each form stops),
`technology/` (filtering, scale, the stochastic image, the embedded
footprint) and `realdata/` (reference datasets, forecasting, GPS, the archive
showcase, measured stochastic series). A notebook is the experiment and its
report: the code, the tables, the figures, the findings and the limits, in
one file that runs top to bottom. The index with a summary of each is on
[Experiments](Experiments); the application view is on [Domains](Domains).

Under the notebooks sits the study tier, `dtfit_experimental.study`: the
model families and their truth, the [baselines](Experimental-Baselines), the
Monte-Carlo grid, the plants, the GPS simulator and benchmark loaders, the
historical stages, the dataset paths. It is imported module by module and is
not part of the package's public names.

Where dtfit trails a baseline the notebook says so and names the cause; each
domain page closes with a section of those cases.

---

## 4. Install and run

```bash
pip install -e packages/dtfit                       # stable dtfit
pip install -e "packages/dtfit-experimental[bench]" # this package + matplotlib/torch/statsmodels/pandas
pip install -e packages/dtfit-legacy                # for the evolution matrix

python -m dtfit_experimental.study.download_data    # fetch the public datasets

jupyter lab packages/dtfit-experimental/experiments/method/12_discrete_image.ipynb
# or headless:
jupyter nbconvert --to notebook --execute --inplace \
    packages/dtfit-experimental/experiments/method/12_discrete_image.ipynb
```

`DTFIT_QUICK=1` runs a notebook at a reduced size (the package's tests run
every notebook that way); `DTFIT_DATA` points the loaders at a dataset
directory other than `experiments/data`. Next:
**[the baselines](Experimental-Baselines)** -- what every comparison method is
and why it was chosen.
