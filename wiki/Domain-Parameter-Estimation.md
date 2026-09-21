# Domain -- Model parameter estimation

Recover the parameters of a known nonlinear model from noisy samples, with an
interval that means what it says, from clean or contaminated data and from
records too short or too uneven to be comfortable.

**Notebooks:**
[15_families](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/15_families.ipynb),
[14_robust_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/14_robust_image.ipynb),
[31_reference_datasets](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/realdata/31_reference_datasets.ipynb).

## Routes and baselines

- **dtfit:** `fit(basis="legendre")`, `fit(basis="block")`, `fit(basis="auto")`,
  each with `robust=True` for the reweighted image.
- **Baselines:** SciPy `curve_fit` (Levenberg-Marquardt and trust-region NLLS),
  `least_squares` with the soft-L1 loss, the historical integral forms from
  `dtfit-legacy`, and the certified values of the NIST StRD corpus.

## What is measured

**Sixteen families** from control, spectroscopy, growth and kinetics (damped
oscillation, sine, first-order rise, biexponential, Gaussian and Lorentzian
peaks, logistic, Gompertz, Weibull, Michaelis-Menten, Hill and others), 40
seeds at each of five noise levels, and dtfit's own 25-scenario accuracy
corpus.

| | Legendre / NLLS | block / NLLS |
|---|---|---|
| median parameter-error ratio, sixteen families, 5 percent noise | 1.000 | 1.025 |
| range of that median over 2 to 30 percent noise | 0.997 to 1.000 | 0.985 to 1.055 |
| median over the 25-scenario corpus | 1.000 | 0.980 |

The image reaches the pointwise fit's own optimum: the largest relative
distance between the two estimates runs 4.0e-06 to 2.6e-02 over the families.
The parity is a median. Thirteen of the sixteen families trail a basis by more
than a tenth somewhere in the noise grid, none at all five levels, and the list
of trailing families changes with the noise draw. `basis="auto"` returns the
Legendre candidate on all sixteen.

**Sampling regimes.** A concentrated transient reads 1.04 and 1.12 times NLLS
(Legendre, block) against seed spreads of 0.08 and 0.29; sparse irregular
sampling 1.00 and 1.05; a short record 1.01 and 0.96. None separates the image
from NLLS by more than the scatter.

**Contamination.** The plain fit has no robustness in either basis. At 10
percent scattered outliers at 8 sigma the median parameter error is 5.30
percent for NLLS, 5.11 for plain Legendre and 4.77 for plain block; the
reweighted image reads 0.44 and 0.45. On clean data the reweighting costs a
few percent at most. Under bursts the Legendre basis is the one to use: robust
block grows 2.9 times over burst lengths 1 to 40 where robust Legendre stays
flat.

**NIST StRD.** On the ten nonlinear-regression sets from both NIST starts the
Legendre image's error falls toward the certified parameters as the order
rises wherever the record holds enough observations. On MGH17 and BoxBOD,
where `curve_fit` diverges from the far start, the default order guards and a
fixed order recovers both under 1 percent. The block image does not keep pace
on these short non-uniform records: where both build at twice `order_for` its
error is 19.5 times Legendre's or more. The Puromycin replicate design lands
both bases within 0.6 percent of NLLS.

**Model mismatch.** No estimator rescues a structurally wrong model, and the
image and NLLS reach the same in-sample optimum on it. The R2 gap flags two of
three mismatches; a biexponential fitted as one decay shows only in the
residual autocorrelation, at 4.1 times the correct fit's.

## Where dtfit loses

- The historical integral forms trail the image on cycles (7.74 times NLLS on
  the damped oscillation against 1.01), and direct equal areas reads 1.153
  times NLLS in the corpus median.
- The block basis on short, non-uniform records (NIST) and under bursts.
- A record that holds fewer observations than the order needs: the Legendre
  image refuses on four of the ten NIST sets at some order or start and
  returns no number.

## Reading it

The image is NLLS restricted to the span of a basis, and at the default order
the restriction costs nothing measurable. What the image adds is not accuracy
on a resident record: it is the fixed-size additive statistic
([Big data](Domain-Big-Data), [Archive showcase](Domain-Image-Showcase)), the
reweighting that lives in the statistic, and the window form
([Embedded control](Domain-Embedded-Control)).
