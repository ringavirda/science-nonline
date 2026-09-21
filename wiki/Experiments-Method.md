# Method studies -- notebooks 11 to 19

What the image is, what each basis buys, and where each form of the method
stops. Every number below is printed by the notebook named in its heading; the
notebook carries the full tables, the figures and the limits. Index and run
instructions: [Experiments](Experiments).

## 11 -- the source forms and where they stop

[11_source_forms](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/11_source_forms.ipynb)
runs the three source criteria -- the spectrum balance, the monomial integral
criterion and the equal-areas criterion -- against the image route.

- The balance's two adequacy indicators part ways: on the exponential the
  intermediate polynomial's residual keeps falling with degree while the
  parameter error has a minimum at degree 3 (0.78 percent) and is 31 percent at
  degree 8. On a sine the balance never recovers the parameters to better than
  230 percent.
- The monomial weight matrix passes a condition number of 1e16 at order 11 and
  its Cholesky factorization fails by order 13; the Legendre weight matrix is
  diagonal with ratio `2K+1` at every order.
- The monomial integral criterion reads the amplitude to 6 percent on half a
  period and to 87 to 96 percent from one period on; the image stays with the
  pointwise fit at every record length.
- The truncated equal-areas form is biased by 8 percent at `max|wt| = 0.6` and
  by 194 percent at 12; the block image evaluates the model directly and has
  no such bias. The source form's variance trade holds only inside the
  polynomial base it is built on, and the pointwise fit is below both there.

## 12 -- the image is the discrete statistic

[12_discrete_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/12_discrete_image.ipynb):
`Image.S` and `Image.G` are `Phi^T y` and `Phi^T Phi` to machine precision. At
the order `order_for` reports the image fit's RMSE ratio to the pointwise fit
stays within 0.0005 of 1.000 on uniform, random and clustered grids, relative
bias under 0.04 percent, interval coverage 0.94 to 0.97. Reading the same
criterion by quadrature on the model side costs 1.31 to 1.83 times the
pointwise RMSE on the uniform grid, 41 to 51 times on the random grid and 358
to 656 times on the clustered one. Order above `order_for`, up to 96, moves
the ratio by at most 0.0005.

## 13 -- two bases

[13_two_bases](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/13_two_bases.ipynb):
a Legendre coefficient buys global smoothness, a block coefficient buys
locality. On a smooth target Legendre is at machine precision by K=16 where
block is at 2.2e-2; a step on a window boundary is exact in the block basis at
every K; a step off the boundary is parity (ratio 0.75 to 1.29, the lead
changing with where the jump lands). The block Gram's condition number never
exceeds 10; Legendre's reaches 374762 on a clustered grid at order 96. Block
pays in efficiency: 0.9375 on a slope at order 4, reaching 1 only when the
window count equals the sample count.

## 14 -- the robust image

[14_robust_image](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/14_robust_image.ipynb):
the plain block fit has no robustness of its own -- at 10 percent scattered
contamination it reads 4.77 percent median parameter error against 5.30 for
plain NLLS and 5.11 for plain Legendre. Reweighting the image, in either
basis, is what carries it: 0.44 and 0.45 percent at the same contamination,
for a price of a few percent on clean data. Under bursts the block basis is the
worst: robust Legendre stays at 1.18 to 1.35 times the clean curve RMSE over
run lengths 1 to 40 while robust block grows 2.9 times, because narrow windows
end up more than half outliers.

## 15 -- sixteen families and the accuracy corpus

[15_families](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/15_families.ipynb):
the image recovers parameters at parity with pointwise least squares. The
median error ratio to NLLS over sixteen families is 1.000 for Legendre and
1.025 for block at 5 percent noise, and stays inside 0.997 to 1.000 and 0.985
to 1.055 from 2 to 30 percent; over dtfit's own 25-scenario corpus it is 1.000
and 0.980. Which families trail by a tenth does not reproduce from one noise
draw to the next. `basis="auto"` returns Legendre on all sixteen. The
historical integral least squares costs 7.74 times NLLS on the damped
oscillation where the image costs 1.01. No sampling regime (concentrated
transient, sparse irregular, short record) separates the image from NLLS by
more than the seed scatter, and no estimator rescues a structurally wrong
model. Application view: [Parameter estimation](Domain-Parameter-Estimation).

## 16 -- the weak form

[16_weak_form](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/16_weak_form.ipynb):
weak-form identification of a rate law needs no starting guess and no ODE
solve, and a call takes about 1 ms against 31 to 205 ms for the solved fit. It
is less accurate than the solved fit from a good start at every law and noise
level, by 1.69 to 4.39 times, and beats finite-difference regression at every
cell. Seeding the solved fit with the weak estimate reaches the good-start
accuracy to within 0.005 percentage points, and on Michaelis-Menten removes
the 97 to 100 percent failure rate of a start at three times the truth. It
breaks on a stiff pair: 88 percent error on roots -1 and -10 where the solved
fit is at 1.3 percent, and resolving the fast mode does not rescue it.

## 17 -- the evolution matrix

[17_evolution_matrix](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/17_evolution_matrix.ipynb)
runs every historical stage (from `dtfit-legacy`) and the image on one problem
set. Batch: every image stage is within 1.42 times the reference fit's error
on all eight families; every source-form criterion that can be asked trails by
at least 5.45 times; the two numeric integral forms are at parity with the
reference and tens of times cheaper per fit than the image at n = 200;
interval coverage orders image, direct integral forms, adequate-degree balance
at 0.883, 0.734 and 0.133. Streaming: the window image tracks an amplitude jump
at 0.018 of the recursive equal-areas form's RMSE. Map-reduce: the image merge
reproduces the whole-data fit to rounding, and the legacy accumulators reduce
16 to 48 times faster.

## 18 -- where the block basis applies

[18a_block_basis_synthetic](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/18a_block_basis_synthetic.ipynb)
and
[18b_block_basis_real_data](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/18b_block_basis_real_data.ipynb)
map the block basis by case.

| case | reading |
|---|---|
| jumps at known epochs | ahead: one window edge on each epoch lifts the jump columns to 0.989 to 0.991 at K=32 where Legendre is at 0.895; on 120 GPS station series the shifts are recovered to 0.032 standard errors against 0.393 (Legendre) and 0.435 (uniform windows) |
| jumps at unknown epochs | `fit_aligned` reaches 0.955 to 0.969 of its oracle where detection finds every epoch; on the archive, detection decides the answer and the basis does not |
| data aggregated per window | needed: `fit_aggregated` removes the midpoint reading's bias of -6.1 to +6.6 percent (under 0.053 percent after); on temperature read as 3 h and 6 h means it returns the hourly amplitudes to 1.0001 to 1.0048 where the midpoint reading returns 0.906 to 0.978 |
| a fixed budget of stored numbers on an irregular grid | ahead on a 10-cycle and a two-jump model, level on a plain decay |
| gapped records | behind: the block Gram stays under 200 where Legendre's passes 1e12, and the accuracy does not follow the conditioning -- block reads 1.05 to 3.19 times the pointwise RMSE where Legendre reads 1.00 |
| smooth records | behind on the cycle terms at equal K |

## 19 -- adaptations in trial

[19_adaptations_in_trial](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/experiments/method/19_adaptations_in_trial.ipynb)
holds each adaptation of `dtfit_experimental` to one rule: a win over the plain
route on its own signal class, outside the seed scatter, or parity where no
plain route exists. `InformationFilter` agrees with a single undivided pass to
7e-15 and is kept. Of the image bases, `FourierBasis` reaches a 1e-6
reconstruction at 9 coefficients against Legendre's 26 on a periodic signal;
`LaguerreBasis` needs 9 against 10 on a decay, which is parity. The API is on
[Adaptations API](Experimental-Adaptations-API).
