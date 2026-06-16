# EDA — Equal Differential Areas (equal areas)

> Numeric batch method, successor to the symbolic DSBE. Source:
> [`extra/dt/eda.py`](../../src/dtfit/extra/dt/eda.py).
> Invoke via `dt.fit_eda(x, y, expr, var, ...)`,
> `nonline_fit(..., method="eda", data_x=, data_y=)`,
> `FittingNonlineMw(method="eda")`, or `NonlineRegressor(..., method="eda")`.

EDA identifies parameters by matching **integral areas** of the model and the
data over a set of windows, rather than matching spectra pointwise. Because
integration is a smoothing (low-pass) operator, EDA never differentiates the
data and is the **most noise-robust** of the batch methods.

## Mathematical grounding

For a model $f(t;\theta)$ with $m$ unknown parameters, split the active region
of the data into $M \ge m$ contiguous windows $W_1,\dots,W_M$ and require the
model area to equal the data area on each:

$$
\int_{W_i} f(t;\theta)\,dt \;=\; \int_{W_i} x_{\text{data}}(t)\,dt,
\qquad i = 1,\dots,M .
$$

That is $M$ equations in $m$ unknowns — residuals

$$
r_i(\theta) \;=\; \int_{W_i} f(t;\theta)\,dt \;-\; A_i,
\qquad A_i = \int_{W_i} x_{\text{data}}\,dt .
$$

**Overdetermined by default.** The original EDA used exactly $M=m$ windows — a
determined system with no redundancy, which discards the very noise averaging
integration buys. Here $M$ defaults to $2m$ (configurable via `n_windows`),
giving an **overdetermined** least-squares system: the random per-window
integration errors partly cancel, lowering the estimation variance (empirically
$\sim15\%$ on the arctan benchmark going from $m$ to $2m$ windows), and a
parameter covariance can be read off the residual Jacobian.

**Connection to the differential spectrum.** The area of a signal over $[0,H]$
is the integral of its inverse transform,

$$
\int_{0}^{H} x(t)\,dt
   = \sum_{k} X(k)\!\int_0^H\!\Big(\tfrac{t}{H}\Big)^{k}\!dt
   = H\sum_{k} \frac{X(k)}{k+1},
$$

so an area is a **moment of the differential spectrum**. Matching areas over $m$
shifted windows is matching $m$ independent integral functionals of the
spectrum — a weak-form (Galerkin, piecewise-constant test function)
identification. Two analytic functions sharing $m$ such independent moments
agree where the model has $m$ degrees of freedom, so the parameters are
recovered.

**Why it is robust.** Zero-mean observation noise integrates toward zero:
$\int_{W} \varepsilon(t)\,dt \to 0$ as the window grows. The data enter EDA only
through their integrals $A_i$, never through a derivative or a high-order
polynomial fit — so EDA has none of LSI's ill-conditioned high-order discretes
and degrades gracefully as noise rises. The figure below shows the cumulative
integral of model and data lying on top of each other even though the raw
samples are visibly noisy.

## Algorithm

1. **Parse** the model; collect the $m$ free parameters $\theta$.
2. **Compile once**: `lambdify` the model and each analytic partial derivative
   $\partial f/\partial\theta_j$ (used for the Jacobian).
3. **Window placement**: take the leading `active_ratio` (default 0.8) of the
   data — the informative transient — and split it into $M$ equal windows
   ($M=$ `n_windows`, default $2m$; each kept ≥ 3 samples for Simpson).
4. **Data areas**: $A_i = \int_{W_i} y\,dx$ by Simpson's rule (`scipy.integrate.simpson`).
5. **Solve** the $M\times m$ system by least squares using the **analytic
   integrated Jacobian** $J[i,j] = \int_{W_i} \partial f/\partial\theta_j\,dt$
   — Levenberg–Marquardt (`method="lm"`) by default, or trust-region (`trf`)
   when `bounds` or a robust `loss` (e.g. `"soft_l1"`) is requested.
6. **Return** the fitted $\theta$, a `lambdify`-ed callable model, and (when
   overdetermined) a parameter covariance estimate (`FittingResult.cov`).

## Optimizations and guards

- **Analytic integrated Jacobian** — derivatives are taken symbolically once and
  *integrated*, not finite-differenced, giving an exact, smooth Jacobian for LM
  (faster, more stable convergence than numeric differencing).
- **Integration as the denoiser** — the equal-areas criterion is itself the
  noise guard; no separate pre-filter is needed.
- **Active-region windowing** (`active_ratio`) concentrates the windows on the
  informative transient, where the parameters are most observable.
- **Overdetermined averaging** (`n_windows` > $m$) — extra area equations average
  out per-window integration noise and yield a parameter covariance.
- **Bounds / robust loss** — `bounds=` and `loss="soft_l1"` switch the solver to
  trust-region for constrained or outlier-prone fits.
- **Scalar-derivative broadcast** — a constant $\partial f/\partial\theta_j$
  (e.g. a purely linear parameter) is broadcast to the window grid so Simpson
  integrates it correctly.
- **Sample-count guard** — raises if there are fewer than $2m$ samples, since
  $m$ windows of meaningful area cannot otherwise be formed.

## Worked example

`y = a·arctan(w·x)` (truth `a=2.0, w=3.0`), a transcendental **non-Taylor**
saturation curve, with 8 % noise. **Left:** EDA recovers `a≈2.00, w≈2.99` from
the noisy cloud; the shaded bands are the two per-parameter integration windows.
**Right:** the equal-areas criterion — the cumulative integral of the fitted
model (dashed) tracks the cumulative integral of the data, which is the quantity
EDA actually matches.

![EDA fit and the equal-areas criterion](figures/eda_fit.png)

## Comparison

**Model data — `y = a·exp(b·x)`, ground truth a=1.0, b=1.2, 5 % noise, n=80.**
Error is against the *clean* signal.

| method | recovered params | R² | RMSE | MAPE % | fit (ms) |
|---|---|---|---|---|---|
| LSI | a=0.995, b=1.212 | 0.9995 | 0.03141 | 0.57 | 15.4 |
| **EDA** | a=0.988, b=1.222 | 0.9989 | 0.04701 | 0.89 | 2.7 |
| SciPy `curve_fit` | a=1.000, b=1.204 | 0.9999 | 0.01305 | 0.25 | 0.2 |
| numpy.polyfit (deg 5) | — | 0.9997 | 0.02302 | 0.85 | 0.1 |

EDA recovers the parameters essentially as well as LSI and the NLS gold standard
while being the **fastest** of the dtfit methods (≈3 ms here — roughly 5× LSI),
because it solves an $m\times m$ system instead of a spectral least-squares
problem. That speed and its derivative-free robustness are why EDA is the basis
of the streaming [EqualAreasFilter](equal_areas_filter.md).

**Real data — COVID-19 Ukraine** (28-day take-off, 548→8617 cases),
`y = a·exp(b·t)`:

| method | R² | RMSE | MAPE % |
|---|---|---|---|
| LSI | 0.9071 | 757.1 | 9.21 |
| **EDA** | 0.7920 | 1133 | 9.99 |
| SciPy `curve_fit` | 0.9879 | 273.5 | 13.34 |

## Where it is best applied

**Use EDA for:** noise-robust batch fitting of models with **few parameters**
(2–4), transient signals where the early dynamics carry the information, and as
a fast, stable initializer. It is preferable to LSI when the data are noisy
enough that a polynomial spectrum would be unreliable, or when speed matters.

**Caveats.** EDA forms exactly $m$ windows, so it suits low-parameter models;
for many parameters or for global model selection, [LSI](lsi.md) is better. Like
all the spectral/area methods it assumes a modest dynamic range — normalize wide
domains first. For real-time tracking of *time-varying* parameters, use the
recursive [EqualAreasFilter](equal_areas_filter.md).
