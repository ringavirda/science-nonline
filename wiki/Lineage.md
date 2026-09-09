# Lineage

dtfit's estimator is projected weighted nonlinear least squares on the
image, one function with two bases. It grew out of the
differential-transformation criteria in four stages, each fixing a limit
of the one before. Every stage is runnable in the `dtfit-legacy` package
so the evolution can be measured on one problem set.

| Stage | What it does | Its limit | Where |
|---|---|---|---|
| Spectrum balance | equates the model's Maclaurin spectrum to the data's, discrete by discrete, from an intermediate polynomial | a symbolic model, an exactly determined balance, no noise averaging | `dtfit_legacy.book.dsb_balance`, `dtfit_legacy.dsb.fit_dsb` |
| Integral least squares, monomial basis | minimizes the integral of the squared residual with both sides in the monomial spectrum | the Hilbert-like weight matrix, condition above 1e16 by order 11 | `dtfit_legacy.book.lsi_integral_monomial` |
| Integral least squares, Legendre basis | the same criterion on the orthogonal basis; condition linear in the order | the integral quadrature carries grid-dependent bias | `dtfit_legacy.integral.fit_lsi` |
| Equal areas | one area equation per window, the model's areas from its truncated spectrum | bias outside the spectrum's convergence radius | `dtfit_legacy.book.eac_areas`, direct form `dtfit_legacy.integral.fit_eac` |
| The image | the discrete sufficient statistic `S = Phi^T (w y)`, `G = Phi^T diag(w) Phi`; fit by whitened NLLS | needs a start for a model nonlinear in its parameters | `dtfit.fit`, `dtfit.ImageFilter` |

The recursive filters of the integral criteria are `dtfit_legacy.streaming`;
their map-reduce form is `dtfit_legacy.scale`. The current filter and
stream are `dtfit.ImageFilter` and `dtfit.ImageStream`.
