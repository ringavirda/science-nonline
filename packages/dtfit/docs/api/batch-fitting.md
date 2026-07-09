# Batch fitting

The differential-transformation batch fitters and the self-describing result
type they return. Start with `fit_lsi` (the accurate general default); switch to
`fit_eac` when the data are noisy or you need speed; reach for `ensemble_fit`
when outliers contaminate the record. See the
[Guide](../guide/choosing-a-method.md) for the decision tree.

All three accept the model as a SymPy string, a `sympy.Expr`, or a Python
callable `f(x, *params)` -- resolved by `resolve_model`.

::: dtfit.fit_lsi

::: dtfit.fit_eac

::: dtfit.ensemble_fit

::: dtfit.EnsembleResult

::: dtfit.FittingResult

::: dtfit.methods.resolve_model
