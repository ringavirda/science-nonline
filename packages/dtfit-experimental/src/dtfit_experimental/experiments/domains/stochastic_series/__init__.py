"""Stochastic-series domain validation.

Asks whether dtfit's deterministic fitters can be used on *random* series
(economic / financial data) by fitting the **deterministic functionals** of a
stochastic process -- its autocovariance, spectrum, aggregated variance and
trend/cycle -- and recovering the process's parameters from them.

``backend.py`` is the single source of truth: ground-truth data generators
(ARFIMA, AR(1)/OU, GARCH(1,1), AR(2) pseudo-cycle, trend+cycle) plus the
evaluation harness ``run()`` / ``summary()`` that scores each possibility
(VIABLE / MARGINAL / NOT VIABLE) by parameter-recovery error against the known
truth. The reusable estimators it exercises now live in the promoted stable
:mod:`dtfit.stochastic` (``fit_stochastic`` / ``StochasticModel`` /
``StochasticFilter`` / ``dtfit.Stochastic``).
"""
