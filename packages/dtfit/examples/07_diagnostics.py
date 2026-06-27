"""Diagnostics, serialization and logging.

Evaluate a fitted dtfit model: information criteria, residual-structure tests,
ready-made plots, opt-in logging, and round-trip serialization. (For plain scalar
metrics on arrays, use sklearn.metrics / scipy.stats directly.)

Run headless:        python examples/07_diagnostics.py
Show the plots too:  python examples/07_diagnostics.py --plot   (needs the viz extra)
"""

import sys

import numpy as np

from dtfit import fit_lsi, FittingResult


def main() -> None:
    rng = np.random.default_rng(2)
    x = np.linspace(0, 4, 250)
    y = 0.5 + 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.2, x.size)
    res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x")

    # fit_report -- sample/param counts, RSS, RMSE, r2, AIC/BIC, Durbin-Watson.
    from dtfit.diagnostics import fit_report, residual_diagnostics

    rep = fit_report(res, x, y)
    print("== fit_report ==")
    for k in ("n", "rmse", "r2", "aic", "bic", "durbin_watson", "converged"):
        if k in rep:
            print("  {:14s}: {}".format(k, round(rep[k], 4)
                                        if isinstance(rep[k], float) else rep[k]))

    # residual_diagnostics -- autocorrelation / normality of the residuals.
    rd = residual_diagnostics(res, x, y)
    print("\n== residual_diagnostics ==")
    print("  durbin_watson :", round(rd["durbin_watson"], 3))
    print("  lag1_autocorr :", round(rd["lag1_autocorr"], 3))
    print("  normality_p   :", round(rd["normality_p"], 3))

    # Serialize -- everything needed to rebuild the model round-trips through a
    # JSON-friendly dict.
    blob = res.to_dict()
    restored = FittingResult.from_dict(blob)
    print("\n== to_dict / from_dict round-trip ==")
    print("  params:", {k: round(v, 3) for k, v in restored.params.items()})

    # Opt-in logging -- dtfit logs under the "dtfit" logger with a NullHandler by
    # default; enable_logging(DEBUG) surfaces the fitting internals.
    import logging
    from dtfit import enable_logging

    enable_logging(logging.WARNING)   # quiet here; use DEBUG to see fit detail
    logging.getLogger("dtfit").handlers = [logging.NullHandler()]

    # Optional plots (scikit-learn-style Display objects; need matplotlib).
    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt
        from dtfit.diagnostics import FitDisplay, ResidualsDisplay

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        FitDisplay.from_predictions(x, y, res.predict(x), ax=axes[0],
                                    estimator_name="LSI")
        ResidualsDisplay.from_predictions(y, res.predict(x), ax=axes[1],
                                          estimator_name="LSI")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
