"""dtfit quickstart -- your first fit in a minute.

dtfit fits models that are *nonlinear in their parameters* (exponential,
transcendental, oscillatory, mixed). You bring a model as a small sympy
expression string (e.g. "a*exp(b*t)") and your data; dtfit recovers the
parameters and returns a self-describing FittingResult.

Run headless:   python examples/01_quickstart.py
"""

import numpy as np

from dtfit import fit_lsi, auto_estimate


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 200)
    y = 1.4 * np.exp(0.8 * x) + rng.normal(0, 0.15, x.size)

    # 1. A first fit. Everything in the expression except the variable "t" is a
    #    free parameter -- here a and b.
    res = fit_lsi(x, y, "a*exp(b*t)", "t")
    print("== fit_lsi: a*exp(b*t) ==")
    print(res.summary())
    print("params:", {k: round(v, 4) for k, v in res.params.items()})

    # 2. Uncertainty: an overdetermined fit carries a parameter covariance, so it
    #    reports standard errors, confidence intervals and a prediction band.
    print("\n== uncertainty ==")
    print("stderr:", {k: round(v, 4) for k, v in res.stderr().items()})
    print("95% CI:", {k: tuple(round(b, 3) for b in ci)
                      for k, ci in res.confidence_intervals().items()})
    xs = np.linspace(x.min(), x.max(), 5)
    y_hat, y_sd = res.predict(xs, return_std=True)
    print("predict(return_std) sd:", np.round(y_sd, 4))

    # 3. Don't want to choose the estimator? auto_estimate routes by signal shape.
    res2 = auto_estimate(x, y, "a*exp(b*t)", "t")
    print("\n== auto_estimate ==")
    print("params:", {k: round(v, 4) for k, v in res2.params.items()})


if __name__ == "__main__":
    main()
