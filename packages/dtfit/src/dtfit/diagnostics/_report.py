"""Fit-aware diagnostics for a :class:`dtfit.FittingResult`.

Unlike the generic ``(y_true, y_pred)`` scalar metrics in ``sklearn.metrics`` /
``scipy.stats`` (use those for plain numbers), these are **specific to evaluating
a fitted dtfit model**: they take the :class:`FittingResult` itself so they can
report parameter uncertainty, information criteria for *model comparison* (the
"which model" question the domain studies turn on), and whether the residuals
still carry structure the model missed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dtfit.methods._common import information_criteria


def _basic_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """``r2`` and ``rmse`` of a prediction (used for plot annotations)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    resid = y_true - y_pred
    rss = float(resid @ resid)
    tss = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(rss / y_true.size)),
        "r2": float(1.0 - rss / tss) if tss > 0 else float("nan"),
    }


def fit_report(result: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Goodness-of-fit + parsimony report for a fitted model on ``(x, y)``.

    Returns a dict with sample/parameter counts, ``rss``/``rmse``/``r2``, the
    **AIC and BIC** (Gaussian-likelihood, for comparing candidate models on the
    same data), the Durbin-Watson statistic (≈2 = no residual autocorrelation),
    and -- when the fit carries a covariance -- the parameter values and standard
    errors. AIC/BIC make this the building block for model selection: fit several
    candidates and keep the lowest IC.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(result.predict(x), dtype=float).ravel()
    resid = y - yhat
    n = y.size
    k = int(np.asarray(result.coeffs).size)
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    rmse = float(np.sqrt(rss / n))
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
    aic, bic = information_criteria(rss, n, k)
    dw = float(np.sum(np.diff(resid) ** 2) / rss) if rss > 0 else float("nan")
    report: dict[str, Any] = {
        "n": n, "n_params": k, "rss": rss, "rmse": rmse, "r2": r2,
        "aic": aic, "bic": bic, "durbin_watson": dw,
    }
    converged = getattr(result, "converged", None)
    if converged is not None:
        report["converged"] = bool(converged)
    if getattr(result, "cov", None) is not None:
        report["params"] = result.params
        report["stderr"] = result.stderr()
    return report


def residual_diagnostics(result: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Tests for **structure the model left in the residuals**.

    A structured (DT) fit should leave white-noise residuals; leftover
    autocorrelation means the model class is wrong (e.g. a trend with no cycle on
    a seasonal series). Returns the residuals plus the Durbin-Watson and lag-1
    autocorrelation statistics and a normality p-value (Shapiro-Wilk).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    resid = y - np.asarray(result.predict(x), dtype=float).ravel()
    n = resid.size
    rss = float(resid @ resid)
    dw = float(np.sum(np.diff(resid) ** 2) / rss) if rss > 0 else float("nan")
    lag1 = (float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
            if n > 2 and resid.std() > 0 else float("nan"))
    p_norm = float("nan")
    if 3 <= n <= 5000:
        try:
            from scipy.stats import shapiro

            p_norm = float(shapiro(resid).pvalue)
        except Exception:
            pass
    return {
        "residuals": resid,
        "durbin_watson": dw,
        "lag1_autocorr": lag1,
        "normality_p": p_norm,
        "mean": float(resid.mean()),
        "std": float(resid.std()),
    }
