"""Backend infrastructure for the real-world forecasting experiment.

This module is the **single source of truth for the data, forecasters and
train/holdout evaluation** behind ``04_realworld_forecasting.ipynb``; the
notebook imports it and does all the presentation (tables, figures, narrative).
Keeping the infra here means the loaders/models/scoring are defined once and the
notebook stays a thin, rerunnable layer over them.

The study fits a structured dtfit model on the first 80% of four real series with
distinct structure -- exponential growth (COVID-19 Ukraine), exponential
depreciation (USD/UAH), an ~11-year cycle (sunspots) and trend+seasonality
(Mauna Loa CO2) -- and forecasts the last 20%, comparing against the standard
forecasters that can be run fairly (ARIMA, a scikit-learn MLP, a PyTorch LSTM and
the random-walk benchmark).

It provides:

* the **data loaders** -> a 1-D series -- :func:`load_covid`, :func:`load_uah`,
  :func:`load_sunspots`, :func:`load_co2`;
* the **dtfit parametric forecasters** (fit train, predict full ``t``) --
  :func:`dtfit_exp`, :func:`dtfit_sunspots`, :func:`dtfit_co2`, and the
  :data:`DATASETS` registry binding each series to its loader / model / ARIMA
  order;
* the **train/holdout evaluation** -- :func:`run_one`, which fits every method on
  the 80% train split, forecasts the 20% holdout, and returns the series, split,
  predictions and per-method scores as plain numbers/arrays/dicts.

dtfit is a *parametric* fit-then-extrapolate forecaster, so it shines where the
series has clear nonlinear structure to extrapolate, while the general learners
shine on complex/irregular dynamics. The evaluation reports it honestly.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit_experimental import fit_lsi_basis, boosted_fit

from dtfit_experimental.experiments.common import EXPERIMENTS_DIR, metrics
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "load_covid", "load_uah", "load_sunspots", "load_co2",
    "dtfit_exp", "dtfit_sunspots", "dtfit_co2",
    "DATASETS", "run_one",
]


# --------------------------------------------------------------------------- #
# data loaders -> a 1-D series
# --------------------------------------------------------------------------- #
def load_covid():
    import csv
    p = EXPERIMENTS_DIR / "data" / "covid_ukraine_confirmed.csv"
    rows = list(csv.reader(p.open()))[1:]
    cum = np.array([float(r[1]) for r in rows])
    start = next(i for i, v in enumerate(cum) if v >= 500)
    return cum[start:start + 28]  # clean exponential take-off window


def load_uah():
    import csv
    p = EXPERIMENTS_DIR / "data" / "usd_uah_2014_2015.csv"
    rows = list(csv.reader(p.open()))[1:]
    return np.array([float(r[1]) for r in rows])


def load_sunspots():
    import statsmodels.api as sm
    return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)


def load_co2():
    import statsmodels.api as sm
    s = sm.datasets.co2.load_pandas().data["co2"]
    s = s.interpolate().bfill().ffill()
    return s.to_numpy(float)[::4]  # thin weekly->~monthly for a shorter series


# --------------------------------------------------------------------------- #
# dtfit parametric forecasters per dataset (fit train, predict full t)
# --------------------------------------------------------------------------- #
def dtfit_exp(t_tr, y_tr, t_all):
    y0 = y_tr[0]
    r = dt.fit_lsi(t_tr, y_tr / y0, "a*exp(b*x)", "x", bounds=[(0.1, 5), (0.05, 5)])
    return np.asarray(r.model(t_all)) * y0, "LSI exp"


def dtfit_sunspots(t_tr, y_tr, t_all):
    # cyclic: c + A sin(w t + p), fitted via Fourier-basis LSI (adaptation #2)
    expr = "c + A*sin(w*x + p)"
    r = fit_lsi_basis(t_tr, y_tr, expr, "x", basis="fourier", order=8,
                      bounds=[(10, 120), (0, 200), (0.1, 1.5), (-np.pi, np.pi)])
    return np.asarray(r.model(t_all)), "Fourier-LSI (#2)"


def dtfit_co2(t_tr, y_tr, t_all):
    # trend + seasonal via stage-wise boosting (adaptation #5)
    bm = boosted_fit(t_tr, y_tr, [
        dict(expr="a0 + a1*x + a2*x**2", var="x", method="lsi",
             p0=[y_tr[0], 1.0, 0.0]),
        dict(expr="A*sin(w*x + p)", var="x", method="lsi",
             bounds=[(0.1, 20), (0.1, 60), (-np.pi, np.pi)]),
    ])
    return bm.predict(t_all), "boosted LSI (#5)"


DATASETS = {
    "COVID-19 UA (exp growth)": (load_covid, dtfit_exp, dict(order=(2, 2, 2))),
    "USD/UAH (exp depreciation)": (load_uah, dtfit_exp, dict(order=(2, 1, 2))),
    "Sunspots (~11y cycle)": (load_sunspots, dtfit_sunspots, dict(order=(3, 0, 3))),
    "Mauna Loa CO2 (trend+season)": (load_co2, dtfit_co2, dict(order=(2, 1, 2))),
}


def run_one(name, loader, dtfit_fn, arima_kw, *, quick=True):
    """Fit every method on the 80% train split and forecast the 20% holdout.

    Returns a dict with the series ``y`` / time axis ``t`` / split index ``n_tr``,
    the per-method holdout ``preds``, the per-method ``scores`` (R2/RMSE/MAE/MAPE),
    and the ``dtfit_label`` for the chosen parametric form.

    ``quick`` trims the heavy learners so the whole suite runs in a couple of
    minutes: the MLP runs fewer iterations and the (slow) PyTorch **LSTM is
    skipped**. Set ``quick=False`` to run the MLP to convergence and add the LSTM.
    Any baseline whose optional dependency (statsmodels / sklearn / torch) is
    missing is skipped gracefully (its forecast becomes NaN)."""
    y = loader()
    n = y.size
    n_tr = int(n * 0.8)
    h = n - n_tr
    t = np.linspace(0, 1.5, n)
    t_tr, y_tr = t[:n_tr], y[:n_tr]

    preds = {}
    try:
        full, label = dtfit_fn(t_tr, y_tr, t)
        preds[label] = full[n_tr:]
    except Exception:
        preds["dtfit (failed)"] = np.full(h, np.nan)
        label = "dtfit (failed)"
    # baselines forecast the next h points from the training array
    try:
        preds["ARIMA"] = bl.arima_forecast(y_tr, h, order=arima_kw["order"])
    except Exception:
        preds["ARIMA"] = np.full(h, np.nan)
    try:
        preds["MLP"] = bl.mlp_forecast(y_tr, h, lookback=min(24, n_tr // 3),
                                       max_iter=600 if quick else 1500)
    except Exception:
        preds["MLP"] = np.full(h, np.nan)
    if not quick:
        try:
            preds["LSTM"] = bl.lstm_forecast(y_tr, h, lookback=min(24, n_tr // 3),
                                             epochs=150)
        except Exception:
            preds["LSTM"] = np.full(h, np.nan)
    preds["random walk"] = bl.random_walk_forecast(y_tr, h)

    y_te = y[n_tr:]
    scores = {m: metrics(y_te, p) for m, p in preds.items()}
    return dict(name=name, y=y, t=t, n_tr=n_tr, preds=preds, scores=scores,
                dtfit_label=label)
