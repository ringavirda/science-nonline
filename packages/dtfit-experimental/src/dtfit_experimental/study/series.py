"""The twelve series the forecasting comparison runs on.

Three kinds of provenance, named per entry in :data:`SERIES`:

* measured, from a file under ``data_dir()`` -- USD/UAH, COVID-19 Ukraine,
  the LTSF sets;
* measured, from ``statsmodels.datasets`` -- sunspots, Mauna Loa CO2, Nile
  flow, El Nino SST, bundled with the package and never missing;
* generated from a governing equation plus measurement noise -- the RLC
  transient, the AC harmonics, the AM signal, the chirp, each seeded so a
  rerun reproduces the same array.

A loader whose file is not on disk returns ``None`` rather than raising, so a
notebook cell can skip its section on a missing value instead of failing the
whole run under ``nbclient``.
"""

from __future__ import annotations

import numpy as np

from . import datasets as ltsf
from .paths import data_dir


def _csv(name, col=1, start_row=1):
    import csv
    rows = list(csv.reader((data_dir() / name).open()))[start_row:]
    return np.array([float(r[col]) for r in rows])


def load_covid() -> np.ndarray | None:
    """COVID-19 Ukraine cumulative confirmed cases, 30 days from the first day
    the count reaches 500. Measured, from ``data_dir()/"covid_ukraine_confirmed.csv"``.

    Returns:
        The 30-day window as a 1-D array, or ``None`` when the CSV is absent.
        Shorter than 30 samples when the crossing falls within the last 29
        days of the file (the slice truncates silently at the file's end).

    Raises:
        StopIteration: the recorded series never reaches 500 cumulative
            cases.
    """
    try:
        cum = _csv("covid_ukraine_confirmed.csv")
    except FileNotFoundError:
        return None
    start = next(i for i, v in enumerate(cum) if v >= 500)
    return cum[start:start + 30]


def load_uah() -> np.ndarray | None:
    """USD/UAH daily official rate, 2014-2015 hryvnia crisis. Measured, from
    ``data_dir()/"usd_uah_2014_2015.csv"``.

    Returns:
        The daily rate as a 1-D array, or ``None`` when the CSV is absent.
    """
    try:
        return _csv("usd_uah_2014_2015.csv")
    except FileNotFoundError:
        return None


def load_sunspots() -> np.ndarray:
    """Yearly sunspot activity. Measured, from ``statsmodels.datasets``
    (bundled with the package, never missing)."""
    import statsmodels.api as sm
    return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)


def load_co2() -> np.ndarray:
    """Mauna Loa atmospheric CO2, 4-weekly (every 4th sample of the bundled
    weekly series, so 28 days apart). Measured, from
    ``statsmodels.datasets``."""
    import statsmodels.api as sm
    s = sm.datasets.co2.load_pandas().data["co2"]
    return s.interpolate().bfill().ffill().to_numpy(float)[::4]


def load_nile() -> np.ndarray:
    """Annual Nile river flow at Aswan. Measured, from
    ``statsmodels.datasets``."""
    import statsmodels.api as sm
    return sm.datasets.nile.load_pandas().data["volume"].to_numpy(float)


def load_elnino() -> np.ndarray:
    """Monthly El Nino sea-surface temperature, period 12. Measured, from
    ``statsmodels.datasets``."""
    import statsmodels.api as sm
    d = sm.datasets.elnino.load_pandas().data
    return d.iloc[:, 1:].to_numpy(float).ravel()        # monthly SST, period 12


def load_ltsf(name: str, channel: int = 0, tail: int = 1500) -> np.ndarray | None:
    """One channel of an LTSF benchmark series, its final ``tail`` samples.
    Measured, replayed from ``data_dir()/"ltsf"/<name>.csv``.

    Args:
        name: LTSF dataset key, one of :data:`dtfit_experimental.study.datasets.FILES`.
        channel: Column index into the dataset's numeric channels.
        tail: Number of trailing samples to keep.

    Returns:
        The channel's trailing ``tail`` samples as a 1-D array, or ``None``
        when the CSV is absent.

    Raises:
        KeyError: ``name`` is not one of :data:`dtfit_experimental.study.datasets.FILES`.
    """
    try:
        return ltsf.load(name)[-tail:, channel]
    except FileNotFoundError:
        return None


# The physics and signal-processing waveforms below are generated from their
# governing equations plus measurement noise, each seeded for reproducibility.
def _sig(seed, n, f):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    return t, f, rng


def load_rlc_transient() -> np.ndarray:
    """A damped oscillation: an RLC circuit or mechanical ring-down transient,
    y = e^{-sigma t}*sin(2 pi f t). dtfit's damped model is its exact structural
    form. Generated, seeded (reproducible)."""
    t, _, rng = _sig(11, 360, None)
    y = np.exp(-3.0 * t) * np.sin(2 * np.pi * 4.0 * t)
    return y + rng.normal(0, 0.02, t.size)


def load_ac_harmonics() -> np.ndarray:
    """AC mains-style waveform with harmonics: fundamental + 3rd + 5th (a
    distorted power-line / audio signal). Generated, seeded (reproducible)."""
    t, _, rng = _sig(12, 360, None)
    f = 6.0
    y = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 3 * f * t)
         + 0.15 * np.sin(2 * np.pi * 5 * f * t))
    return y + rng.normal(0, 0.03, t.size)


def load_am_signal() -> np.ndarray:
    """An amplitude-modulated carrier, (1 + m*cos 2 pi f_m t)*sin 2 pi f_c t: a
    communications signal or a vibration envelope. Generated, seeded
    (reproducible)."""
    t, _, rng = _sig(13, 400, None)
    y = (1 + 0.6 * np.cos(2 * np.pi * 1.5 * t)) * np.sin(2 * np.pi * 9.0 * t)
    return y + rng.normal(0, 0.03, t.size)


def load_chirp() -> np.ndarray:
    """A linear chirp: the frequency sweep sin(2 pi(f0 + k t)t) of radar and
    sonar. Its instantaneous frequency changes, so a fixed-frequency fit is
    honestly hard. Generated, seeded (reproducible)."""
    t, _, rng = _sig(14, 400, None)
    inst = 2.0 + 6.0 * t
    y = np.sin(2 * np.pi * inst * t)
    return y + rng.normal(0, 0.03, t.size)


#: Series config: (loader, trend kind, seasonal?, period in samples, label).
#: The trend kind names the dtfit model fitted to that series, independent of
#: the ``seasonal?`` and period fields, which configure only the baselines
#: (seasonal naive, ETS, SARIMA).
#: trend kinds:
#:   exp                a*e^{bx}                      pure exponential growth
#:   logistic           L/(1+e^{-k(x-x0)})           saturating (epidemic/diffusion)
#:   linear             a0+a1*x                       a level with a slope
#:   linear_wave        a0+a1*x+a2*sin+a3*cos         linear + one slow cycle
#:   poly               a0+a1*x+a2*x^2                a smooth (accelerating) trend
#:   poly_seasonal      poly + A*sin(w*x+p)           trend + one seasonal cycle (joint)
#:   linear_seasonal    linear + A*sin(w*x+p)         level/slope + one cycle (joint)
#:   sine               c + A*sin(w*x+p)              a level + a single cycle
#:   damped             A*e^{-zwx}*sin(...)           ring-down transient
#:   fourier_series     c + sum a_k sin + b_k cos     fundamental + harmonics
#:   am / chirp         modulated carrier / sweep
SERIES = [
    ("COVID-19 UA", load_covid, "logistic", False, None, "epidemic growth"),
    ("USD/UAH", load_uah, "linear_wave", False, None, "currency depreciation"),
    ("Sunspots", load_sunspots, "sine", True, 11, "solar ~11y cycle"),
    ("Mauna Loa CO2", load_co2, "poly_seasonal", True, 12, "climate trend+season"),
    ("El Nino SST", load_elnino, "linear_seasonal", True, 12, "ocean seasonal"),
    ("Nile flow", load_nile, "poly", False, None, "hydrology level"),
    ("ETTh1 oil-temp", lambda: load_ltsf("ETTh1", -1), "linear_seasonal", True, 24,
     "transformer temp"),
    ("Weather LTSF", lambda: load_ltsf("weather", 0), "transient_seasonal", True,
     144, "weather sensor"),
    ("RLC transient", load_rlc_transient, "damped", False, None,
     "physics: electrical ring-down"),
    ("AC + harmonics", load_ac_harmonics, "fourier_series", True, 60,
     "physics: power waveform"),
    ("AM signal", load_am_signal, "am", False, None,
     "physics: modulated carrier"),
    ("Linear chirp", load_chirp, "chirp", False, None,
     "physics: frequency sweep"),
]
