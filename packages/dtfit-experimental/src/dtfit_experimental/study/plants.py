"""The embedded real-time control plants and their estimator adapters.

Shared infrastructure for the filter notebooks: four physical signal
families a control or monitoring loop actually sees, a uniform adapter
around every estimator that could run online on one, and the multi-axis
fault-detection rig built on dtfit's fused ``FilterBank``. Pure compute,
with no ``matplotlib``: a notebook drives these and does the presentation.

* the plant model families, :data:`PLANTS` and :func:`gen_plant`;
* the uniform estimator adapters :class:`EAAd`, :class:`EARobAd`,
  :class:`LegAd`, :class:`LegRobAd`, :class:`EKFAd`, :class:`RLSAd` and
  :class:`RefitAd`, driven one sample at a time by :func:`drive`, scored by
  :func:`perr`;
* the model-mismatch negative control, :data:`MISMATCH_PAIRS` and
  :func:`mismatch_scores`: an estimator configured for the wrong plant,
  driven over another plant's stream;
* the multi-axis fault detection, :func:`make_multi`, :class:`MergedTracker`,
  :func:`run_tracker` and :func:`kalman_multi`, on a fused chi-square
  detector and ``inflate``.
"""

from __future__ import annotations

import time

import numpy as np

from dtfit.streaming import ImageFilter

from dtfit_experimental.streaming import FilterBank

from dtfit_experimental.study import baselines as bl
from dtfit_experimental.study.baselines import KalmanCA, EKFParam, RLSPredictor

__all__ = [
    "OSC", "PLANTS",
    "gen_plant",
    "EAAd", "EARobAd", "LegAd", "LegRobAd", "EKFAd", "RLSAd", "RefitAd",
    "perr", "drive", "adapters",
    "MISMATCH_PAIRS", "MISMATCH_CEILING", "mismatch_scores",
    "make_multi", "MergedTracker", "run_tracker", "kalman_multi",
]

OSC = "A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)"


# the plant model families, each a real embedded signal class
def _f_damped(t, A, w, z):
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t)


def _f_acsine(t, A, c, w):
    return c + A * np.sin(w * t)


def _f_firstorder(t, K, tau):
    return K * (1 - np.exp(-t / tau))


def _f_catraj(t, c0, c1, c2):
    return c0 + c1 * t + c2 * t ** 2


PLANTS = [
    dict(key="damped_osc", app="control / vibration ID", shape="oscillatory",
         expr=OSC, func=_f_damped, true={"A": 2.0, "w": 2.5, "z": 0.12},
         p0=[2.0, 2.0, 0.1], bounds=([0.1, 1, 0.01], [5, 6, 0.9]),
         T=12, n=700, window=60, q=[1e-3, 1e-3, 1e-3]),
    dict(key="ac_sine", app="AC / power monitoring", shape="sustained cycle",
         expr="c + A*sin(w*t)", func=_f_acsine, true={"A": 2.0, "c": 0.5, "w": 2.5},
         p0=[1.5, 0.5, 2.0], bounds=([0.1, -2, 1], [5, 3, 6]),
         T=12, n=700, window=60, q=[1e-3, 1e-3, 1e-3]),
    dict(key="first_order", app="RC / thermal / DC-motor", shape="monotone",
         expr="K*(1-exp(-t/tau))", func=_f_firstorder, true={"K": 3.0, "tau": 1.2},
         p0=[1.0, 1.0], bounds=([0.1, 0.05], [10, 5]),
         T=6, n=600, window=50, q=[1e-2, 1e-2]),
    dict(key="ca_traj", app="GPS / inertial trajectory", shape="polynomial",
         expr="c0 + c1*t + c2*t**2", func=_f_catraj,
         true={"c0": 1.0, "c1": 2.0, "c2": 0.5}, p0=[0.0, 0.0, 0.0],
         bounds=([-10, -10, -10], [10, 10, 10]),
         T=6, n=600, window=40, q=[1e-2, 1e-2, 1e-2]),
]


def gen_plant(plant, rng, *, noise=0.05, outliers=0.0, drop=0.0):
    """One noisy real-time stream from a plant.

    Args:
        plant: An entry of :data:`PLANTS`.
        rng: A ``numpy.random.Generator``.
        noise: Gaussian measurement noise, as a fraction of the clean
            signal's standard deviation.
        outliers: Fraction of samples (0 to 1) that additionally get a gross
            spike (sensor fault, multipath) added on top of the noisy value,
            drawn at 8 times the clean signal's standard deviation (not the
            noise scale: at ``noise=0.05`` this is about 160 times the noise
            standard deviation).
        drop: Fraction of samples (0 to 1) removed after noise and outliers,
            leaving an irregularly-sampled stream.

    Returns:
        ``(t, y, clean)``: the time grid, the noisy observation and the
        clean signal, all the same length (shorter than ``plant["n"]``
        when ``drop`` thins the grid).
    """
    t = np.linspace(0, plant["T"], plant["n"])
    clean = plant["func"](t, *[plant["true"][k] for k in sorted(plant["true"])])
    scale = clean.std() + 1e-9
    y = clean + rng.normal(0, noise * scale, t.size)
    if outliers > 0:
        m = rng.random(t.size) < outliers
        y[m] += rng.normal(0, 8 * scale, int(m.sum()))
    if drop > 0:
        keep = np.sort(rng.choice(t.size, int(t.size * (1 - drop)), replace=False))
        t, y, clean = t[keep], y[keep], clean[keep]
    return t, y, clean


# the uniform estimator adapters, one online step at a time
class _Ad:
    """Common adapter surface: ``step(t, y)`` ingests one sample,
    ``predict(t)`` returns the current model value at ``t``, ``params()``
    the current physical-parameter estimate (``None`` for a black-box
    method that recovers none), and ``gives_params`` whether it does."""

    gives_params = True

    def params(self):
        return None


class EAAd(_Ad):
    """dtfit's block-basis streaming filter (the window's equal-area image).

    Args:
        plant: An entry of :data:`PLANTS`.
        window: The window cap ``W`` in samples, or ``None`` (default) to
            use ``plant["window"]``. ``0`` is falsy and also falls back to
            the plant default, it does not select a zero-length window.
        adaptive_window: Passed through to :class:`~dtfit.streaming.ImageFilter`:
            size the window from the data instead of holding it at
            ``window``. ``True`` by default, so the adaptive- and
            fixed-window rows of a notebook must pass ``False`` explicitly
            for the fixed case.
    """

    name = "dtfit block filter"

    def __init__(self, plant, window=None, adaptive_window=True):
        self.f = ImageFilter(plant["expr"], "t", p0=list(plant["p0"]),
                           window_size=window or plant["window"],
                           order=len(plant["p0"]), q_diag=list(plant["q"]),
                           basis="block", adaptive_window=adaptive_window)

    def step(self, t, y):
        self.f.partial_fit(t, y)

    def predict(self, t):
        return float(self.f.predict(np.array([t]))[0]) if len(self.f._t) else np.nan

    def params(self):
        return dict(self.f.params_)


class EARobAd(EAAd):
    """The block filter on the robust window image (Huber reweighting of
    the window's basis regression), the mechanism that replaced residual
    truncation.

    Args: as :class:`EAAd` (``plant``, ``window``, ``adaptive_window``).
    """
    name = "dtfit block filter (robust image)"

    def __init__(self, plant, window=None, adaptive_window=True):
        self.f = ImageFilter(plant["expr"], "t", p0=list(plant["p0"]),
                           window_size=window or plant["window"],
                           order=len(plant["p0"]), q_diag=list(plant["q"]),
                           basis="block", adaptive_window=adaptive_window,
                           robust=True)


class LegAd(_Ad):
    """dtfit's Legendre-spectrum streaming filter.

    Args:
        plant: An entry of :data:`PLANTS`.
        window: The window cap ``W`` in samples, or ``None`` (default) to
            use ``plant["window"]``. ``0`` is falsy and also falls back to
            the plant default, it does not select a zero-length window.
        adaptive_window: Passed through to :class:`~dtfit.streaming.ImageFilter`:
            size the window from the data instead of holding it at
            ``window``. ``True`` by default, so the adaptive- and
            fixed-window rows of a notebook must pass ``False`` explicitly
            for the fixed case. Unlike :class:`EAAd`, the Legendre order is
            fixed at 5 regardless of the plant's parameter count, since the
            spectrum basis does not need order == len(p0) to be identifiable.
    """

    name = "dtfit Legendre filter"

    def __init__(self, plant, window=None, adaptive_window=True):
        self.f = ImageFilter(plant["expr"], "t", p0=list(plant["p0"]),
                           window_size=window or plant["window"],
                           order=5, q_diag=list(plant["q"]), basis="legendre",
                           adaptive_window=adaptive_window)

    def step(self, t, y):
        self.f.partial_fit(t, y)

    def predict(self, t):
        return float(self.f.predict(np.array([t]))[0]) if len(self.f._t) else np.nan

    def params(self):
        return dict(self.f.params_)


class LegRobAd(LegAd):
    """The Legendre filter on the robust window image.

    Args: as :class:`LegAd` (``plant``, ``window``, ``adaptive_window``).
    """
    name = "dtfit Legendre filter (robust image)"

    def __init__(self, plant, window=None, adaptive_window=True):
        self.f = ImageFilter(plant["expr"], "t", p0=list(plant["p0"]),
                           window_size=window or plant["window"],
                           order=5, q_diag=list(plant["q"]), basis="legendre",
                           adaptive_window=adaptive_window, robust=True)


class EKFAd(_Ad):
    """Extended Kalman filter, parameters-as-state, the pointwise online
    baseline dtfit's window-image filters are scored against.

    Args:
        plant: An entry of :data:`PLANTS`. The process and measurement
            noise (``q=1e-4``, ``r=0.5``) and the initial covariance
            (``p_init=5.0``) are fixed for every plant, not tuned per
            plant like ``q_diag`` in :data:`PLANTS`.
    """

    name = "EKF (params-as-state)"

    def __init__(self, plant):
        self.f = EKFParam(plant["expr"], "t", list(plant["p0"]), q=1e-4, r=0.5,
                          p_init=5.0)

    def step(self, t, y):
        self.f.update(t, y)

    def predict(self, t):
        return float(self.f.predict(t))

    def params(self):
        return dict(self.f.params_)


class RLSAd(_Ad):
    """Recursive-least-squares AR predictor: a black-box streaming baseline
    that recovers no physical parameters.

    Args:
        plant: An entry of :data:`PLANTS`; unused beyond matching the other
            adapters' constructor signature, since an AR predictor carries
            no plant-specific model or window.
        order: The AR order (number of lagged samples the predictor
            regresses on).
    """

    name = "RLS (AR predictor)"
    gives_params = False

    def __init__(self, plant, order=4):
        self.f = RLSPredictor(order=order, lam=1.0, delta=1e3)
        self._last = float("nan")

    def step(self, t, y):
        self.f.update(y)
        self._last = self.f.last_pred_

    def predict(self, t):
        return self._last

    def params(self):
        return None


class RefitAd(_Ad):
    """Sliding-window ``curve_fit``: re-solve the last ``window`` samples by
    batch NLLS every ``refit_every`` steps, the naive online strategy an
    engineer reaches for before adopting a recursive filter.

    Args:
        plant: An entry of :data:`PLANTS`; its ``window`` sets the sliding
            window length.
        refit_every: Steps between batch refits (not seconds or samples of
            model time): the window is re-solved once every ``refit_every``
            calls to :meth:`step`, once it is full.

    A ``curve_fit`` call that raises (non-convergence, a singular Jacobian)
    is caught and discarded silently: the previous parameter estimate is
    kept and the failed refit leaves no trace in ``params()`` or the drive
    RMSE beyond the estimate not having moved.
    """

    name = "sliding-window curve_fit"

    def __init__(self, plant, refit_every=20):
        self.plant = plant
        self.W = plant["window"]
        self.every = refit_every
        self.t, self.y = [], []
        self.p = np.array(plant["p0"], float)
        self._k = 0

    def step(self, t, y):
        self.t.append(t)
        self.y.append(y)
        if len(self.t) > self.W:
            self.t.pop(0)
            self.y.pop(0)
        self._k += 1
        if len(self.t) >= self.W and self._k % self.every == 0:
            try:
                self.p = bl.scipy_curve_fit(np.array(self.t), np.array(self.y),
                                            self.plant["func"], self.p,
                                            bounds=self.plant["bounds"])
            except Exception:
                pass

    def predict(self, t):
        return float(self.plant["func"](t, *self.p))

    def params(self):
        return dict(zip(sorted(self.plant["true"]), self.p))


def perr(params, true):
    """Mean relative parameter error, in percent, averaged over every key of
    ``true``. ``None`` (an adapter with ``gives_params = False``) passes
    through as ``None``."""
    if params is None:
        return None
    return float(np.mean([abs(params[k] - true[k]) / abs(true[k]) for k in true]) * 100)


def drive(adapter, t, y, clean, warm):
    """Run one adapter sample by sample over a stream.

    Args:
        adapter: One of the estimator adapters above.
        t, y, clean: The time grid, the noisy observation and the clean
            signal, as returned by :func:`gen_plant`.
        warm: Samples to discard from the start before scoring, so a cold
            estimator's opening transient does not count.

    Returns:
        ``(rmse_vs_clean, median_latency_us, track)``: the RMSE against the
        clean signal past ``warm``, the median per-step wall time in
        microseconds of ``adapter.step`` alone (``adapter.predict``, called
        right after for the track, is not timed and is not part of this
        figure), and the full predicted track (including the warm-up, for
        plotting).
    """
    track = np.full(t.size, np.nan)
    lat = []
    for i in range(t.size):
        t0 = time.perf_counter()
        adapter.step(float(t[i]), float(y[i]))
        lat.append((time.perf_counter() - t0) * 1e6)
        track[i] = adapter.predict(float(t[i]))
    valid = np.isfinite(track)
    valid[:warm] = False
    rmse = (float(np.sqrt(np.mean((track[valid] - clean[valid]) ** 2)))
            if valid.any() else np.nan)
    return rmse, float(np.median(lat)), track


def adapters(plant):
    """The five estimator adapters for one plant, freshly constructed."""
    return [EAAd(plant), LegAd(plant), EKFAd(plant), RLSAd(plant), RefitAd(plant)]


# Model mismatch, the negative control: the wrong physical model on-device.
# Each pair is the true plant that generates the stream and the wrong plant
# whose model is fitted to it.
MISMATCH_PAIRS = [
    ("damped_osc", "first_order"),  # a decaying sinusoid fitted as a saturating rise
    ("first_order", "ac_sine"),     # a monotone RC rise fitted as a pure sinusoid
    ("ca_traj", "first_order"),     # an accelerating trajectory fitted as a plateau
]


# RMSE ceiling: on some mismatch pairs a wrong-model EKF's covariance windup
# or exponential blow-up pushes the track to the edge of float64 range;
# values past this are clipped and reported as diverged instead.
MISMATCH_CEILING = 1e4


def mismatch_scores(adapter, t, y, clean, warm):
    """Drive one adapter over a stream and score its track two ways.

    Args:
        adapter: An estimator adapter (:class:`EKFAd` and the others),
            constructed for the model under test, which in this control is
            a different plant's model than the one that generated the
            stream.
        t: The time grid of the stream, in seconds, increasing.
        y: The noisy observations on ``t``, in the plant's output unit.
        clean: The clean signal on ``t``, the same length as ``y``.
        warm: Leading samples to exclude from both scores, the adapter's
            warm-up in samples; 0 excludes none.

    Returns:
        ``(rmse_vs_clean, insample_residual_rmse, diverged)``. The residual,
        the track against the noisy observations, is the on-device
        self-diagnosis signal: a wrong model cannot fit even the data it
        sees, so its residual stays structured and large. Both RMSE values
        are clipped at :data:`MISMATCH_CEILING`, and are NaN when the track
        holds no finite sample past ``warm``. ``diverged`` is True when the
        estimate blew up, going non-finite or past that ceiling, and a
        clipped score is not a measurement: count those runs, do not
        average them.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, _, track = drive(adapter, t, y, clean, warm)
    valid = np.isfinite(track)
    valid[:warm] = False
    if not valid.any():
        return float("nan"), float("nan"), True
    rmse_clean = float(np.sqrt(np.mean((track[valid] - clean[valid]) ** 2)))
    resid = float(np.sqrt(np.mean((track[valid] - y[valid]) ** 2)))
    diverged = not np.isfinite(rmse_clean) or rmse_clean > MISMATCH_CEILING
    return min(rmse_clean, MISMATCH_CEILING), min(resid, MISMATCH_CEILING), diverged


# fault detection and on-device re-adaptation, by the multi-axis fused detector
def make_multi(rng, n=900, noise=0.05):
    """A 3-axis damped oscillator carrying a damping fault: zeta jumps on every
    axis at the midpoint. Returns ``(t, noisy, clean, half)``."""
    t = np.linspace(0, 18, n)
    half = n // 2
    A = np.array([2.0, 1.5, 2.5])
    w = np.array([2.5, 2.0, 3.0])
    z1 = np.array([0.08, 0.10, 0.06])
    z2 = np.array([0.30, 0.28, 0.25])
    clean = np.zeros((n, 3))
    dtt = np.diff(t, prepend=t[0])
    for d in range(3):
        z_arr = np.where(np.arange(n) < half, z1[d], z2[d])
        wd = w[d] * np.sqrt(1 - z_arr ** 2)
        clean[:, d] = A[d] * np.exp(-z_arr * w[d] * t) * np.sin(np.cumsum(wd * dtt))
    return t, clean + rng.normal(0, noise, clean.shape), clean, half


class MergedTracker:
    """A multi-axis oscillator tracker on dtfit's streaming API: a
    :class:`~dtfit_experimental.streaming.FilterBank` of per-axis
    Legendre-spectrum filters driven by
    :class:`~dtfit_experimental.streaming.FusedChiSquareDetector`, which
    sums each filter's ``nis_`` into a fused chi^2(sum n_coef) fault
    statistic and re-arms the bank through ``inflate`` on a detection."""

    def __init__(self, n_axes, p0, *, window=60, fuse_alpha=1e-4, inflate=4.0):
        self.bank = FilterBank.from_model(
            OSC, "t", n_axes, basis="legendre", p0=list(p0),
            window_size=window, order=5, q_diag=[1e-3] * len(p0),
            cusum_h=np.inf)
        self.n_axes = n_axes
        self.detector = self.bank.fused_detector(alpha=fuse_alpha, inflate=inflate)

    @property
    def flags_(self):
        return self.detector.flags_

    def step(self, i, t, y):
        self.detector.update(t, y)

    def predict(self, t):
        return self.bank.predict(np.array([t]))


def run_tracker(t, Y, clean, half, inflate):
    """Run the fused FilterBank over a multi-axis stream. Returns
    ``(tracker, pred, rmse, false_alarms, detect_latency_steps, warmup)``."""
    n = Y.shape[0]
    tr = MergedTracker(3, [2.0, 2.5, 0.1], inflate=inflate)
    pred = np.full((n, 3), np.nan)
    for i in range(n):
        tr.step(i, float(t[i]), Y[i])
        pred[i] = tr.predict(float(t[i]))
    warm = tr.bank.filters[0].W
    valid = np.all(np.isfinite(pred), axis=1)
    valid[:warm] = False
    rmse = (float(np.sqrt(np.mean((pred[valid] - clean[valid]) ** 2)))
            if valid.any() else np.nan)
    flags_post = [i for i in tr.flags_ if i >= half]
    fp = len([i for i in tr.flags_ if i < half])
    lat = (flags_post[0] - half) if flags_post else None
    return tr, pred, rmse, fp, lat, warm


def kalman_multi(t, Y, clean, warm):
    """Constant-acceleration Kalman over the multi-axis stream, identifying no
    plant. Returns ``(pred, rmse)``."""
    kf = KalmanCA(dim=3, dt=float(np.mean(np.diff(t))), q=1e-2, r=0.5)
    pred = np.full((Y.shape[0], 3), np.nan)
    for i in range(Y.shape[0]):
        kf.update(Y[i])
        pred[i] = kf.position()
    valid = np.all(np.isfinite(pred), axis=1)
    valid[:warm] = False
    rmse = float(np.sqrt(np.mean((pred[valid] - clean[valid]) ** 2)))
    return pred, rmse
