"""The forecasting model specs, their seeding, and the classical baseline
toolkit, shared by the forecasting notebook's structure-given and blind
routes.

:func:`fit_kind` fits one of :data:`SERIES`'s named model kinds (see
``study.series`` for the table) on a training window and returns its
prediction over the full x-range, LSI (Legendre) on the seed :func:`_trend_spec`
builds. Every seed and frequency detector here (:func:`_w0_from`,
:func:`_detect_modulation`, :func:`_detect_chirp`) reads its estimate off the
training data itself, never off the series' declared kind, so the same
machinery underlies both the "structure given" column, which is handed the
correct kind, and any blind router built on top of it.

:func:`baseline_preds` runs the established forecasting toolkit a practitioner
would reach for -- random walk, drift, polynomial extrapolation, seasonal
naive, Holt-Winters ETS, Theta, (S)ARIMA, an MLP and, when available, an
LSTM -- beside it.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit._signal import dominant_period

from . import baselines as bl
from . import notebook

#: How many harmonics the Fourier-series model carries: enough for the 5th
#: harmonic of the fundamental, which is the AC waveform's content.
N_HARMONICS = 5
#: Model kinds fitted by local optimization from p0, without bounds; every
#: other kind fits by a global differential-evolution search over its bounds.
LOCAL_FIT_KINDS = {"fourier_series", "chirp", "linear_wave", "poly_seasonal",
                   "linear_seasonal", "transient_seasonal"}

#: Cached result of the one-time torch probe, so a notebook without torch
#: prints ``notebook.optional_import``'s skip line once per process instead
#: of once per series per horizon. ``None`` until the first probe.
_have_torch: bool | None = None

#: The human-readable disclosure of the fixed-order convention below, which
#: the notebook renders next to the baseline table.
FIXED_ORDER_NOTE = (
    "Note: the (S)ARIMA / ETS baselines use fixed orders (ARIMA (2,1,2), "
    "SARIMA (1,1,1)x(1,0,1,period), ETS additive+damped) applied uniformly across "
    "series, NOT a per-series AIC / auto_arima search -- a mild, disclosed "
    "fixed-order handicap on the classical baselines.")


def _torch_available() -> bool:
    """Whether torch is importable, probed at most once per process.

    Returns:
        ``True`` when ``import torch`` succeeds. The probe itself runs once
        (caching in :data:`_have_torch`), so its skip line prints at most
        once even across many :func:`baseline_preds` calls in one notebook.
    """
    global _have_torch
    if _have_torch is None:
        _have_torch = notebook.optional_import("torch") is not None
    return _have_torch


def _fit_bounds(spec, kind):
    """Bounds to pass to the LSI fitter: ``None``, meaning local optimization
    from p0, for the seed-reliable high-cost models, and otherwise the spec's
    own bounds, meaning a global differential-evolution search."""
    return None if kind in LOCAL_FIT_KINDS else spec.get("bounds")


def _stage(spec, kind):
    """A boosting stage spec, dropping bounds for the local-fit kinds so the
    Legendre fit runs local optimization."""
    if kind in LOCAL_FIT_KINDS:
        return {k: v for k, v in spec.items() if k != "bounds"}
    return spec


def _dx(t_tr):
    return float(t_tr[-1] - t_tr[0]) / max(t_tr.size - 1, 1)


def _w0_from(y_tr, t_tr, period_hint=None):
    """Angular frequency of the dominant cycle, in the x-coordinate of ``t_tr``.

    It comes from the dominant sample period times the actual sample spacing
    ``dx``, which keeps the seed correct however many samples the training
    window holds."""
    dx = _dx(t_tr)
    period_samp, strength = dominant_period(y_tr)
    # Accept a dominant period up to half the window, so at least two observed
    # cycles. The ``<=`` admits a cycle of exactly N/2, which the weather
    # sensor has; a strict ``<`` sends it to the wrong fallback frequency.
    if not (np.isfinite(period_samp) and strength > 0.03
            and period_samp <= y_tr.size / 2):
        period_samp = period_hint if period_hint else y_tr.size / 6
    return 2 * np.pi / (period_samp * dx)


def _detect_modulation(y_tr, t_tr):
    """The AM modulation (envelope) angular frequency, from the dominant cycle
    of the analytic-signal envelope."""
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(y_tr - y_tr.mean()))
        ps, strength = dominant_period(env - env.mean())
        if np.isfinite(ps) and strength > 0.02 and ps < y_tr.size / 2:
            return 2 * np.pi / (ps * _dx(t_tr))
    except Exception:
        pass
    return _w0_from(y_tr, t_tr) / 6.0


def _detect_chirp(y_tr, t_tr):
    """The linear-chirp start angular frequency ``w0`` and sweep rate ``k`` for
    the model ``sin(w0*x + k*x^2 + p)``, whose instantaneous angular frequency
    is ``omega(x) = w0 + 2k*x``.

    The estimate comes from the analytic-signal (Hilbert) instantaneous phase.
    For a linear chirp the unwrapped phase is exactly the quadratic
    ``phi(x) = p + w0*x + k*x^2``, so a degree-2 polynomial fit of that phase
    reads ``w0`` and ``k`` off directly. Hilbert phase is the standard
    instantaneous-frequency estimator, and it is the only route here that
    resolves the sweep at all: an FFT peak returns a frequency averaged over
    the window, which loses both the magnitude and the sign of ``k``, and a
    coarse zero-crossing count saturates at the low-frequency end."""
    try:
        from scipy.signal import hilbert
        phase = np.unwrap(np.angle(hilbert(y_tr - float(np.mean(y_tr)))))
        k, w0, _ = np.polyfit(t_tr, phase, 2)        # phi = k x^2 + w0 x + p
        if np.isfinite(w0) and np.isfinite(k):
            return max(float(w0), 0.2 * abs(float(w0))), float(k)
    except Exception:
        pass
    w = _w0_from(y_tr, t_tr)
    return w, 0.0


def _spec_from(expr, pmap, *, method="lsi", **extra):
    """Build a fit spec from a ``{name: (p0, lo, hi)}`` map, with bounds and p0
    ordered to match SymPy's name-sorted parameter layout, the convention
    the Legendre fit uses. Going through the map removes any chance of getting
    that ordering wrong by hand."""
    import sympy as sp
    syms = sorted((s for s in sp.sympify(expr).free_symbols if str(s) != "x"),
                  key=str)
    spec = dict(expr=expr, var="x", method=method,
                p0=[pmap[str(s)][0] for s in syms],
                bounds=[(pmap[str(s)][1], pmap[str(s)][2]) for s in syms])
    spec.update(extra)
    return spec


def _osc_order(w0, t_tr, n_cycles_mult=1.0):
    """The Fourier-basis order that resolves ``n_cycles_mult`` times the
    fundamental's cycle count over the training x-span."""
    cycles = w0 * float(t_tr[-1] - t_tr[0]) / (2 * np.pi)
    return int(1.4 * n_cycles_mult * cycles) + 10


def _poly_seed(y_tr, t_tr, deg):
    """Seed coefficients ``[a0, a1, ..., a_deg]`` in ascending powers of x,
    from a plain polynomial least-squares fit. From a good starting point
    the joint trend+seasonal models fit locally instead of by global
    search."""
    pc = np.polyfit(t_tr, y_tr, deg)             # numpy: highest power first
    return [float(pc[deg - i]) for i in range(deg + 1)]


def _trend_spec(kind, y_tr, t_tr, period_hint=None):
    """Return ``(stage_spec, scale)``. For a sinusoidal model class the spec
    carries ``k_star``, a high spectral order, and ``filter_data=False``, so
    that the cycle and its harmonics survive; the Fourier-basis order is that
    same ``k_star``."""
    if kind == "exp":                                    # params [a, b]
        return dict(expr="a*exp(b*x)", var="x", method="lsi",
                    bounds=[(0.05, 20), (-10, 10)], p0=[1.0, 1.0]), float(y_tr[0])
    if kind == "logistic":                               # params [L, k, x0]
        # Epidemic and diffusion growth saturates. A pure exponential
        # compounds and badly overshoots the deceleration; the logistic
        # captures the carrying limit L.
        ylast = float(y_tr[-1])
        xspan = float(t_tr[-1] - t_tr[0]) or 1.0
        return dict(expr="L/(1 + exp(-k*(x - x0)))", var="x", method="lsi",
                    bounds=[(ylast * 0.8, ylast * 12), (0.1, 60.0),
                            (t_tr[0], t_tr[0] + 2.5 * xspan)],
                    p0=[ylast * 1.5, 6.0 / xspan, t_tr[0] + xspan], k_star=6), 1.0
    if kind == "linear":                                 # params [a0, a1]
        s = _poly_seed(y_tr, t_tr, 1)
        return dict(expr="a0 + a1*x", var="x", method="lsi", p0=s), 1.0
    if kind == "linear_wave":                            # a0+a1 x+a2 sin+a3 cos
        # A level and slope plus one slow cycle, a single period over the
        # training span. It captures the rise-peak-settle wave of, say, a
        # currency crash and its partial recovery, which no monotone
        # trend can.
        xspan = float(t_tr[-1] - t_tr[0]) or 1.0
        w = 2 * np.pi / xspan
        a0, a1 = _poly_seed(y_tr, t_tr, 1)
        amp = float(np.std(y_tr)) + 1e-3
        return _spec_from(
            "a0 + a1*x + a2*sin(w*x) + a3*cos(w*x)",
            {"a0": (a0, -1e6, 1e6), "a1": (a1, -1e6, 1e6),
             "a2": (0.0, -5 * amp, 5 * amp), "a3": (amp, -5 * amp, 5 * amp),
             "w": (w, 0.3 * w, 3 * w)}, k_star=10), 1.0
    if kind in ("poly_seasonal", "linear_seasonal"):     # joint trend + cycle
        deg = 2 if kind == "poly_seasonal" else 1
        s = _poly_seed(y_tr, t_tr, deg)
        w0 = _w0_from(y_tr, t_tr, period_hint)
        amp = float(np.std(y_tr)) + 1e-3
        pterms = " + ".join(f"a{i}*x**{i}" for i in range(deg + 1))
        pmap = {f"a{i}": (s[i], -1e6, 1e6) for i in range(deg + 1)}
        pmap["A"] = (amp, 1e-3, 5 * amp)
        pmap["p"] = (0.0, -np.pi, np.pi)
        pmap["w"] = (w0, 0.7 * w0, 1.3 * w0)
        return _spec_from(f"{pterms} + A*sin(w*x + p)", pmap,
                          k_star=_osc_order(w0, t_tr), filter_data=False), 1.0
    if kind == "transient_seasonal":                     # settling trend + cycle
        # Trend term a1*x*e^{-c*x} settles to the stable level a0 rather than
        # extrapolating a slope indefinitely.
        s0 = float(np.mean(y_tr))
        w0 = _w0_from(y_tr, t_tr, period_hint)
        amp = float(np.std(y_tr)) + 1e-3
        return _spec_from(
            "a0 + a1*x*exp(-c*x) + A*sin(w*x + p)",
            {"a0": (s0, -1e6, 1e6), "a1": (amp, -1e6, 1e6), "c": (3.0, 0.05, 60.0),
             "A": (amp, 1e-3, 5 * amp), "p": (0.0, -np.pi, np.pi),
             "w": (w0, 0.7 * w0, 1.3 * w0)},
            k_star=_osc_order(w0, t_tr), filter_data=False), 1.0
    if kind == "sine":                                   # params [A, c, p, w]
        w0 = _w0_from(y_tr, t_tr, period_hint)
        amp = float(np.std(y_tr)) * 1.5 + 1e-3
        order = _osc_order(w0, t_tr)
        return _spec_from(
            "c + A*sin(w*x + p)",
            {"A": (amp, 1e-3, 5 * amp),
             "c": (float(np.mean(y_tr)), float(y_tr.min()) - amp, float(y_tr.max()) + amp),
             "p": (0.0, -np.pi, np.pi), "w": (w0, 0.3 * w0, 3 * w0)},
            k_star=order, filter_data=False), 1.0
    if kind == "fourier_series":                         # AC + harmonics
        w0 = _w0_from(y_tr, t_tr, period_hint)
        amp = float(np.max(np.abs(y_tr))) + 1e-3
        K = N_HARMONICS
        terms = ["c"] + [f"a{k}*sin({k}*w*x) + b{k}*cos({k}*w*x)"
                         for k in range(1, K + 1)]
        pmap = {"c": (0.0, -amp, amp), "w": (w0, 0.85 * w0, 1.18 * w0)}
        for k in range(1, K + 1):
            pmap[f"a{k}"] = (0.0, -2 * amp, 2 * amp)
            pmap[f"b{k}"] = (0.0, -2 * amp, 2 * amp)
        order = _osc_order(w0, t_tr, n_cycles_mult=K)
        return _spec_from(" + ".join(terms), pmap, k_star=order,
                          filter_data=False), 1.0
    if kind == "am":                                     # (1+m cos wm x) sin(wc x+p)
        wc = _w0_from(y_tr, t_tr, period_hint)
        wm = _detect_modulation(y_tr, t_tr)
        order = _osc_order(wc, t_tr, n_cycles_mult=1.5)
        return _spec_from(
            "(1 + m*cos(wm*x))*sin(wc*x + p)",
            {"m": (0.5, 0.0, 3.0), "p": (0.0, -np.pi, np.pi),
             "wc": (wc, 0.85 * wc, 1.18 * wc), "wm": (wm, 0.3 * wm, 3 * wm)},
            k_star=order, filter_data=False), 1.0
    if kind == "chirp":                                  # A sin(w0 x + k x^2 + p)
        w0, kr = _detect_chirp(y_tr, t_tr)
        amp = float(np.max(np.abs(y_tr))) + 1e-3
        x_span = float(t_tr[-1] - t_tr[0]) or 1.0
        wmax = max(w0 + abs(kr) * x_span * 2, w0 * 2)
        order = _osc_order(wmax, t_tr)
        return _spec_from(
            "A*sin(w0*x + k*x**2 + p)",
            {"A": (amp, 0.2 * amp, 3 * amp),
             "k": (kr, -abs(kr) * 3 - 5, abs(kr) * 3 + 5),
             "p": (0.0, -np.pi, np.pi), "w0": (w0, 0.3 * w0, 2 * w0)},
            k_star=order, filter_data=False), 1.0
    if kind == "damped":                                 # params [A, w, z]
        w0 = _w0_from(y_tr, t_tr, period_hint)
        amp = float(np.max(np.abs(y_tr))) + 1e-3
        return dict(expr="A*exp(-z*w*x)*sin(w*sqrt(1-z**2)*x)", var="x",
                    method="lsi",
                    bounds=[(0.1 * amp, 5 * amp), (0.3 * w0, 3 * w0), (1e-3, 0.9)],
                    p0=[amp, w0, 0.05]), 1.0
    return dict(expr="a0 + a1*x + a2*x**2", var="x", method="lsi",   # [a0,a1,a2]
                p0=_poly_seed(y_tr, t_tr, 2)), 1.0


def _seasonal_stage(y_tr, t_tr, period_hint):
    """A boosting seasonal stage ``A*sin(w*x + p)``, its params name-sorted to
    [A, p, w], or ``None`` when no dominant cycle is found."""
    n = y_tr.size
    period_samp, strength = dominant_period(y_tr)
    if not (np.isfinite(period_samp) and strength > 0.05 and period_samp < n / 2):
        if not period_hint:
            return None
        period_samp = period_hint
    w0 = 2 * np.pi / (period_samp * _dx(t_tr))
    amp = float(np.std(y_tr - np.polyval(np.polyfit(np.arange(n), y_tr, 1),
                                         np.arange(n)))) + 1e-3
    return dict(expr="A*sin(w*x + p)", var="x", method="lsi",
                bounds=[(1e-3, 5 * amp), (-np.pi, np.pi), (0.3 * w0, 3 * w0)],
                p0=[amp, 0.0, w0])


def fit_kind(kind, t_tr, y_tr, t_all, period_hint=None):
    """Fit one of :data:`dtfit_experimental.study.series.SERIES`'s named model
    kinds (its ``trend`` field, for example ``"damped"`` or ``"linear_seasonal"``)
    on the training window and return the model evaluated over the full
    x-range.

    The fit is Legendre LSI, seeded by :func:`_trend_spec` (a polynomial or
    FFT-based estimate of the trend, frequency and amplitude, read off
    ``y_tr``/``t_tr`` themselves, never off ``kind``), at the spectral order
    and bounds that seed builds.

    Args:
        kind: Model kind, one of the trend kinds :mod:`dtfit_experimental.study.series`
            documents (``"exp"``, ``"logistic"``, ``"linear"``, ``"linear_wave"``,
            ``"poly"``, ``"poly_seasonal"``, ``"linear_seasonal"``,
            ``"transient_seasonal"``, ``"sine"``, ``"fourier_series"``, ``"am"``,
            ``"chirp"``, ``"damped"``, or any other value, which falls back to a
            plain quadratic trend).
        t_tr: Training x-coordinates, ascending, shape ``(n_tr,)``.
        y_tr: Training observations, shape ``(n_tr,)``, same units as the
            returned model.
        t_all: x-coordinates to evaluate the fitted model at, shape ``(n,)``;
            typically ``t_tr`` extended with the holdout's x-coordinates.
        period_hint: Optional period, in samples, used as a fallback when the
            FFT-detected dominant cycle is not reliable enough.

    Returns:
        The fitted model evaluated at every point of ``t_all``, shape ``(n,)``.

    Raises:
        Whatever ``dtfit.fit`` raises on a spec this series' ``_trend_spec``
        builds (a degenerate window, an order the record cannot support).
    """
    spec, scale = _trend_spec(kind, y_tr, t_tr, period_hint)
    r = dt.fit(spec["expr"], dt.Original(t_tr, y_tr / scale), spec["var"],
              basis="legendre", order=spec.get("k_star", 5), p0=spec.get("p0"),
              bounds=_fit_bounds(spec, kind))
    return np.asarray(r.model(t_all)) * scale


def baseline_preds(y_tr, h, cfg, quick):
    """Run the established forecasting toolkit over one series' horizon.

    Args:
        y_tr: Training observations, shape ``(n_tr,)``.
        h: Forecast horizon, in samples.
        cfg: Series config dict with at least ``"seasonal"`` (bool) and
            ``"period"`` (samples, used only when ``seasonal`` is true).
        quick: When true, skips SARIMA and the LSTM baseline for a fast
            smoke run (``DTFIT_QUICK``).

    Returns:
        A dict of baseline name to its length-``h`` forecast array. A
        baseline whose optional dependency is missing or whose fit fails is
        either absent (LSTM without torch) or present with every value
        ``nan`` (ETS, Theta, ARIMA, SARIMA, MLP, LSTM on failure), so a
        caller can filter on ``np.isfinite``.
    """
    period = cfg["period"] if cfg["seasonal"] else None
    out = {}
    out["random walk"] = bl.random_walk_forecast(y_tr, h)
    out["drift"] = bl.drift_forecast(y_tr, h)
    out["poly extrap"] = bl.poly_extrap_forecast(y_tr, h, deg=2)
    if period:
        out["seasonal naive"] = bl.seasonal_naive_forecast(y_tr, h, period=period)
    try:
        # the fixed ETS spec: additive trend, damped, additive season, and
        # no per-series structure search (see FIXED_ORDER_NOTE above)
        out["ETS (Holt-Winters)"] = bl.ets_forecast(
            y_tr, h, trend="add", damped=True,
            seasonal="add" if period else None, period=period)
    except Exception:
        out["ETS (Holt-Winters)"] = np.full(h, np.nan)
    try:
        out["Theta"] = bl.theta_forecast(y_tr, h, period=period)
    except Exception:
        out["Theta"] = np.full(h, np.nan)
    try:
        # the fixed ARIMA order (2,1,2): a convention, not an AIC search
        out["ARIMA"] = bl.arima_forecast(y_tr, h, order=(2, 1, 2))
    except Exception:
        out["ARIMA"] = np.full(h, np.nan)
    if period and period <= 12 and not quick:
        try:
            # the fixed SARIMA order (1,1,1)x(1,0,1,period), likewise uniform
            out["SARIMA"] = bl.sarima_forecast(
                y_tr, h, order=(1, 1, 1), seasonal_order=(1, 0, 1, period))
        except Exception:
            out["SARIMA"] = np.full(h, np.nan)
    try:
        out["MLP"] = bl.mlp_forecast(
            y_tr, h, lookback=min(36, max(6, y_tr.size // 3)),
            max_iter=300 if quick else 1000)
    except Exception:
        out["MLP"] = np.full(h, np.nan)
    if not quick and _torch_available():
        try:
            out["LSTM"] = bl.lstm_forecast(
                y_tr, h, lookback=min(36, max(6, y_tr.size // 3)), epochs=120)
        except Exception:
            out["LSTM"] = np.full(h, np.nan)
    return out
