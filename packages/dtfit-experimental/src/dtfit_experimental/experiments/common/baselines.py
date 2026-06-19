"""Baseline methods compared against dtfit across the experiment suite.

Everything here is a *fairly runnable* baseline -- the established methods a
practitioner would actually reach for, wrapped behind small uniform helpers:

* classical curve fitting -- SciPy ``curve_fit`` (Levenberg-Marquardt NLLS),
  ``numpy.polyfit``;
* neural nets -- scikit-learn ``MLPRegressor`` (batch and incremental
  ``partial_fit``), and PyTorch MLP / LSTM sequence forecasters;
* classical time series -- statsmodels ARIMA / SARIMAX;
* trajectory tracking -- a constant-acceleration Kalman filter (numpy);
* the naive random-walk forecast benchmark.

The deep / statsmodels backends are imported lazily and guarded by
``HAVE_TORCH`` / ``HAVE_STATSMODELS`` so the suite still runs (skipping those
rows) on a core install. Modern deep-forecasting research methods
(DLinear/TimesNet/Time-LLM) are *not* re-implemented here -- they are compared
against by reproducing their published benchmark numbers in
``experiments/06_benchmark_ltsf``.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    HAVE_STATSMODELS = False


# --------------------------------------------------------------------------- #
# classical curve fitting
# --------------------------------------------------------------------------- #
def scipy_curve_fit(x, y, func, p0, *, bounds=None, maxfev=20000):
    """Levenberg-Marquardt / trust-region NLLS via scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit

    kwargs = {"p0": p0, "maxfev": maxfev}
    if bounds is not None:
        kwargs["bounds"] = bounds
    popt, _ = curve_fit(func, x, y, **kwargs)
    return popt


def polyfit_predict(x, y, x_eval, deg=5):
    c = np.polyfit(x, y, deg)
    return np.polyval(c, x_eval)


def mlp_curve(x, y, x_eval, *, hidden=(64, 64), max_iter=2000, seed=0):
    """A black-box neural-net regressor f(x)->y (1-D curve fitting)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(x.reshape(-1, 1))
    Xtr = xs.transform(x.reshape(-1, 1))
    net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                       random_state=seed)
    net.fit(Xtr, y)
    return net.predict(xs.transform(np.asarray(x_eval).reshape(-1, 1)))


# --------------------------------------------------------------------------- #
# forecasting baselines (train on a series, predict `horizon` steps ahead)
# --------------------------------------------------------------------------- #
def random_walk_forecast(train, horizon):
    """Persist the last observed value (the standard hard-to-beat benchmark)."""
    return np.full(horizon, float(train[-1]))


def _make_windows(series, lookback):
    X = np.array([series[i:i + lookback] for i in range(len(series) - lookback)])
    Y = np.array([series[i + lookback] for i in range(len(series) - lookback)])
    return X, Y


def mlp_forecast(train, horizon, *, lookback=24, hidden=(64, 64), max_iter=1500,
                 seed=0, incremental=False):
    """Autoregressive sklearn-MLP forecaster (recursive multi-step).

    With ``incremental=True`` it is trained by ``partial_fit`` over mini-batches
    -- the streaming-friendly NN baseline used by the big-data experiment.
    """
    from sklearn.neural_network import MLPRegressor

    train = np.asarray(train, dtype=float)
    mu, sd = train.mean(), train.std() + 1e-12
    s = (train - mu) / sd
    X, Y = _make_windows(s, lookback)
    if len(X) < 5:
        return random_walk_forecast(train, horizon)
    net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                       random_state=seed)
    if incremental:
        bs = max(32, len(X) // 20)
        for _ in range(10):
            for i in range(0, len(X), bs):
                net.partial_fit(X[i:i + bs], Y[i:i + bs])
    else:
        net.fit(X, Y)
    window = list(s[-lookback:])
    out = []
    for _ in range(horizon):
        nxt = float(net.predict(np.array(window[-lookback:]).reshape(1, -1))[0])
        out.append(nxt)
        window.append(nxt)
    return np.array(out) * sd + mu


def arima_forecast(train, horizon, *, order=(2, 1, 2), seasonal_order=None):
    """statsmodels ARIMA / SARIMAX point forecast."""
    if not HAVE_STATSMODELS:
        raise RuntimeError("statsmodels not available")
    train = np.asarray(train, dtype=float)
    if seasonal_order is not None:
        model = sm.tsa.statespace.SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(train, order=order)
    fit = model.fit() if seasonal_order is None else model.fit(disp=False)
    return np.asarray(fit.forecast(steps=horizon), dtype=float)


# --------------------------------------------------------------------------- #
# torch sequence nets (small, CPU)
# --------------------------------------------------------------------------- #
def _torch_seq_forecast(train, horizon, *, lookback, kind, epochs, seed):
    torch.manual_seed(seed)
    train = np.asarray(train, dtype=float)
    mu, sd = train.mean(), train.std() + 1e-12
    s = (train - mu) / sd
    X, Y = _make_windows(s, lookback)
    if len(X) < 5:
        return random_walk_forecast(train, horizon)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    if kind == "lstm":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                o, _ = self.lstm(x.unsqueeze(-1))
                return self.fc(o[:, -1, :])
        net = Net()
    else:  # mlp
        net = nn.Sequential(nn.Linear(lookback, 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xt), Yt)
        loss.backward()
        opt.step()

    net.eval()
    window = list(s[-lookback:])
    out = []
    with torch.no_grad():
        for _ in range(horizon):
            xin = torch.tensor(window[-lookback:], dtype=torch.float32).reshape(1, -1)
            nxt = float(net(xin).item())
            out.append(nxt)
            window.append(nxt)
    return np.array(out) * sd + mu


def lstm_forecast(train, horizon, *, lookback=24, epochs=200, seed=0):
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    return _torch_seq_forecast(train, horizon, lookback=lookback, kind="lstm",
                               epochs=epochs, seed=seed)


def torch_mlp_forecast(train, horizon, *, lookback=24, epochs=300, seed=0):
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    return _torch_seq_forecast(train, horizon, lookback=lookback, kind="mlp",
                               epochs=epochs, seed=seed)


# --------------------------------------------------------------------------- #
# constant-acceleration Kalman filter (trajectory tracking gold standard)
# --------------------------------------------------------------------------- #
class KalmanCA:
    """Per-axis constant-acceleration Kalman filter (position measurements).

    State ``[p, v, a]`` per axis; the standard model used for tracking and
    short-horizon trajectory prediction. ``dim`` axes are tracked independently.
    """

    def __init__(self, dim=3, dt=1.0, q=1e-2, r=1.0):
        self.dim = dim
        self.dt = dt
        self.F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
        self.H = np.array([[1.0, 0.0, 0.0]])
        self.Q = q * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                               [dt**3 / 2, dt**2, dt],
                               [dt**2 / 2, dt, 1.0]])
        self.R = np.array([[r]])
        self.x = [np.zeros((3, 1)) for _ in range(dim)]
        self.P = [np.eye(3) * 10.0 for _ in range(dim)]
        self._init = False
        # Per-axis one-step innovations of the last update, and their fused
        # normalized-innovation-squared (~chi-square(dim) under no maneuver).
        # Lets an external detector apply the *same* self-calibrating adaptive
        # re-arming to the Kalman baseline as to dtfit -- a fair maneuver-tracking
        # comparison driven by identical machinery.
        self.last_residuals_ = np.zeros(dim)
        self.last_nis_ = 0.0

    def update(self, z):
        """Ingest one position measurement ``z`` (length ``dim``)."""
        z = np.asarray(z, dtype=float)
        if not self._init:
            for d in range(self.dim):
                self.x[d][0, 0] = z[d]
            self._init = True
            self.last_residuals_ = np.zeros(self.dim)
            self.last_nis_ = 0.0
            return self.position()
        nis = 0.0
        res = np.zeros(self.dim)
        for d in range(self.dim):
            xp = self.F @ self.x[d]
            Pp = self.F @ self.P[d] @ self.F.T + self.Q
            y = z[d] - (self.H @ xp)[0, 0]
            S = (self.H @ Pp @ self.H.T + self.R)[0, 0]
            res[d] = y
            nis += y * y / S
            K = (Pp @ self.H.T) / S
            self.x[d] = xp + K * y
            self.P[d] = (np.eye(3) - K @ self.H) @ Pp
        self.last_residuals_ = res
        self.last_nis_ = float(nis)
        return self.position()

    def inflate(self, factor):
        """Inflate every axis' covariance -- the adaptive re-arming hook, mirror
        of the dtfit filter's ``inflate``."""
        for d in range(self.dim):
            self.P[d] = self.P[d] * float(factor)

    def position(self):
        return np.array([self.x[d][0, 0] for d in range(self.dim)])

    def forecast(self, horizon):
        """Roll the state forward ``horizon`` steps (no new measurements)."""
        xs = [self.x[d].copy() for d in range(self.dim)]
        out = np.zeros((horizon, self.dim))
        for k in range(horizon):
            for d in range(self.dim):
                xs[d] = self.F @ xs[d]
                out[k, d] = xs[d][0, 0]
        return out


# --------------------------------------------------------------------------- #
# additional classical forecasting baselines (the standard toolkit)
# --------------------------------------------------------------------------- #
def seasonal_naive_forecast(train, horizon, *, period):
    """Repeat the last observed season -- the standard seasonal benchmark."""
    train = np.asarray(train, dtype=float)
    if period <= 0 or train.size < period:
        return random_walk_forecast(train, horizon)
    last = train[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last, reps)[:horizon]


def drift_forecast(train, horizon):
    """Random walk *with drift*: extrapolate the average per-step change
    (Hyndman's "drift method")."""
    train = np.asarray(train, dtype=float)
    if train.size < 2:
        return random_walk_forecast(train, horizon)
    slope = (train[-1] - train[0]) / (train.size - 1)
    return train[-1] + slope * np.arange(1, horizon + 1)


def poly_extrap_forecast(train, horizon, *, deg=2):
    """Fit a global polynomial and extrapolate -- the surrogate-fit baseline
    (no parametric structure; extrapolates by curvature only)."""
    train = np.asarray(train, dtype=float)
    t = np.arange(train.size)
    c = np.polyfit(t, train, deg)
    return np.polyval(c, np.arange(train.size, train.size + horizon))


def ets_forecast(train, horizon, *, trend="add", seasonal=None, period=None,
                 damped=False):
    """Holt-Winters exponential smoothing (statsmodels ``ExponentialSmoothing``)
    -- the workhorse classical forecaster (level + optional trend + season)."""
    if not HAVE_STATSMODELS:
        raise RuntimeError("statsmodels not available")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    train = np.asarray(train, dtype=float)
    kw = dict(trend=trend, damped_trend=damped and trend is not None)
    if seasonal is not None and period and train.size >= 2 * period:
        kw.update(seasonal=seasonal, seasonal_periods=period)
    fit = ExponentialSmoothing(train, **kw).fit()
    return np.asarray(fit.forecast(horizon), dtype=float)


def theta_forecast(train, horizon, *, period=None):
    """The Theta method (statsmodels ``ThetaModel``) -- the M3-competition
    winner; a robust, widely-used decomposition forecaster."""
    if not HAVE_STATSMODELS:
        raise RuntimeError("statsmodels not available")
    from statsmodels.tsa.forecasting.theta import ThetaModel
    train = np.asarray(train, dtype=float)
    pr = period if (period and train.size >= 2 * period) else 1
    tm = ThetaModel(train, period=pr) if pr > 1 else ThetaModel(train, period=1,
                                                                deseasonalize=False)
    return np.asarray(tm.fit().forecast(horizon), dtype=float)


def sarima_forecast(train, horizon, *, order=(1, 1, 1), seasonal_order=None):
    """Seasonal ARIMA point forecast (statsmodels SARIMAX) -- ARIMA's seasonal
    extension, the standard statistical model for seasonal series."""
    return arima_forecast(train, horizon, order=order,
                          seasonal_order=seasonal_order)


# --------------------------------------------------------------------------- #
# robust / nonparametric curve fitting baselines (parameter estimation)
# --------------------------------------------------------------------------- #
def robust_curve_fit(x, y, func, p0, *, bounds=None, loss="soft_l1",
                     f_scale=1.0, maxfev=20000):
    """NLLS with a robust loss (scipy ``least_squares``) -- the standard way to
    fit a known model in the presence of outliers (Huber/soft-L1 down-weights
    large residuals; the established robust analog of ``curve_fit``)."""
    from scipy.optimize import least_squares
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def resid(p):
        return np.asarray(func(x, *p), dtype=float) - y

    kw = dict(loss=loss, f_scale=f_scale, max_nfev=maxfev)
    if bounds is not None:
        kw["bounds"] = bounds
    sol = least_squares(resid, np.asarray(p0, float), **kw)
    return np.asarray(sol.x, dtype=float)


def gp_curve(x, y, x_eval, *, seed=0):
    """Gaussian-process regression (sklearn) -- the standard nonparametric
    Bayesian smoother; fits any smooth curve but recovers no physical parameters
    (the nonparametric counterpart to dtfit's structured fit)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float)
    span = float(x.max() - x.min()) or 1.0
    k = ConstantKernel(1.0) * RBF(span / 10) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                  random_state=seed, n_restarts_optimizer=1)
    gp.fit(x, y)
    return gp.predict(np.asarray(x_eval, float).reshape(-1, 1))


# --------------------------------------------------------------------------- #
# online estimators (the established real-time / streaming baselines)
# --------------------------------------------------------------------------- #
class RLSPredictor:
    """Recursive Least Squares one-step predictor on an AR(``order``) model.

    The classical online system-identification / adaptive-filtering algorithm:
    it tracks the linear predictor coefficients of the signal online with a
    forgetting factor ``lam`` (so it adapts to drift). It yields one-step
    predictions but, being a black-box AR model, **no physical parameters** --
    the streaming counterpart of the MLP baseline.
    """

    def __init__(self, order=2, lam=0.99, delta=100.0):
        self.order = int(order)
        self.lam = float(lam)
        self.w = np.zeros(self.order)
        self.P = np.eye(self.order) * float(delta)
        self.hist: list[float] = []
        self.last_pred_ = float("nan")

    def update(self, y):
        y = float(y)
        if len(self.hist) >= self.order:
            xv = np.array(self.hist[-self.order:][::-1])
            self.last_pred_ = float(self.w @ xv)
            err = y - self.last_pred_
            Px = self.P @ xv
            k = Px / (self.lam + xv @ Px)
            self.w = self.w + k * err
            self.P = (self.P - np.outer(k, Px)) / self.lam
        self.hist.append(y)
        return self

    def predict_next(self):
        if len(self.hist) >= self.order:
            xv = np.array(self.hist[-self.order:][::-1])
            return float(self.w @ xv)
        return float(self.hist[-1]) if self.hist else 0.0


class EKFParam:
    """Extended Kalman Filter that estimates the parameters of a *known*
    nonlinear model online.

    This is the textbook established method for online nonlinear parameter
    estimation: the parameters are a random-walk state, the measurement is
    ``y = f(t; p)``, and the EKF linearizes ``f`` about the current estimate via
    its parameter-Jacobian (``∂f/∂p``, compiled once with SymPy). It is the
    fair, same-job baseline for dtfit's streaming equal-areas / Legendre filters
    -- both track the model parameters online; they differ in the *measurement*
    (a pointwise value here vs an integrated area / spectrum for dtfit).
    """

    def __init__(self, expr, var, p0, *, q=1e-4, r=1.0, p_init=1.0):
        import sympy as sp
        t = sp.Symbol(var)
        model = sp.sympify(expr)
        self.params = sorted((s for s in model.free_symbols if s != t), key=str)
        n = len(self.params)
        self._f = sp.lambdify([t, *self.params], model, "numpy")
        self._jac = [sp.lambdify([t, *self.params], sp.diff(model, p), "numpy")
                     for p in self.params]
        self.p = np.asarray(p0, dtype=float)
        self.P = np.eye(n) * float(p_init)
        self.Q = np.eye(n) * float(q)
        self.R = float(r)
        self.last_residual_ = float("nan")

    def update(self, t, y):
        n = self.p.size
        # predict (random-walk dynamics): state unchanged, covariance grows
        self.P = self.P + self.Q
        yhat = float(self._f(t, *self.p))
        H = np.array([float(jk(t, *self.p)) for jk in self._jac])
        if not (np.isfinite(yhat) and np.all(np.isfinite(H))):
            return self
        S = float(H @ self.P @ H + self.R)
        if S <= 0:
            return self
        K = self.P @ H / S
        innov = float(y) - yhat
        self.p = self.p + K * innov
        self.P = (np.eye(n) - np.outer(K, H)) @ self.P
        self.last_residual_ = innov
        return self

    @property
    def params_(self):
        return {str(s): float(v) for s, v in zip(self.params, self.p)}

    def predict(self, x):
        return self._f(np.asarray(x, dtype=float), *self.p)
