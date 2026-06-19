"""Ground-truth scenarios for every catalogued model family.

Each :class:`Scenario` pins a catalogue factory to a concrete ground-truth
parameter set and an observation grid in the regime that family is meant for,
plus the metric by which a fit is judged:

* ``"params"`` -- the parameters are identifiable, so we score *recovery* of the
  ground truth (the honest target for a parameter-estimation method);
* ``"r2"`` -- the family is only weakly identifiable in its parameters (sums of
  exponentials, overlapping peaks, phase of a cycle), so we score *curve
  quality* (R^2 against the clean signal) instead.

The metric / threshold split records, in code, where a method genuinely
estimates parameters versus where it can only fit a curve -- the distinction the
docs must make honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from dtfit import models
from dtfit.models import Model


@dataclass
class Scenario:
    """One ground-truth case: a catalogue family on data it should handle."""

    name: str
    true: dict[str, float]
    x0: float
    x1: float
    n: int = 200
    metric: str = "params"          # "params" | "r2"
    # default per-parameter relative-error ceiling (metric="params") or minimum
    # R^2 (metric="r2"); the gate may scale these by noise level.
    tol: float = 0.10
    r2_min: float = 0.99
    note: str = ""
    factory_name: str = ""
    factory_kwargs: dict = field(default_factory=dict)

    def model(self) -> Model:
        return getattr(models, self.factory_name)(**self.factory_kwargs)

    def clean(self, x: np.ndarray) -> np.ndarray:
        m = self.model()
        t = sp.Symbol(m.var)
        f = sp.sympify(m.expr)
        params = sorted((s for s in f.free_symbols if s != t), key=str)
        missing = [str(p) for p in params if str(p) not in self.true]
        if missing:
            raise KeyError(f"{self.name}: no ground truth for {missing}")
        vals = [self.true[str(p)] for p in params]
        fn = sp.lambdify(t, f.subs(dict(zip(params, vals))), "numpy")
        y = np.asarray(fn(np.asarray(x, float)), dtype=float)
        return np.full_like(x, float(y)) if y.ndim == 0 else y

    def grid(self) -> np.ndarray:
        return np.linspace(self.x0, self.x1, self.n)

    def make(
        self,
        noise: float,
        seed: int,
        *,
        outlier_frac: float = 0.0,
        outlier_scale: float = 8.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x, y, clean)`` with Gaussian noise at ``noise`` * signal-std.

        With ``outlier_frac > 0`` a random fraction of samples are additionally
        corrupted by spikes of ``outlier_scale`` * signal-std (the
        contamination the ensemble estimator is built for). The default
        ``outlier_frac=0`` leaves the signal a plain Gaussian-noise series, so
        the golden baseline is unaffected.
        """
        x = self.grid()
        clean = self.clean(x)
        rng = np.random.default_rng(seed)
        sig = float(np.std(clean)) or 1.0
        y = clean + rng.normal(0.0, noise * sig, x.size)
        if outlier_frac > 0.0:
            k = max(1, int(round(outlier_frac * x.size)))
            idx = rng.choice(x.size, size=k, replace=False)
            y[idx] += rng.normal(0.0, outlier_scale * sig, k)
        return x, y, clean


def _s(name, factory_name, true, x0, x1, **kw) -> Scenario:
    return Scenario(name=name, factory_name=factory_name, true=true,
                    x0=x0, x1=x1, **kw)


# The full catalogue, one ground-truth scenario per family. Parameters and
# domains are chosen to put each family in the regime it is documented for.
SCENARIOS: list[Scenario] = [
    # --- trends -------------------------------------------------------------
    _s("linear", "linear", {"a0": 1.0, "a1": 2.0}, 0.0, 5.0),
    _s("quadratic", "quadratic", {"a0": 1.0, "a1": 0.5, "a2": 0.3}, 0.0, 5.0),
    _s("cubic", "cubic", {"a0": 1.0, "a1": 0.5, "a2": -0.4, "a3": 0.1}, 0.0, 4.0),
    _s("power_law", "power_law", {"a": 2.0, "b": 1.6}, 0.0, 5.0),
    _s("logarithmic", "logarithmic", {"a": 1.0, "b": 2.0}, 0.0, 10.0),
    _s("sqrt_law", "sqrt_law", {"a": 1.0, "b": 2.0}, 0.0, 10.0),
    # --- growth -------------------------------------------------------------
    _s("exponential", "exponential", {"a": 1.5, "b": 0.8}, 0.0, 2.0),
    _s("exp_growth_offset", "exp_growth_offset",
       {"a": 1.0, "b": 0.7, "c": 2.0}, 0.0, 2.0),
    # --- decay / relaxation -------------------------------------------------
    _s("exp_decay", "exp_decay", {"a": 3.0, "b": 1.0}, 0.0, 3.0),
    _s("exp_decay_offset", "exp_decay_offset",
       {"a": 4.0, "b": 1.5, "c": 1.0}, 0.0, 3.0),
    _s("first_order", "first_order", {"K": 2.0, "tau": 0.5}, 0.0, 3.0),
    _s("biexponential", "biexponential",
       {"a": 3.0, "b": 3.0, "c": 1.0, "d": 0.4}, 0.0, 5.0,
       metric="r2", r2_min=0.99,
       note="sum of exponentials: parameters only weakly identifiable"),
    _s("stretched_exponential", "stretched_exponential",
       {"A": 2.0, "tau": 1.0, "q": 1.5}, 0.0, 4.0),
    # --- sigmoids -----------------------------------------------------------
    _s("logistic", "logistic", {"L": 5.0, "k": 1.5, "x0": 5.0}, 0.0, 10.0),
    _s("gompertz", "gompertz", {"A": 3.0, "b": 2.5, "c": 0.8}, 0.0, 8.0),
    _s("weibull_cdf", "weibull_cdf", {"K": 2.0, "lam": 2.0, "k": 1.8}, 0.0, 8.0),
    _s("tanh_step", "tanh_step", {"a": 1.0, "b": 2.0, "c": 1.5, "d": 5.0}, 0.0, 10.0),
    # --- saturating / rational ---------------------------------------------
    _s("michaelis_menten", "michaelis_menten", {"Vmax": 4.0, "K": 1.5}, 0.0, 10.0),
    _s("hill", "hill", {"Vmax": 4.0, "K": 2.0, "n": 2.5}, 0.0, 8.0),
    # --- peaks --------------------------------------------------------------
    _s("gaussian", "gaussian", {"A": 3.0, "mu": 5.0, "s": 1.0}, 0.0, 10.0),
    _s("lorentzian", "lorentzian", {"A": 3.0, "mu": 5.0, "g": 1.0}, 0.0, 10.0),
    _s("double_gaussian", "double_gaussian",
       {"A1": 3.0, "m1": 3.0, "s1": 0.7, "A2": 2.0, "m2": 7.0, "s2": 0.8},
       0.0, 10.0, metric="r2", r2_min=0.99,
       note="overlapping peaks: label-swap / weak identifiability"),
    # --- oscillatory --------------------------------------------------------
    _s("sine", "sine", {"c": 1.0, "A": 2.0, "w": 1.5, "p": 0.5}, 0.0, 12.0, n=400),
    _s("damped_oscillation", "damped_oscillation",
       {"A": 2.0, "w": 2.0, "z": 0.12}, 0.0, 12.0, n=400),
    _s("fourier_series", "fourier_series",
       {"c": 1.0, "w": 1.0, "a1": 1.5, "b1": 0.5, "a2": 0.6, "b2": 0.0,
        "a3": 0.3, "b3": 0.0}, 0.0, 14.0, n=500,
       metric="r2", r2_min=0.99, factory_kwargs={"n_harmonics": 3},
       note="harmonic amplitudes individually weak; curve quality is the target"),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}

# Noise levels (fraction of signal std) swept by the accuracy matrix.
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10)
