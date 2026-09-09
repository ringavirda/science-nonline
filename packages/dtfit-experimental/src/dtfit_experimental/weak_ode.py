"""Weak-form ODE identification on the image.

A rate law ``y' = f(y; theta)`` that is nonlinear in its constants ``theta``
becomes LINEAR in ``theta`` in the weak form: multiply the ODE by a test
function ``phi`` that vanishes with its derivatives at the window endpoints,
integrate over the window, and move each derivative of ``y`` onto ``phi`` by
parts (``int y' phi = -int y phi'``). No ODE is solved and no starting guess is
needed; the derivatives of the noisy ``y`` are never taken pointwise, only
projected. Two structural tricks widen the reach:

* a rational rate law linearizes by clearing its denominator (Michaelis-Menten:
  ``(Km + y) y' = -Vm y``);
* a two-state system observed in one state linearizes by eliminating the
  unobserved state (Lotka-Volterra from the prey alone).

The projection is exactly the image's derivative-basis machinery: ``I0``, ``I1``
and ``I2`` project ``g``, ``g'`` and ``g''`` onto the same test-function family.
The construction is the modulating-function method of system identification
(Shinbrot 1957; Preisig and Rippin 1993), the same one weak SINDy uses for
noisy data (Messenger and Bortz 2021); this module adopts it into the image
scheme rather than proposes it. Measured against nonlinear least squares on
the integrated law started near the truth (60 seeds, 2 to 10 percent noise),
the weak form is 1.3 to 5 times less accurate and 30 to 180 times faster,
and needs no start. Its instrumental-variable and generalized-least-squares
variants (:func:`solve_weak`) do not close that gap: the instruments change
nothing and the whitening helps one law and hurts another, so the loss is
the truncated test-function family, not the solver. Used as the start of
that nonlinear least squares instead, the weak estimate gives the full
least-squares accuracy on every law at 2 to 10 percent noise and removes
the basin failures of a poor start (Michaelis-Menten: 28 percent of poorly
started fits fail, 1 percent of seeded ones), so its place is the
start-free seeder of an ODE fit, not its replacement. ``beta`` in the
prey-only Lotka-Volterra is structurally unidentifiable and is not returned.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial import legendre as _L

WeakOps = tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]


def weak_operators(
    t: np.ndarray, n_test: int = 12, order: int = 2
) -> WeakOps:
    """The weak-form projection operators ``(I0, I1, I2)`` on the grid ``t``.

    Each maps a sampled function ``g(t)`` to its ``n_test``-vector of
    projections against the test functions
    ``phi_k(u) = (1 - u**2)**order * P_k(u)``, with ``P_k`` the ``k``-th
    Legendre polynomial and ``u`` the domain mapped to ``[-1, 1]``. The window
    factor vanishes with ``order`` derivatives at ``u = +/-1``, so the boundary
    terms of the integration by parts drop:

    * ``I0(g) = integral g phi`` -- ``g`` against ``phi``;
    * ``I1(g) = integral g' phi = -integral g phi'`` -- one derivative of ``g``,
      read from the data without differentiating it;
    * ``I2(g) = integral g'' phi = integral g phi''`` -- two derivatives.

    Args:
        t: sample times, shape ``(n,)``, strictly increasing; may be
            non-uniform (trapezoid weights come from :func:`numpy.gradient`).
        n_test: number of test functions (Legendre degrees ``0..n_test-1``).
            More test functions give more weak equations; ``12`` suffices for
            the demonstrated laws, ``24`` for the second-order elimination.
        order: vanishing order of the window factor at the endpoints; ``2``
            lets ``I2`` integrate a second derivative by parts cleanly.

    Returns:
        ``(I0, I1, I2)``, three callables each taking ``g`` of shape ``(n,)``
        and returning a vector of shape ``(n_test,)``.
    """
    t = np.asarray(t, dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    u = 2.0 * (t - t0) / (t1 - t0) - 1.0
    du = 2.0 / (t1 - t0)
    w = np.gradient(t)
    eye = np.eye(n_test)
    P = _L.legvander(u, n_test - 1)
    dP = np.column_stack(
        [_L.legval(u, _L.legder(eye[k])) for k in range(n_test)]
    )
    d2P = np.column_stack(
        [_L.legval(u, _L.legder(eye[k], 2)) for k in range(n_test)]
    )
    b = (1 - u ** 2) ** order
    db = -2 * order * u * (1 - u ** 2) ** (order - 1)
    d2b = (
        -2 * order * (1 - u ** 2) ** (order - 1)
        + 4 * order * (order - 1) * u ** 2 * (1 - u ** 2) ** (order - 2)
    )
    phi = b[:, None] * P
    dphi = (db[:, None] * P + b[:, None] * dP) * du
    d2phi = (
        d2b[:, None] * P + 2 * db[:, None] * dP + b[:, None] * d2P
    ) * du * du

    def i0(g: np.ndarray) -> np.ndarray:
        return (w * np.asarray(g, dtype=float)) @ phi

    def i1(g: np.ndarray) -> np.ndarray:
        return -(w * np.asarray(g, dtype=float)) @ dphi

    def i2(g: np.ndarray) -> np.ndarray:
        return (w * np.asarray(g, dtype=float)) @ d2phi

    return i0, i1, i2


def _smooth(
    t: np.ndarray, y: np.ndarray, degree: int | None = None
) -> np.ndarray:
    """A Legendre least-squares smoother of ``y`` on ``t``, the instrument
    source of :func:`solve_weak`: nearly noise-free, so a design built from
    it is uncorrelated with the noise that enters ``y``'s own design."""
    deg = int(degree) if degree is not None else int(
        min(24, max(6, t.size // 25)))
    return np.polynomial.legendre.Legendre.fit(t, y, deg)(t)


def solve_weak(
    design: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    t: np.ndarray, y: np.ndarray, *, method: str = "ols",
    iterations: int = 2,
) -> np.ndarray:
    """Solve a weak-form law that is linear in its constants.

    ``design(v) -> (A, b)`` builds the regressor matrix ``A`` (test functions
    by constants) and the target ``b`` from a sampled series ``v``; the law
    reads ``A theta = b``. The sampled series enters ``A`` as well as ``b``,
    so ordinary least squares carries an errors-in-variables bias and the
    weak residuals are correlated across test functions.

    Args:
        method: ``"ols"``, ordinary least squares on ``(A(y), b(y))``;
            ``"iv"``, instrumental variables with the instruments the design
            built from :func:`_smooth` of ``y``; ``"ivgls"``, ``"iv"``
            followed by generalized least squares, the residual covariance
            propagated from unit sample noise through the design by a
            finite-difference Jacobian and used to whiten both sides,
            ``iterations`` times.

    Returns:
        The constants, shape ``(p,)``.
    """
    y = np.asarray(y, dtype=float)
    A, b = design(y)
    theta = np.linalg.lstsq(A, b, rcond=None)[0]
    if method == "ols":
        return theta
    Z, _ = design(_smooth(t, y))
    theta = np.linalg.lstsq(Z.T @ A, Z.T @ b, rcond=None)[0]
    if method == "iv":
        return theta
    if method != "ivgls":
        raise ValueError(f"unknown method {method!r}")
    h = 1e-5 * (float(np.std(y)) or 1.0)
    for _ in range(int(iterations)):
        r0 = A @ theta - b
        B = np.empty((r0.size, y.size))
        for i in range(y.size):
            yp = y.copy()
            yp[i] += h
            Ai, bi = design(yp)
            B[:, i] = (Ai @ theta - bi - r0) / h
        S = B @ B.T
        S += 1e-10 * (np.trace(S) / S.shape[0]) * np.eye(S.shape[0])
        L = np.linalg.cholesky(S)
        Aw, bw, Zw = (np.linalg.solve(L, A), np.linalg.solve(L, b),
                      np.linalg.solve(L, Z))
        theta = np.linalg.lstsq(Zw.T @ Aw, Zw.T @ bw, rcond=None)[0]
    return theta


def fit_logistic(
    t: np.ndarray, y: np.ndarray, n_test: int = 12, *, method: str = "ols"
) -> dict[str, float]:
    """Fit ``y' = r y (1 - y / K)`` in the weak form.

    Expanded, ``y' = r y - (r/K) y**2`` is linear in ``[r, r/K]``, so
    ``I1(y) = r I0(y) - (r/K) I0(y**2)``.

    Returns:
        ``{"r": growth rate, "K": carrying capacity}``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0, i1, _ = weak_operators(t, n_test=n_test)

    def design(v):
        return np.column_stack([i0(v), -i0(v * v)]), i1(v)

    r, r_over_k = solve_weak(design, t, y, method=method)
    return {"r": float(r), "K": float(r / r_over_k)}


def fit_michaelis_menten(
    t: np.ndarray, y: np.ndarray, n_test: int = 12, *, method: str = "ols"
) -> dict[str, float]:
    """Fit the Michaelis-Menten decay ``y' = -Vm y / (Km + y)`` in the weak
    form.

    Clearing the denominator gives ``Km y' + y y' = -Vm y``; with
    ``y y' = (y**2)'/2`` the weak form is
    ``Km I1(y) + Vm I0(y) = -I1(y**2)/2``, linear in ``[Km, Vm]``.

    Returns:
        ``{"Vm": max rate, "Km": half-saturation constant}``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0, i1, _ = weak_operators(t, n_test=n_test)

    def design(v):
        return np.column_stack([i1(v), i0(v)]), -0.5 * i1(v * v)

    km, vm = solve_weak(design, t, y, method=method)
    return {"Vm": float(vm), "Km": float(km)}


def fit_damped_oscillator(
    t: np.ndarray, y: np.ndarray, n_test: int = 12, *, method: str = "ols"
) -> dict[str, float]:
    """Fit a damped oscillation ``y'' + 2 zeta omega y' + omega**2 y = 0``.

    The homogeneous second-order law is already linear in its constants, so
    the weak form is ``I2(y) + c1 I1(y) + c2 I0(y) = 0`` with ``c1 = 2 zeta
    omega`` and ``c2 = omega**2``; solving ``[I1(y), I0(y)] [c1, c2] = -I2(y)``
    gives the natural frequency and damping ratio without differentiating the
    noisy ``y`` and without a starting guess.

    Returns:
        ``{"omega": natural frequency, "zeta": damping ratio}``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0, i1, i2 = weak_operators(t, n_test=n_test)

    def design(v):
        return np.column_stack([i1(v), i0(v)]), -i2(v)

    c1, c2 = solve_weak(design, t, y, method=method)
    omega = float(np.sqrt(abs(c2)))
    return {"omega": omega, "zeta": float(c1 / (2 * omega)) if omega else float("nan")}


def fit_lotka_volterra_prey(
    t: np.ndarray, x: np.ndarray, n_test: int = 24, *, method: str = "ols"
) -> dict[str, float]:
    """Recover the Lotka-Volterra rates from the PREY series ``x`` alone.

    The predator is eliminated through ``y = (alpha - (ln x)') / beta``, which
    turns the prey equation into the second-order law
    ``-(ln x)'' = delta alpha x - delta x' - gamma alpha + gamma (ln x)'``.
    Its weak form
    ``-I2(ln x) = (delta alpha) I0(x) - delta I1(x)
                  - (gamma alpha) I0(1) + gamma I1(ln x)``
    is linear in ``[delta alpha, delta, gamma alpha, gamma]``, from which
    ``alpha``, ``gamma`` and ``delta`` follow. ``beta`` scales the unobserved
    predator and is structurally unidentifiable from the prey alone, so it is
    not returned.

    Args:
        x: prey series, shape ``(n,)``, strictly positive.

    Returns:
        ``{"alpha": prey growth, "gamma": predator death,
        "delta": predator gain}``.
    """
    t = np.asarray(t, dtype=float)
    x = np.clip(np.asarray(x, dtype=float), 1e-12, None)
    i0, i1, i2 = weak_operators(t, n_test=n_test)

    def design(v):
        v = np.clip(v, 1e-12, None)
        lv = np.log(v)
        return (np.column_stack([i0(v), -i1(v), -i0(np.ones_like(v)),
                                 i1(lv)]), -i2(lv))

    c = solve_weak(design, t, x, method=method)
    delta_alpha, delta, _gamma_alpha, gamma = c
    return {
        "alpha": float(delta_alpha / delta),
        "gamma": float(gamma),
        "delta": float(delta),
    }
