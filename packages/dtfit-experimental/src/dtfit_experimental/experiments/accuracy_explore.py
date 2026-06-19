"""Exploration harness: recovery accuracy across the whole model catalogue.

For every :data:`SCENARIOS` family, over a noise sweep, fit via the realistic
user path -- ``Model.fit()`` self-seeded, exactly as the docs/notebooks do it --
and a ``scipy.curve_fit`` baseline seeded the same way, and report parameter
recovery (max relative error) and curve quality (R^2 vs the clean signal).

This is the Phase-0 lever: it turns "methods sometimes look worse than expected"
into numbers, so the Phase-1 gate thresholds and the docs caveats are grounded in
measurement rather than anecdote.

Run::

    python -m dtfit_experimental.experiments.accuracy_explore
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

# Make the accuracy corpus importable (it lives under dtfit's tests/, the single
# source of truth for ground-truth scenarios shared with the Phase-1 gate).
_TESTS = Path(__file__).resolve()
for _p in _TESTS.parents:
    if (_p / "packages" / "dtfit" / "tests").is_dir():
        sys.path.insert(0, str(_p / "packages" / "dtfit" / "tests"))
        break

from accuracy.scenarios import SCENARIOS, NOISE_LEVELS  # noqa: E402
from accuracy.harness import (  # noqa: E402
    ordered_params as _ordered_params,
    r2 as _r2,
    param_err as _param_err,
    predict as _predict,
    curve_fit_baseline as _curve_fit_baseline,
)


def run() -> int:
    print(f"{'scenario':<22}{'noise':>6} | "
          f"{'dt_perr':>8}{'cf_perr':>8} | {'dt_r2':>8}{'cf_r2':>8} | metric  verdict")
    print("-" * 92)
    n_fail = 0
    for scn in SCENARIOS:
        names = _ordered_params(scn)
        for noise in NOISE_LEVELS:
            x, y, clean = scn.make(noise, seed=0)
            # realistic path: self-seeded Model.fit (what the docs use)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = scn.model().fit(x, y)
                est = np.asarray(res.coeffs, float)
                pred = _predict(res, x)
                dt_perr = _param_err(scn, names, est)
                dt_r2 = _r2(clean, pred)
            except Exception as exc:  # noqa: BLE001
                dt_perr, dt_r2 = float("nan"), float("nan")
                print(f"{scn.name:<22}{noise:>6.2f} | dtfit RAISED: {exc}")
                n_fail += 1
                continue

            cf_popt, cf_pred = _curve_fit_baseline(scn, x, y, names)
            if cf_popt is None:
                cf_perr, cf_r2 = float("nan"), float("nan")
            else:
                cf_perr = _param_err(scn, names, cf_popt)
                cf_r2 = _r2(clean, np.asarray(cf_pred, float))

            if scn.metric == "params":
                ok = np.isfinite(dt_perr) and dt_perr <= max(
                    scn.tol, 0.5 + 6 * noise)
            else:
                ok = np.isfinite(dt_r2) and dt_r2 >= scn.r2_min - 0.5 * noise
            verdict = "ok" if ok else "**FAIL**"
            if not ok:
                n_fail += 1
            print(f"{scn.name:<22}{noise:>6.2f} | "
                  f"{dt_perr:>8.3f}{cf_perr:>8.3f} | "
                  f"{dt_r2:>8.4f}{cf_r2:>8.4f} | {scn.metric:<7} {verdict}")
    print("-" * 92)
    print(f"{n_fail} flagged case(s) of "
          f"{len(SCENARIOS) * len(NOISE_LEVELS)} total")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(0 if run() == 0 else 1)
