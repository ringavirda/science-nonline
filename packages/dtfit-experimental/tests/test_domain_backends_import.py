"""Import smoke test for the per-domain experiment backends.

The domain ``backend.py`` modules are the single source of truth the notebooks
and the paper scripts import. When the package is restructured they can drift out
of sync (a renamed module, a moved symbol) and only fail when a heavy notebook is
run -- which is easy to miss. This cheaply guards that every backend still
imports and exposes the handful of public names its notebook/paper tooling
depends on, so drift is caught in CI instead of at figure-render time.
"""

from __future__ import annotations

import importlib

import pytest

_BASE = "dtfit_experimental.experiments.domains"

# backend module -> a few public symbols its notebook / paper scripts rely on.
_BACKENDS = {
    "forecasting": ("evaluate_series", "merged_forecaster", "win_summary",
                    "best_oracle", "collapse_oracle_scores", "exp_model_mismatch"),
    "parameter_estimation": ("MODELS", "gen", "est_eac", "est_lsi", "est_nlls",
                             "param_err", "FAMILY_REASON", "load_puromycin",
                             "real_puromycin", "exp_model_mismatch"),
    "big_data": (),
    "embedded_control": ("clean_accuracy", "sweep_perr_all", "exp_model_mismatch"),
    "stochastic_series": ("exp_ar_discrimination", "exp_ar_order_recovery",
                          "exp_fracdiff_whitening", "exp_student_t"),
    "realtime_gps": (),
}


@pytest.mark.parametrize("domain,symbols", list(_BACKENDS.items()),
                         ids=list(_BACKENDS))
def test_backend_imports_and_exposes_symbols(domain, symbols):
    mod = importlib.import_module(f"{_BASE}.{domain}.backend")
    missing = [s for s in symbols if not hasattr(mod, s)]
    assert not missing, f"{domain}.backend missing public symbols: {missing}"
