"""Package surface: public API, submodule importability, and carve-out."""

import importlib
import pkgutil

import pytest

import dtfit


def test_top_level_api():
    for name in [
        "Model",
        "nonline_fit",
        "poly_fit",
        "NonlineRegressor",
        "EqualAreasFilter",
        "fit_lsi",
        "fit_eda",
        "fit_dsb",
        "enable_logging",
    ]:
        assert hasattr(dtfit, name), f"missing public name: {name}"


def test_all_submodules_import():
    for mod in pkgutil.walk_packages(dtfit.__path__, "dtfit."):
        importlib.import_module(mod.name)


def test_simulation_not_in_top_namespace():
    # Synthetic-data tooling is opt-in, not flattened into dtfit.*
    for name in [
        "gen_exponential4",
        "apply_normal_noise",
        "DataGenerationMw",
        "DataPollutionMw",
    ]:
        assert not hasattr(dtfit, name), f"{name} leaked into top namespace"


def test_simulation_importable_explicitly():
    from dtfit.simulation import (  # noqa: F401
        DataGenerationMw,
        DataPollutionMw,
        apply_normal_noise,
        gen_exponential4,
    )


@pytest.mark.parametrize(
    "mod",
    [
        "dtfit.extra.dt.dsbi",
        "dtfit.extra.dt.dsbe",
        # Hand-written differential-transform tables replaced by generic Taylor.
        "dtfit.extra.dt.discretes",
        "dtfit.extra.dt.spectrum",
    ],
)
def test_removed_methods_are_gone(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)
