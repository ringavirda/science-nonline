import dtfit


def test_preset_names_are_gone():
    for name in ("fit_lsi", "fit_eac", "LSIFilter", "EACFilter"):
        assert name not in dtfit.__all__
        assert not hasattr(dtfit, name)


def test_surface_is_the_fifteen_names():
    assert set(dtfit.__all__) == {
        "Original", "Image", "ImageStream", "ImageFilter", "fit",
        "order_for", "fit_many", "models", "suggest_models",
        "auto_forecast", "FittingResult", "ForecastResult", "stochastic",
        "diagnostics", "__version__",
    }
