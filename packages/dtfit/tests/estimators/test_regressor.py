"""scikit-learn compatibility of NonlineRegressor."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from dtfit import NonlineRegressor


def _reg():
    return NonlineRegressor("a*atan(w*x)", "x", method="eac", p0=[1.0, 1.0])


def test_fit_predict_score(arctan_data):
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.predict(x).shape == x.shape
    assert reg.n_features_in_ == 1
    assert reg.score(x, y) > 0.8


def test_clone_and_get_params():
    reg = _reg()
    assert reg.get_params()["method"] == "eac"
    cloned = clone(reg)
    assert not hasattr(cloned, "coef_")


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        _reg().predict(np.array([1.0, 2.0]))


def test_pipeline(arctan_data):
    x, y, _ = arctan_data
    pipe = Pipeline([("reg", _reg())]).fit(x.reshape(-1, 1), y)
    assert pipe.score(x.reshape(-1, 1), y) > 0.8


def test_cross_val_score(arctan_data):
    x, y, _ = arctan_data
    scores = cross_val_score(
        _reg(), x.reshape(-1, 1), y, cv=KFold(3, shuffle=True, random_state=0)
    )
    assert len(scores) == 3
    assert scores.mean() > 0.8


def test_multifeature_rejected():
    reg = NonlineRegressor("a*x", "x")
    with pytest.raises(ValueError):
        reg.fit(np.zeros((10, 2)), np.zeros(10))


def test_predict_multifeature_rejected(arctan_data):
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    with pytest.raises(ValueError):
        reg.predict(np.zeros((5, 2)))


def test_sklearn_tags():
    tags = _reg().__sklearn_tags__()
    assert tags.estimator_type == "regressor"
    assert tags.target_tags.required is True


def test_feature_names_in_(arctan_data):
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    df = pd.DataFrame({"x": x})
    reg = _reg().fit(df, y)
    assert list(reg.feature_names_in_) == ["x"]
    assert reg.n_features_in_ == 1


def test_zero_arg_constructible_and_clonable():
    """scikit-learn contract: an estimator must be constructible with no args and
    every ``__init__`` parameter must have a default (so ``clone`` and
    meta-estimator introspection work)."""
    import inspect

    sig = inspect.signature(NonlineRegressor.__init__)
    required = [
        name for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect._empty
    ]
    assert not required, f"__init__ params without defaults: {required}"

    reg = NonlineRegressor()          # zero-arg construction
    cloned = clone(reg)               # relies on get_params/set_params round-trip
    assert isinstance(cloned, NonlineRegressor)
    assert cloned.get_params() == reg.get_params()
    assert not hasattr(cloned, "coef_")


def test_default_estimator_fits_affine():
    """The default model is a runnable affine fit (so ``NonlineRegressor()`` is a
    usable estimator, not merely constructible)."""
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 80)
    y = 1.5 + 0.7 * x + 0.01 * rng.standard_normal(x.size)
    reg = NonlineRegressor().fit(x, y)
    assert reg.score(x, y) > 0.99
