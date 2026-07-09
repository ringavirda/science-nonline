# scikit-learn estimator

`NonlineRegressor` is a scikit-learn-compatible estimator (fit / predict / score)
over the LSI / EAC / DSB methods. It composes with `Pipeline`, `GridSearchCV` and
the rest of the sklearn ecosystem, so you can cross-validate the fitting method
and its knobs like any other hyperparameter.

```python
from sklearn.model_selection import GridSearchCV
from dtfit import NonlineRegressor

reg = NonlineRegressor(expr="a*exp(b*x)", var="x", method="lsi")
search = GridSearchCV(reg, {"method": ["lsi", "eac"]}, cv=5)
search.fit(X, y)          # X is a 2-D column vector of the single feature
```

::: dtfit.NonlineRegressor
