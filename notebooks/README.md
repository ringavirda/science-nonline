# labs

Interactive notebooks for manually exercising the dtfit methods and
interfaces. These are scratchpads, not automated tests (see `tests/` for
those). Plots require the viz extra:

```bash
pip install -e '.[viz]'
```

- `lsi.ipynb` — LSI batch method
- `eda.ipynb` — EDA batch method (function + pipeline forms)
- `streaming.ipynb` — EqualAreasFilter online tracking
- `regressor.ipynb` — NonlineRegressor + scikit-learn integration
