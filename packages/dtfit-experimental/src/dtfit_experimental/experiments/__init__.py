"""dtfit experiment suite -- the *study* tier of ``dtfit-experimental``.

This is the validation research tree, **not** part of the importable library
contract (see :mod:`dtfit_experimental`): it is exempt from the mypy gate,
ruff-relaxed, and run notebook-by-notebook rather than imported. The library tier
(the experimental adaptations) never imports from here.

Two families of experiments, each a folder of self-contained **Jupyter
notebooks**:

* ``cases/`` -- per-adaptation studies (one optimization / structural idea per
  folder, scored on the promotion matrix);
* ``domains/`` -- per-application-domain studies (the validated levers merged and
  run against the methods a practitioner in that domain actually uses).

Each experiment folder holds a ``backend.py`` (the single source of truth for its
simulation / estimation / data infra -- pure compute, no plotting), the notebook
that imports it and produces the report (tables, figures, narrative), and a
``figures/`` directory of the saved figures. Open and run a notebook directly,
e.g. ``jupyter lab cases/01_control_systems/01_control_systems.ipynb``, or
headless::

    jupyter nbconvert --to notebook --execute --inplace \\
        dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb

``cases/REPORTS.md`` and ``domains/DOMAINS.md`` index the notebooks. The shared
``common`` package provides the pure-compute helpers (metrics, baselines,
datasets, plotting) the backends import.
"""
