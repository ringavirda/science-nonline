# dtfit-legacy

The historical stages of the dtfit method, kept runnable for the evolution
matrix in `dtfit_experimental.experiments.evolution`. Depends on `dtfit`
only and receives no new development.

| Stage | Module | Source |
|---|---|---|
| spectrum balance, square | `dtfit_legacy.book.dsb_balance` | Pysarchuk monograph 2.5, pp. 110-117 |
| balance with adequacy order | `dtfit_legacy.dsb.fit_dsb` | dissertation 2.1.3 to 2.1.5 |
| integral LSI, monomial basis | `dtfit_legacy.book.lsi_integral_monomial` | monograph 2.6, pp. 118-126 |
| integral LSI, Legendre basis | `dtfit_legacy.integral.fit_lsi` | dissertation 2.2 |
| equal areas, truncated spectrum | `dtfit_legacy.book.eac_areas` | monograph 2.7, pp. 127-140 |
| equal areas, direct | `dtfit_legacy.integral.fit_eac` | dissertation 2.3 |
| recursive filters | `dtfit_legacy.streaming` | dissertation 2.4 |
| map-reduce and batched | `dtfit_legacy.scale` | dissertation 3.2.1 |

    pip install -e 'packages/dtfit-legacy[dev]'
    ruff check packages/dtfit-legacy
    pytest -q --tb=line          # from inside the package
