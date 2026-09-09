# The reference method

DSB (differential spectra balance) recovers a model's parameters by equating
its Maclaurin spectrum to a polynomial pre-fit's, order by order, and solving
the balance symbolically. It is the ancestor the image estimator is derived
against, not part of `fit`, and it now lives in the `dtfit-legacy` package as
`dtfit_legacy.dsb.fit_dsb`, with `dtfit_legacy.dsb.find_degree` choosing the
pre-fit's degree by an information criterion. The historical criteria it
grew into, and their recursive and map-reduce forms, sit beside it there;
see the Lineage page of the wiki for the whole evolution and where each
stage lives.
