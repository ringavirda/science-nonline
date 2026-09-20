# Experiments

Every number this study quotes comes from a notebook here. Each notebook is
one experiment and its report: open it, read the claim at the top, run it,
and compare the numbers it prints against the prose that follows.

## Index

Runtime is the full run on a workstation; "data" is what has to be on disk
beyond what `python -m dtfit_experimental.study.download_data` fetches.

### method/

| Notebook | Claim | Runtime | Data |
|---|---|---|---|
| 13_two_bases | h versus p refinement: Legendre wins a smooth target, block is exact on a step only where a window boundary falls on the jump and at parity with Legendre otherwise, the Gram conditioning by order and grid, the cost of restricting a linear fit to either basis | a few seconds | none |

## Writing a notebook

1. Title, then the claims it establishes in plain words and what each would
   look like if it were false.
2. One setup cell: imports, seeds, sizes, and
   `QUICK = os.environ.get("DTFIT_QUICK") == "1"`. `QUICK` only shrinks
   counts (sample sizes, replicate counts, coefficient budgets); the
   committed notebook is always the full run, `DTFIT_QUICK` unset.
3. Sections: a question, the code that answers it (calling `dtfit` and
   `dtfit_experimental` directly, not through a wrapper that hides the
   experiment), the table or figure, the reading, then a check cell that
   asserts the ordering or bound the section establishes. A rerun that stops
   supporting the claim fails there instead of printing a different number.
4. A real-data section says where the data comes from and skips with a
   visible message when it is absent. Use `study.notebook.data_file(...)`
   and guard the cells that need it with `if PATH:` rather than letting a
   missing file raise: `nbclient` fails the whole notebook on an uncaught
   exception, and a skip is a value here, never an exception.
5. Tables that a paper or the dissertation restyles are also written to
   `experiments/results/<notebook>/<table>.csv` through
   `study.notebook.save_table`, small and tracked, so the figure builders in
   the thesis repository plot from the notebook's own numbers instead of
   recomputing them. This is an export of data, not a report.
6. A closing section: findings, the limits that were not tested, and a
   provenance line from `study.notebook.provenance(started)`.
7. Plain ASCII in every markdown cell and printed label; the two bases are
   called Legendre and block, never a retired name.

Only reusable or heavy machinery moves to `study/`: baselines, simulators,
loaders, metrics, plot helpers, the Monte-Carlo scaffolding
(`study.montecarlo`) and the notebook helpers themselves
(`study.notebook`). A module earns its place there when two notebooks use
it, or it is an estimator, simulator or loader that deserves its own tests,
or it runs for hours over an archive. Everything else - the actual
construction of data, the calls into the method, the reading of a result -
stays in the notebook's own cells.

## What a notebook may import

At execution time, a notebook may import only what the CI research job
installs: `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, `statsmodels`,
`dtfit`, `dtfit_experimental`, `dtfit_legacy`. `numpy`, `scipy`, `sympy`
and `dtfit` are core dependencies of this package; `matplotlib`, `pandas`,
`statsmodels`, `dtfit-legacy` and the notebook harness (`nbclient`,
`ipykernel`) come with its `bench` extra, so
`pip install -e 'packages/dtfit-experimental[bench]'` is what a notebook
needs. Anything heavier -
`torch`, `cupy` - goes through `study.notebook.optional_import(name)`, and
the section that needs it skips the same way a missing data file does.
`dtfit_legacy` is imported only where a comparison of historical stages is
the point (the source forms, the families, the evolution matrix), never as
a stand-in for the current method.

## Figures

Inline only; there is no `figures/` directory here, the thesis repository
plots from the exported CSVs. `figure.dpi = 110` in the setup cell, at most
two panels to a figure, and a committed notebook stays under 1 MB.

## Running the suite

```bash
python -m dtfit_experimental.study.download_data      # fetch datasets once

# open and rerun interactively
jupyter lab experiments/method/13_two_bases.ipynb

# or execute headless, full run, outputs written in place
jupyter nbconvert --to notebook --execute --inplace experiments/method/13_two_bases.ipynb
```

A notebook is written and edited in Jupyter like any other. Before it is
committed it is rerun from a fresh kernel with `DTFIT_QUICK` unset -
Restart and Run All, or the `nbconvert` line above - so that the committed
file carries the outputs of the full run and the CSV exports on disk match
them.

`tests/test_notebooks.py` executes every notebook here under
`DTFIT_QUICK=1` with a per-cell and a whole-notebook time budget; that is
the check CI runs, never a substitute for opening the committed, full-run
notebook.
