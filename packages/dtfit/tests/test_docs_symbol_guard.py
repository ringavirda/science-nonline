"""Guard the published docs and wiki against removed core symbols and the
old EAC-robustness claims.

The wiki publishes verbatim on every push, so a page that names a removed
core symbol or calls EAC the noise-robust method is a public regression.
This test reads the pages (no dtfit import) and fails on any match. The
allowlist is empty: names that survive in dtfit_experimental or
dtfit.reference are matched only in their core-qualified form, so a correct
experimental mention does not trip the guard.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
WIKI = ROOT / "wiki"
DOCS = ROOT / "packages" / "dtfit" / "docs"


def _pages(*dirs):
    out = []
    for d in dirs:
        out += sorted(p for p in d.rglob("*.md"))
    return out


ALL_PAGES = _pages(WIKI, DOCS)

# Group 1: fully removed everywhere (no source in dtfit or dtfit_experimental).
GROUP1 = re.compile(
    r"\b(ensemble_fit|auto_estimate|EnsembleResult|fit_eac_adaptive)\b"
    r"|method\s*=\s*[\"']adaptive[\"']"
)

# Group 2: removed from core, live in dtfit_experimental / dtfit.reference.
# Match only the core-qualified form. "dtfit." immediately before the name
# does not match "dtfit_experimental.scale.NAME" or "dtfit.reference.fit_dsb"
# (the char before the name there is not "dtfit.").
_CORE_NAMES = (
    "PartitionedLSI|PartitionedEAC|PartitionedBatchLSI|fit_lsi_batched|"
    "project_spectra|FilterBank|FusedChiSquareDetector|fit_dsb|"
    "fit_lsi|fit_eac|LSIFilter|EACFilter"
)
GROUP2 = re.compile(
    r"\bdtfit\.(?:%s)\b" % _CORE_NAMES
    + r"|from\s+dtfit\s+import[^\n]*\b(?:%s|ensemble_fit)\b" % _CORE_NAMES
    + r"|\bdtfit\.reference\b"
    + r"|Promoted to the stable|stable library primitive"
    + r"|re-exported from `?dtfit`?|PROMOTED as `?dtfit\."
)

# Group 3: old EAC-robustness claims.
GROUP3 = re.compile(
    r"most noise-robust|noise-robust batch|Why it is robust"
    r"|EAC[^\n]*\b(?:the noise-robust|the outlier-robust)\b"
)

# Group 4: legacy-ignored core keywords shown as live fit arguments, scoped
# to the core surface pages (experimental pages document live experimental
# params of the same spelling and are excluded).
GROUP4_PAGES = {
    WIKI / "Methods-EAC.md", WIKI / "Methods-LSI.md",
    WIKI / "Guides-Choosing-a-Method.md", WIKI / "API-Fitting.md",
    WIKI / "API.md", WIKI / "Comparison.md",
    DOCS / "guide" / "eac.md", DOCS / "guide" / "lsi.md",
    DOCS / "guide" / "choosing-a-method.md", DOCS / "comparison.md",
}
GROUP4 = re.compile(r"\b(window_mode|active_ratio|f_scale)\b")

# Group 5: the removed experiments tree, its report pages and the
# adaptations removed from dtfit_experimental. The experiments are the
# notebooks under packages/dtfit-experimental/experiments.
GROUP5 = re.compile(
    r"dtfit_experimental[./]experiments\b|dtfit_experimental\.scale\b"
    r"|cases/REPORTS\.md|domains/DOMAINS\.md|domains/\*/report\.md"
    r"|\b(fit_lsi_basis|fit_joint|boosted_fit|JointResult|BoostedModel)\b"
    r"|\]\((?:Cases-[A-Za-z0-9-]+|Domains-Reports)[)#]"
)


def _hits(pattern, pages):
    bad = []
    for p in pages:
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                bad.append(
                    "{}:{}: {}".format(p.relative_to(ROOT), i, line.strip())
                )
    return bad


def test_no_fully_removed_symbol():
    assert _hits(GROUP1, ALL_PAGES) == []


def test_no_core_qualified_experimental_symbol():
    assert _hits(GROUP2, ALL_PAGES) == []


def test_no_old_eac_robustness_claim():
    assert _hits(GROUP3, ALL_PAGES) == []


def test_no_legacy_keyword_as_core_argument():
    assert _hits(GROUP4, sorted(GROUP4_PAGES)) == []


def test_no_removed_experiments_tree():
    """Fails under a page that links a Cases-* or Domains-Reports page,
    names dtfit_experimental.experiments or dtfit_experimental.scale, or
    documents fit_lsi_basis, fit_joint or boosted_fit."""
    assert _hits(GROUP5, ALL_PAGES) == []
