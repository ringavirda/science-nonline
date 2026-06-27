#!/usr/bin/env python3
"""Generate the generated GitHub-wiki pages from their in-repo sources.

The prose wiki pages (guides / api / methods / experimental and ``Home``) are
maintained **directly** in the repo's ``wiki/`` directory. This script (re)builds
only the pages that have a generated source of truth, merging them into ``wiki/``
without touching the hand-edited pages:

  * ``docs/guides/notebooks/*.ipynb``          the tutorial notebooks (-> markdown)
  * the ``dtfit-experimental`` experiment and  per-case / per-domain reports
    domain reports

  * ``packages/dtfit/examples/*.py``          the runnable example scripts
    (rendered to guide pages, docstring + source)

It also refreshes the ``_Sidebar`` and ``_Footer`` and copies every referenced
figure. A companion workflow (``.github/workflows/wiki.yml``) pushes ``wiki/`` to
the wiki git repo. (This script was also the one-time bootstrap that produced the
prose pages from the original ``docs/`` tree before they moved into ``wiki/``.)

What it does to each page:

  * **flattens** the folder tree to flat hyphenated page names
    (``docs/methods/lsi.md`` -> ``Methods-LSI``);
  * **rewrites cross-links** -- relative ``.md`` / ``.ipynb`` links become wiki
    page links, links into package source become absolute GitHub blob URLs, and
    image links are copied into the wiki and repointed;
  * **normalises to ASCII** so the pages render the same for everyone -- typographic
    Unicode (em dashes, arrows, superscripts, Greek, etc.) is transliterated, while
    fenced code (box-drawing is mapped 1:1 to keep diagrams aligned) and ``$...$``
    math (rendered by GitHub's MathJax) are preserved.

Run from anywhere::

    python scripts/build_wiki.py                  # refresh generated pages in wiki/
    python scripts/build_wiki.py --out build/wiki # write somewhere else
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
EXP = REPO / "packages/dtfit-experimental/src/dtfit_experimental/experiments"
EXAMPLES = REPO / "packages/dtfit/examples"

GH_REPO = "ringavirda/science-pylab"
GH_BLOB = f"https://github.com/{GH_REPO}/blob/main/"

# Words that should be fully upper-cased in generated page titles.
ACRONYMS = {
    "lsi", "eac", "dsb", "api", "gpu", "gil", "gemm", "ltsf", "gps",
    "rlc", "ac", "am", "eac", "id", "cpu", "io",
}

IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


# --------------------------------------------------------------------------- #
# Page-name mapping
# --------------------------------------------------------------------------- #
SMALL_WORDS = {"and", "a", "an", "of", "the", "to", "in", "for",
               "vs", "or", "on", "with", "by"}


def _title(stem: str) -> str:
    """``map_reduce_partitioned`` -> ``Map-Reduce-Partitioned``.

    Acronyms are upper-cased; small joiner words stay lower-case (except as the
    first word) so names read naturally, e.g. ``Choosing-a-Method``.
    """
    parts = [p for p in re.split(r"[-_\s]+", stem) if p]
    out = []
    for i, p in enumerate(parts):
        low = p.lower()
        if low in ACRONYMS:
            out.append(p.upper())
        elif low in SMALL_WORDS and i != 0:
            out.append(low)
        else:
            out.append(p.capitalize())
    return "-".join(out)


def _build_page_map() -> dict[str, str]:
    """Map every source file (repo-relative POSIX path) to its wiki page name."""
    m: dict[str, str] = {}

    def rel(p: Path) -> str:
        return p.relative_to(REPO).as_posix()

    # ---- docs/ -----------------------------------------------------------
    m[rel(DOCS / "README.md")] = "Home"
    section_title = {
        "guides": "Guides",
        "api": "API",
        "methods": "Methods",
        "experimental": "Experimental",
    }
    for section, title in section_title.items():
        sec = DOCS / section
        for p in sorted(sec.rglob("*.md")):
            r = rel(p)
            if "notebooks" in p.parts:
                continue  # handled below
            if p.name == "README.md" and p.parent == sec:
                m[r] = title
            elif p.name == "README.md":
                m[r] = f"{title}-{_title(p.parent.name)}"
            else:
                m[r] = f"{title}-{_title(p.stem)}"

    # examples (runnable scripts -> rendered guide pages)
    m[rel(EXAMPLES / "README.md")] = "Examples"
    for p in sorted(EXAMPLES.glob("*.py")):
        m[rel(p)] = f"Example-{_title(p.stem)}"

    # ---- experiments -----------------------------------------------------
    m[rel(EXP / "README.md")] = "Experiments"
    m[rel(EXP / "cases/REPORTS.md")] = "Cases-Reports"
    m[rel(EXP / "cases/analysis/README.md")] = "Cases-Analysis"
    for p in sorted((EXP / "cases/analysis").glob("*.md")):
        if p.name == "README.md":
            continue
        m[rel(p)] = f"Cases-Analysis-{_title(p.stem)}"
    for p in sorted((EXP / "cases").glob("*/report.md")):
        m[rel(p)] = f"Case-{_title(p.parent.name)}"
    m[rel(EXP / "domains/README.md")] = "Domains"
    m[rel(EXP / "domains/DOMAINS.md")] = "Domains-Reports"
    for p in sorted((EXP / "domains").glob("*/report.md")):
        m[rel(p)] = f"Domain-{_title(p.parent.name)}"

    return m


# --------------------------------------------------------------------------- #
# ASCII normalisation
# --------------------------------------------------------------------------- #
# Applied everywhere except protected (code / math) spans.
PROSE_MAP = {
    "—": "--", "–": "-", "―": "--", "‒": "-",
    "→": "->", "←": "<-", "↔": "<->", "⇒": "=>",
    "⇐": "<=", "↦": "|->", "⟶": "->", "⟵": "<-",
    "≈": "~=", "≅": "~=", "≃": "~=", "≤": "<=",
    "≥": ">=", "≠": "!=", "≡": "==", "∝": "~",
    "≲": "<~", "≳": ">~", "≪": "<<", "≫": ">>",
    "−": "-", "∓": "-/+", "⁄": "/", "‰": " per mille",
    "±": "+/-", "×": "x", "÷": "/", "·": "*",
    "∗": "*", "∙": "*", "•": "-", "∘": "o",
    "°": " deg", "′": "'", "″": '"',
    "²": "^2", "³": "^3", "¹": "^1", "⁴": "^4",
    "⁰": "^0", "ⁿ": "^n", "⁺": "^+", "⁻": "^-",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3",
    "₄": "_4", "ₖ": "_k", "ⱼ": "_j", "ᵢ": "_i",
    "ₙ": "_n",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "…": "...",
    "∞": "inf", "√": "sqrt", "∑": "sum", "∫": "integral",
    "∂": "d", "∇": "grad", "∈": "in", "∉": "not in",
    "⊂": "subset", "⊆": "subseteq", "∩": "cap", "∪": "cup",
    "∀": "forall", "∃": "exists", "¬": "not", "∧": "and",
    "∨": "or", "∼": "~", "∥": "||", "∅": "empty",
    "⊥": "perp", "∠": "angle",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi",
    "ψ": "psi", "ω": "omega", "ϕ": "phi",
    "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Π": "Pi",
    "Σ": "Sigma", "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi",
    "Ω": "Omega", "Γ": "Gamma",
    "✓": "yes", "✔": "yes", "✗": "no", "✘": "no",
    "©": "(c)", "®": "(R)", "™": "(TM)",
    "€": "EUR", "£": "GBP",
    "½": "1/2", "¼": "1/4", "¾": "3/4",
    "§": "Sec.", "µ": "u", "⭐": "*", "〔": "[", "〕": "]",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}

# Applied inside fenced/inline code -- strictly one char -> one char so that
# the column alignment of ASCII-art diagrams survives.
CODE_MAP = {
    "─": "-", "━": "-", "│": "|", "┃": "|",
    "┄": "-", "┅": "-", "┈": "-", "┉": "-",
    "┌": "+", "┍": "+", "┎": "+", "┏": "+",
    "┐": "+", "┑": "+", "┒": "+", "┓": "+",
    "└": "+", "┕": "+", "┖": "+", "┗": "+",
    "┘": "+", "┙": "+", "┚": "+", "┛": "+",
    "├": "+", "┠": "+", "┤": "+", "┨": "+",
    "┬": "+", "┰": "+", "┴": "+", "┸": "+",
    "┼": "+", "╀": "+",
    "╭": "+", "╮": "+", "╯": "+", "╰": "+",
    "═": "=", "║": "|", "╔": "+", "╗": "+",
    "╚": "+", "╝": "+", "╠": "+", "╣": "+",
    "╦": "+", "╩": "+", "╬": "+",
    "↑": "^", "↓": "v", "→": ">", "←": "<",
    "⮞": ">", "⮜": "<",
    "▶": ">", "◀": "<", "▲": "^", "▼": "v",
    "▪": "*", "▫": ".", "●": "*", "○": "o",
    "█": "#", "▓": "#", "▒": ":", "░": ".",
    "·": ".", "…": "...", "—": "-", "–": "-", "−": "-",
    " ": " ", " ": " ",
}


def _placeholder(i: int) -> str:
    return f"\x00{i}\x00"


def _ascii_code(text: str) -> str:
    return "".join(CODE_MAP.get(ch, ch) for ch in text)


def to_ascii(text: str, unknown: set[str]) -> str:
    """Transliterate to ASCII, preserving code fences and ``$...$`` math."""
    stash: list[str] = []

    def protect(pattern: str, transform=None, flags=0):
        def repl(mo: re.Match) -> str:
            s = mo.group(0)
            if transform:
                s = transform(s)
            stash.append(s)
            return _placeholder(len(stash) - 1)
        return lambda t: re.sub(pattern, repl, t, flags=flags)

    # Order matters: outermost constructs first.
    text = protect(r"```.*?```", _ascii_code, re.DOTALL)(text)   # fenced code
    text = protect(r"\$\$.*?\$\$", None, re.DOTALL)(text)        # display math
    text = protect(r"`[^`\n]+?`", _ascii_code)(text)            # inline code
    text = protect(r"\$[^$\n]+?\$", None)(text)                 # inline math

    # Map known prose Unicode.
    out = []
    for ch in text:
        if ord(ch) < 128 or ch == "\x00":
            out.append(ch)
        elif ch in PROSE_MAP:
            out.append(PROSE_MAP[ch])
        else:
            # Best-effort: strip accents, else drop with a record.
            norm = unicodedata.normalize("NFKD", ch)
            stripped = "".join(c for c in norm if ord(c) < 128)
            if stripped:
                out.append(stripped)
            else:
                unknown.add(ch)
                out.append("?")
    text = "".join(out)

    # Restore protected spans (reverse order is unnecessary -- tokens are unique).
    for i, original in enumerate(stash):
        text = text.replace(_placeholder(i), original)
    return text


# --------------------------------------------------------------------------- #
# Link rewriting
# --------------------------------------------------------------------------- #
LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")

# resolve() sentinels: keep the original link, or drop it but keep the label.
KEEP = object()
STRIP = object()


def rewrite_links(text: str, src_rel: str, page_map: dict[str, str],
                  figures: dict[str, Path]) -> str:
    """Rewrite every markdown link/image target relative to ``src_rel``."""
    src_dir = Path(src_rel).parent

    def resolve(target: str) -> str:
        # Split off any #fragment / ?query.
        frag = ""
        mo = re.search(r"[#?].*$", target)
        if mo:
            frag, target = mo.group(0), target[: mo.start()]

        # Leave absolute URLs, mailto, and pure same-page anchors untouched.
        if not target:  # was a bare "#anchor"
            return KEEP
        if re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
            return KEEP

        # Resolve relative to the source file's directory.
        resolved = (src_dir / target).as_posix()
        resolved = Path(resolved)
        # Normalise .. without touching the filesystem.
        norm = []
        for part in resolved.parts:
            if part == "..":
                if norm:
                    norm.pop()
            elif part != ".":
                norm.append(part)
        rp = "/".join(norm)

        ext = Path(rp).suffix.lower()

        # Image -> copy into the wiki and point at figures/.
        if ext in IMG_EXT:
            abs_img = REPO / rp
            if abs_img.exists():
                dest = _figure_dest(rp, figures)
                figures[dest] = abs_img
                return f"figures/{dest}{frag}"
            return STRIP

        # Known page (markdown / notebook, or a directory's README).
        if rp in page_map:
            return page_map[rp] + frag
        readme = f"{rp}/README.md" if rp else ""
        if readme in page_map:
            return page_map[readme] + frag
        # A directory that maps via its index file name.
        for idx in ("REPORTS.md", "DOMAINS.md"):
            if f"{rp}/{idx}" in page_map:
                return page_map[f"{rp}/{idx}"] + frag

        # Anything else that exists in the repo (source code, etc.) -> blob URL.
        if rp and (REPO / rp).exists():
            return GH_BLOB + rp + frag
        # Unresolved relative path (e.g. a stale source link) -> drop the link.
        return STRIP

    def repl(mo: re.Match) -> str:
        bang, label, target = mo.group(1), mo.group(2), mo.group(3).strip()
        new = resolve(target)
        if new is KEEP:
            return mo.group(0)
        if new is STRIP:
            # Keep the (often code-styled) label, drop the dead link.
            return label if not bang else ""
        return f"{bang}[{label}]({new})"

    return LINK_RE.sub(repl, text)


def _figure_dest(rp: str, figures: dict[str, Path]) -> str:
    """Pick a collision-free flat filename for a figure."""
    base = Path(rp).name
    if base not in figures or figures[base] == REPO / rp:
        return base
    # Collision with a different source -> prefix with parent folder.
    prefix = Path(rp).parent.name
    return f"{prefix}-{base}"


# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #
def _run_example(path: Path) -> str | None:
    """Execute an example headless and return its captured output.

    The examples are deterministic (fixed RNG seeds) and print their results, so
    the captured stdout is a stable snapshot to embed under the source. Returns
    ``None`` if the script could not be run (e.g. the package is not installed in
    this environment); a non-zero exit appends stderr so failures are visible.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, timeout=300,
            cwd=str(path.parent.parent),  # so "python examples/<f>.py" paths read
        )
    except Exception:
        return None
    out = proc.stdout.rstrip()
    if proc.returncode != 0:
        out = (out + "\n--- stderr ---\n" + proc.stderr).strip()
    return out


def render_example(path: Path, page: str, out_dir: Path, unknown: set[str]) -> None:
    """Render a runnable example script as a markdown guide page.

    The module docstring becomes the page intro; the source is embedded in a
    ``python`` fence (with a GitHub link), followed by the example's captured
    output so the page shows what running it produces. Examples are authored
    ASCII-only, so ``to_ascii`` is a safety net.
    """
    src = path.read_text(encoding="utf-8")
    mo = re.match(r'\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', src, re.DOTALL)
    if mo:
        doc = (mo.group(1) or mo.group(2) or "").strip()
        body_src = src[mo.end():].lstrip("\n")
    else:
        doc, body_src = "", src
    rel = path.relative_to(REPO).as_posix()
    title = page.replace("Example-", "").replace("-", " ")
    run_cmd = f"python examples/{path.name}"

    parts = [f"# Example {title}", ""]
    if doc:
        parts += [doc, ""]
    parts += [f"Source: [`{rel}`]({GH_BLOB}{rel})", "", "```python",
              body_src.rstrip(), "```"]

    output = _run_example(path)
    if output:
        parts += ["", f"## Output (`{run_cmd}`)", "", "```text",
                  output.rstrip(), "```"]

    text = to_ascii("\n".join(parts), unknown)
    (out_dir / f"{page}.md").write_text(text.rstrip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Sidebar / Home
# --------------------------------------------------------------------------- #
def make_sidebar() -> str:
    groups = [
        ("Start here", [("Home", "Home")]),
        ("Guides", [
            ("Overview", "Guides"),
            ("Methods explained", "Guides-Methods-Explained"),
            ("Lineage & variants", "Guides-Lineage-and-Variants"),
            ("Choosing a method", "Guides-Choosing-a-Method"),
        ]),
        ("Examples", [
            ("Overview", "Examples"),
            ("01 Quickstart", "Example-01-Quickstart"),
            ("02 Fitting methods", "Example-02-Fitting-Methods"),
            ("03 Models & auto", "Example-03-Models-and-Auto"),
            ("04 sklearn estimator", "Example-04-Sklearn-Estimator"),
            ("05 Streaming", "Example-05-Streaming"),
            ("06 Scaling", "Example-06-Scaling"),
            ("07 Diagnostics", "Example-07-Diagnostics"),
        ]),
        ("API reference", [
            ("Overview", "API"),
            ("Fitting", "API-Fitting"),
            ("Types", "API-Types"),
            ("Estimator", "API-Estimator"),
            ("Auto", "API-Auto"),
            ("Models", "API-Models"),
            ("Stochastic", "API-Stochastic"),
            ("Streaming", "API-Streaming"),
            ("Scaling", "API-Scaling"),
            ("Diagnostics", "API-Diagnostics"),
        ]),
        ("Methods (math)", [
            ("Overview", "Methods"),
            ("DSB", "Methods-DSB"),
            ("LSI", "Methods-LSI"),
            ("EAC", "Methods-EAC"),
            ("Ensemble", "Methods-Ensemble"),
            ("EACFilter", "Methods-Equal-Areas-Filter"),
            ("LSIFilter", "Methods-Legendre-Filter"),
            ("Filter bank", "Methods-Filter-Bank"),
            ("Scaling", "Methods-Scaling"),
            ("Auto", "Methods-Auto"),
            ("Stochastic", "Methods-Stochastic"),
        ]),
        ("Experimental", [
            ("Overview", "Experimental"),
            ("Adaptations API", "Experimental-Adaptations-API"),
            ("Baselines", "Experimental-Baselines"),
        ]),
        ("Validation", [
            ("Experiments", "Experiments"),
            ("Cases - reports", "Cases-Reports"),
            ("Cases - analysis", "Cases-Analysis"),
            ("Domains", "Domains"),
            ("Domains - reports", "Domains-Reports"),
        ]),
    ]
    lines = ["### dtfit wiki", ""]
    for title, items in groups:
        lines.append(f"**{title}**")
        lines.append("")
        for label, page in items:
            lines.append(f"- [{label}]({page})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def make_footer() -> str:
    return (
        "---\n"
        "_Published automatically from the repo's `wiki/` directory by the "
        f"[wiki workflow]({GH_BLOB}.github/workflows/wiki.yml). The notebook and "
        "experiment/domain report pages are generated by "
        f"[`scripts/build_wiki.py`]({GH_BLOB}scripts/build_wiki.py) from their "
        "sources; the other pages are edited directly in `wiki/`._\n"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "wiki"),
                    help="output directory (default: wiki/)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    page_map = _build_page_map()
    figures: dict[str, Path] = {}
    unknown: set[str] = set()

    # Markdown pages. We merge into the output rather than wiping it: the prose
    # pages (guides / api / methods / experimental) are now maintained directly
    # in the wiki, so only the sources that still exist -- the notebooks and the
    # experiment/domain reports -- are (re)generated here.
    for src_rel, page in sorted(page_map.items()):
        src = REPO / src_rel
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        text = rewrite_links(text, src_rel, page_map, figures)
        text = to_ascii(text, unknown)
        (out_dir / f"{page}.md").write_text(text.rstrip() + "\n", encoding="utf-8")

    # Examples (rendered from the runnable scripts).
    for ex in sorted(EXAMPLES.glob("*.py")):
        page = f"Example-{_title(ex.stem)}"
        render_example(ex, page, out_dir, unknown)

    # Copy referenced figures.
    for dest, src in figures.items():
        shutil.copyfile(src, out_dir / "figures" / dest)

    # Navigation chrome.
    (out_dir / "_Sidebar.md").write_text(make_sidebar(), encoding="utf-8")
    (out_dir / "_Footer.md").write_text(make_footer(), encoding="utf-8")

    n_pages = len(list(out_dir.glob("*.md")))
    n_figs = len(list((out_dir / "figures").glob("*")))
    print(f"wrote {n_pages} pages and {n_figs} figures to {out_dir}")
    if unknown:
        chars = ", ".join(f"U+{ord(c):04X}" for c in sorted(unknown))
        sys.stdout.buffer.write(
            f"WARNING: {len(unknown)} unmapped non-ASCII char(s): {chars}\n"
            .encode("ascii", "replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
