"""Generalized driver shared by the experiment suites (cases & domains).

Both the per-adaptation ``cases`` suite and the per-domain ``domains`` suite are
the same shape: a registry of self-contained experiment folders, each exposing a
``run.main(quick=bool)`` that writes its own ``report.md`` + ``figures/``. They
are independent (separate folders, no shared state), so they run **in parallel**
across worker processes -- the suite finishes in about the time of its slowest
member instead of the sum. Afterwards an index is regenerated.

Processes (not threads) are used because the experiments are CPU-bound and use
matplotlib's non-thread-safe global ``pyplot`` state; separate processes give
true parallelism and full isolation.

A suite is described by a :class:`Suite` (its package, its experiment registry,
and how to render its index); :func:`main` wires up the CLI and runs it. This
replaces the two previously copy-pasted ``run_suite.py`` / ``run_domains.py``
drivers, which now only declare their :class:`Suite` and call :func:`main`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

# (name, title, runtime_seconds, status)
Result = tuple[str, str, float, str]


@dataclass
class Suite:
    """Declarative description of one experiment suite.

    Attributes:
        package: importable package holding the experiment subpackages, e.g.
            ``"dtfit_experimental.experiments.cases"``. Each experiment ``name``
            resolves to ``<package>.<name>.run`` with a ``main(quick=bool)``.
        experiments: ordered ``(name, title)`` registry.
        index_path: where the regenerated index markdown is written.
        index_title: H1 of the index.
        intro: paragraph(s) under the H1 (a list of markdown blocks).
        noun: singular label used in console/index ("experiment" / "domain").
        numbered: whether the index table has a leading ``#`` column.
        extra_sections: optional callback adding suite-specific markdown sections
            (e.g. the adaptation-effectiveness matrix) after the index table.
    """

    package: str
    experiments: Sequence[tuple[str, str]]
    index_path: Path
    index_title: str
    intro: Sequence[str]
    noun: str = "experiment"
    numbered: bool = True
    extra_sections: Callable[[list[Result]], list[str]] | None = None
    _extra: dict = field(default_factory=dict)  # free-form data for extra_sections


def _run_one(task: tuple[str, str, str, bool]) -> Result:
    """Run a single experiment, possibly in a worker process.

    Module-level so it is picklable for ``ProcessPoolExecutor`` (spawn start
    method on Windows). Exceptions are caught and reported so one failing
    experiment never tears down the pool.
    """
    package, name, title, quick = task
    print(f"=== START {name} ===", flush=True)
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(f"{package}.{name}.run")
        mod.main(quick=quick)
        status = "ok"
    except Exception:
        status = "FAILED"
        traceback.print_exc()
    elapsed = time.perf_counter() - t0
    print(f"=== {status:7s} {name} ({elapsed:.0f}s) ===", flush=True)
    return name, title, elapsed, status


def run_all(suite: Suite, quick: bool, jobs: int) -> list[Result]:
    tasks = [(suite.package, name, title, quick) for name, title in suite.experiments]
    if jobs <= 1:
        return [_run_one(t) for t in tasks]
    # Preserve registry order for the index regardless of finish order.
    results: list[Result | None] = [None] * len(tasks)
    print(f"Running {len(tasks)} {suite.noun}s across {jobs} worker processes...\n")
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut_to_idx = {ex.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()
    return [r for r in results if r is not None]


def write_index(suite: Suite, results: list[Result]) -> Path:
    lines = [f"# {suite.index_title}\n", *suite.intro]
    if suite.numbered:
        lines += [f"\n## {suite.noun.capitalize()}s\n",
                  f"| # | {suite.noun} | report | status | runtime (s) |",
                  "|---|---|---|---|---|"]
        for i, (name, title, secs, status) in enumerate(results, 1):
            lines.append(f"| {i} | {title} | [{name}/report.md]({name}/report.md) "
                         f"| {status} | {secs:.0f} |")
    else:
        lines += [f"\n## {suite.noun.capitalize()}s\n",
                  f"| {suite.noun} | report | status | runtime (s) |",
                  "|---|---|---|---|"]
        for name, title, secs, status in results:
            lines.append(f"| {title} | [{name}/report.md]({name}/report.md) "
                         f"| {status} | {secs:.0f} |")
    if suite.extra_sections is not None:
        lines += suite.extra_sections(results)
    suite.index_path.write_text("\n".join(lines), encoding="utf-8")
    return suite.index_path


def _default_jobs(suite: Suite) -> int:
    return min(len(suite.experiments), os.cpu_count() or 4)


def main(suite: Suite) -> None:
    """CLI entry point for a suite: parse args, run, regenerate the index."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--jobs", "-j", type=int, default=_default_jobs(suite),
                    help=f"worker processes (default: min(#{suite.noun}s, #CPUs); "
                         "1 = serial). Lower it if memory-heavy "
                         f"{suite.noun}s (big-data / GPU) contend.")
    args = ap.parse_args()
    jobs = max(1, args.jobs)

    t0 = time.perf_counter()
    results = run_all(suite, args.quick, jobs)
    wall = time.perf_counter() - t0

    idx = write_index(suite, results)
    print(f"\nWrote index: {idx}")
    for name, _, secs, status in results:
        print(f"  {status:7s} {name} ({secs:.0f}s)")
    total = sum(secs for _, _, secs, _ in results)
    mode = "serial" if jobs == 1 else f"{jobs} workers"
    print(f"\nWall-clock: {wall:.0f}s ({mode}); "
          f"summed {suite.noun} time: {total:.0f}s")
