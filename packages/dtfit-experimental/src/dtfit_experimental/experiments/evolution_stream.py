"""The streaming and map-reduce matrices of the evolution.

Streaming: the recursive forms of the integral criteria
(``dtfit_legacy.streaming.LSIFilter`` and ``EACFilter``) against
``dtfit.ImageFilter`` in both bases, on a sinusoid whose amplitude jumps
mid-stream and on one whose amplitude drifts. Metrics per filter: the RMSE
of the tracked amplitude on the stable segments after burn-in, the delay
between the jump and the first drift flag, the number of flags before the
jump, and the cost per step.

Map-reduce: ``dtfit_legacy.scale.PartitionedLSI`` (integral accumulator)
against ``dtfit.Image`` merging (discrete image) on a long exponential
decay reduced in chunks: the merged estimate against the whole-data
estimate, and the wall time of the reduce.

    python -m dtfit_experimental.experiments.evolution_stream --out DIR
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time
import warnings

import numpy as np

from dtfit import Image, ImageFilter, Original, fit
from dtfit_legacy.scale import PartitionedLSI
from dtfit_legacy.streaming import EACFilter, LSIFilter

W, DT = 60, 0.1
JUMP_AT = 30.0
T_END = 60.0


def _stream(kind, seed, noise=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T_END, DT)
    if kind == "jump":
        amp = np.where(t < JUMP_AT, 2.0, 3.5)
    else:
        amp = 2.0 + 1.5 * t / T_END
    y = amp * np.sin(1.5 * t) + noise * 2.0 * rng.standard_normal(t.size)
    return t, y, amp


def _filters():
    return [
        ("legacy_lsi", lambda: LSIFilter("A*sin(w*t)", "t", window_size=W,
                                         order=5, adaptive_window=False)),
        ("legacy_eac", lambda: EACFilter("A*sin(w*t)", "t", window_size=W,
                                         adaptive_window=False)),
        ("image_legendre", lambda: ImageFilter(
            "A*sin(w*t)", "t", basis="legendre", order=5, window_size=W,
            adaptive_window=False)),
        ("image_block", lambda: ImageFilter(
            "A*sin(w*t)", "t", basis="block", order=2, window_size=W,
            adaptive_window=False)),
        ("image_legendre_adaptive", lambda: ImageFilter(
            "A*sin(w*t)", "t", basis="legendre", order=5, window_size=W,
            adaptive_window=True)),
    ]


def _set_p0(flt, p0):
    try:
        flt.p = np.asarray(p0, dtype=float)
    except Exception:  # noqa: BLE001
        pass


def streaming_matrix(out: pathlib.Path, seeds: int) -> None:
    path = out / "streaming.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filter", "scenario", "seed", "rmse_amp", "delay",
                    "false_alarms", "step_us", "error"])
        for scen in ("jump", "drift"):
            for seed in range(seeds):
                t, y, amp = _stream(scen, seed)
                for name, make in _filters():
                    try:
                        flt = make()
                        _set_p0(flt, [2.0, 1.5])
                        est = np.full(t.size, np.nan)
                        flags = np.zeros(t.size, dtype=bool)
                        tic = time.perf_counter()
                        for i in range(t.size):
                            flt.partial_fit(float(t[i]), float(y[i]))
                            p = np.asarray(getattr(flt, "p"), dtype=float)
                            est[i] = p[0]
                            flags[i] = bool(getattr(flt, "drift_flag_",
                                                    False))
                        step_us = 1e6 * (time.perf_counter() - tic) / t.size
                        burn = 2 * W
                        if scen == "jump":
                            j = int(np.searchsorted(t, JUMP_AT))
                            stable = np.zeros(t.size, dtype=bool)
                            stable[burn:j] = True
                            stable[j + 2 * W:] = True
                            after = np.flatnonzero(flags[j:])
                            delay = float(after[0] * DT) if after.size \
                                else float("nan")
                            false = int(flags[burn:j].sum())
                        else:
                            stable = np.zeros(t.size, dtype=bool)
                            stable[burn:] = True
                            delay, false = float("nan"), int(
                                flags[burn:].sum())
                        rmse = float(np.sqrt(np.nanmean(
                            (est[stable] - amp[stable]) ** 2)))
                        w.writerow([name, scen, seed, rmse, delay, false,
                                    step_us, ""])
                    except Exception as exc:  # noqa: BLE001
                        w.writerow([name, scen, seed, "", "", "", "",
                                    type(exc).__name__])
                fh.flush()
            print(f"streaming {scen}: done", flush=True)


def mapreduce_matrix(out: pathlib.Path, sizes=(200_000, 1_000_000),
                     chunk=10_000, order=6) -> None:
    path = out / "mapreduce.csv"
    expr, var, dom = "a*exp(b*t)", "t", (0.0, 10.0)
    truth = np.array([3.0, -0.4])
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "max_rel_diff_vs_whole", "err_vs_truth",
                    "reduce_s", "fit_s", "error"])
        for n in sizes:
            rng = np.random.default_rng(0)
            t = np.linspace(*dom, n)
            y = truth[0] * np.exp(truth[1] * t) + 0.05 * 3.0 * \
                rng.standard_normal(n)
            p0 = truth * 1.25
            tic = time.perf_counter()
            whole = fit(expr, Original(t, y, domain=dom), var,
                        basis="legendre", order=order, p0=p0).coeffs
            t_whole = time.perf_counter() - tic
            w.writerow(["whole_image_fit", n, 0.0,
                        float(np.max(np.abs(whole / truth - 1))), 0.0,
                        t_whole, ""])
            # the discrete image, chunk by chunk
            try:
                tic = time.perf_counter()
                merged = None
                for i in range(0, n, chunk):
                    img = Image.of(Original(t[i:i + chunk], y[i:i + chunk],
                                            domain=dom), "legendre", order)
                    merged = img if merged is None else merged.merge(img)
                t_red = time.perf_counter() - tic
                tic = time.perf_counter()
                c = fit(expr, merged, var, basis="legendre", p0=p0).coeffs
                t_fit = time.perf_counter() - tic
                w.writerow(["image_merge", n,
                            float(np.max(np.abs(c / whole - 1))),
                            float(np.max(np.abs(c / truth - 1))), t_red,
                            t_fit, ""])
            except Exception as exc:  # noqa: BLE001
                w.writerow(["image_merge", n, "", "", "", "",
                            type(exc).__name__])
            # the integral accumulator
            try:
                tic = time.perf_counter()
                acc = None
                for i in range(0, n, chunk):
                    part = PartitionedLSI(expr, var, domain=dom, order=order)
                    part.update(t[i:i + chunk], y[i:i + chunk])
                    acc = part if acc is None else acc.merge(part)
                t_red = time.perf_counter() - tic
                tic = time.perf_counter()
                c = acc.fit(p0=p0).coeffs
                t_fit = time.perf_counter() - tic
                w.writerow(["legacy_partitioned", n,
                            float(np.max(np.abs(c / whole - 1))),
                            float(np.max(np.abs(c / truth - 1))), t_red,
                            t_fit, ""])
            except Exception as exc:  # noqa: BLE001
                w.writerow(["legacy_partitioned", n, "", "", "", "",
                            type(exc).__name__])
            fh.flush()
            print(f"mapreduce n={n}: done", flush=True)


def report(out: pathlib.Path) -> None:
    import pandas as pd

    lines = ["# Streaming and map-reduce matrices", ""]
    sp = out / "streaming.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        g = s.groupby(["scenario", "filter"]).agg(
            rmse=("rmse_amp", "median"), delay=("delay", "median"),
            false=("false_alarms", "median"), step_us=("step_us", "median"),
            ok=("error", lambda c: float(c.isna().mean()))).reset_index()
        lines += ["## streaming: median over seeds", "",
                  "| scenario | filter | amplitude RMSE | detection delay, s "
                  "| false alarms | us per step | ok |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.scenario} | {r['filter']} | {r.rmse:.3f} | "
                         f"{r.delay:.1f} | {r.false:.0f} | {r.step_us:.0f} | "
                         f"{100 * r.ok:.0f}% |")
        lines.append("")
    mp = out / "mapreduce.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        lines += ["## map-reduce", "",
                  "| method | n | max rel diff vs whole | err vs truth | "
                  "reduce, s | fit, s |", "|---|---|---|---|---|---|"]
        for _, r in m.iterrows():
            lines.append(f"| {r.method} | {int(r.n)} | "
                         f"{r.max_rel_diff_vs_whole:.2e} | "
                         f"{100 * r.err_vs_truth:.3f}% | {r.reduce_s:.2f} | "
                         f"{r.fit_s:.2f} |" if r.error != r.error else
                         f"| {r.method} | {int(r.n)} | failed: {r.error} |")
        lines.append("")
    (out / "summary_stream.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--streaming", action="store_true",
                    help="the streaming matrix only")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    if a.report:
        report(out)
        return
    if a.smoke:
        streaming_matrix(out, 1)
        mapreduce_matrix(out, sizes=(20_000,), chunk=5_000)
        report(out)
        return
    streaming_matrix(out, a.seeds)
    if not a.streaming:
        mapreduce_matrix(out)
    report(out)


if __name__ == "__main__":
    main()
