# Example 05 Streaming

Streaming / online trackers.

These estimators ingest one sample at a time with bounded per-update cost
(partial_fit(t, y)), for control loops and big-data streams. Each filter is the
streaming twin of a batch method and carries built-in drift detection. Start from
the .tracking() / .robust() presets instead of the ~20 raw knobs.

- EACFilter           -- streaming equal-areas (twin of fit_eac).
- LSIFilter           -- streaming Legendre spectrum (twin of fit_lsi).
- FilterBank          -- many independent streams updated in lockstep.
- FusedChiSquareDetector -- pools a bank's innovations into one fault test.

Run headless:   python examples/05_streaming.py

Source: [`packages/dtfit/examples/05_streaming.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/examples/05_streaming.py)

```python
import numpy as np

from dtfit import EACFilter, FilterBank


def track_drifting_parameter(rng) -> None:
    # The exponential rate b jumps mid-stream; the filter re-adapts (its drift
    # test detects the change and re-arms the covariance).
    T = 500
    t = np.linspace(0, 8, T)
    b_true = np.where(t < 4, 0.30, 0.55)
    y = np.exp(b_true * t) + rng.normal(0, 0.05, T)

    flt = EACFilter("exp(b*t)", "t", p0=[0.2], window_size=40, q_diag=[1e-4], r=0.5)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    print("== EACFilter: track a mid-stream step ==")
    print("final b estimate:", round(flt.params_["b"], 3), " (true 0.55)")
    print("drifts detected :", flt.n_drifts_)


def preset(rng) -> None:
    # The .tracking() preset turns on auto window sizing; .robust() turns on the
    # outlier-resilient gains. Both keep the full kwargs for overrides.
    t = np.linspace(0, 6, 300)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 0.05, t.size)
    flt = EACFilter.tracking("A*sin(w*x)", "x")
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    print("\n== EACFilter.tracking() preset ==")
    print("params:", {k: round(v, 3) for k, v in flt.params_.items()})


def filter_bank(rng) -> None:
    # Build K identically-configured filters for one model and drive them over a
    # block of samples; run() returns final per-stream params and drift counts.
    K = 4
    t = np.linspace(0, 20, 400)
    b_true = np.array([0.30, 0.50, 0.70, 0.90])
    Y = np.column_stack([np.exp(b * t) + rng.normal(0, 0.05, t.size) for b in b_true])
    bank = FilterBank.from_model("a*exp(b*t)", "t", K,
                                 p0=[1.0, 0.4], window_size=40,
                                 q_diag=[1e-4, 1e-3], r=0.3)
    out = bank.run(t, Y, n_jobs=1)        # n_jobs>1 fans streams across threads
    print("\n== FilterBank: many streams at once ==")
    print("recovered b:", np.round(out["params"][:, 1], 3))
    print("true b     :", b_true)


def fused_detector(rng) -> None:
    # A change hitting EVERY stream (an amplitude collapse at t=20) is weak in any
    # one innovation but strong in the pooled chi2(K) statistic.
    K = 3
    t = np.linspace(0, 40, 600)
    amp = np.where(t < 20, 1.0, 0.5)
    phases = (0.0, 0.7, 1.4)
    Y = np.column_stack([amp * np.sin(1.2 * t + p) + rng.normal(0, 0.05, t.size)
                         for p in phases])
    bank = FilterBank.from_model("A*sin(1.2*t + p)", "t", K,
                                 p0=[1.0, 0.0], window_size=40)
    det = bank.fused_detector(alpha=1e-4)
    fired = [t[s] for s in range(t.size) if det.update(t[s], Y[s])]
    print("\n== FusedChiSquareDetector: shared-fault detection ==")
    print("flags raised:", det.n_flags_,
          " first flag at t =", round(fired[0], 1) if fired else None, "(fault at 20)")


def main() -> None:
    rng = np.random.default_rng(0)
    track_drifting_parameter(rng)
    preset(rng)
    filter_bank(rng)
    fused_detector(rng)


if __name__ == "__main__":
    main()
```

## Output (`python examples/05_streaming.py`)

```text
== EACFilter: track a mid-stream step ==
final b estimate: 0.55  (true 0.55)
drifts detected : 1

== EACFilter.tracking() preset ==
params: {'A': 1.998, 'w': 1.499}

== FilterBank: many streams at once ==
recovered b: [0.3   0.501 0.7   0.9  ]
true b     : [0.3 0.5 0.7 0.9]

== FusedChiSquareDetector: shared-fault detection ==
flags raised: 1  first flag at t = 20.0 (fault at 20)
```
