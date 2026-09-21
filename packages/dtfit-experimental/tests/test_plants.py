"""The embedded plant simulators, estimator adapters and MCU footprint,
shared by the filter notebooks."""

import numpy as np
import pytest

from dtfit_experimental.study import plants
from dtfit_experimental.study.plants import (
    PLANTS, EAAd, EKFAd, LegAd, gen_plant, perr, drive,
    MISMATCH_PAIRS, mismatch_scores,
    make_multi, MergedTracker, run_tracker, kalman_multi,
)
from dtfit_experimental.study.cost import MCUS, footprint_rows


def test_gen_plant_outliers_replace_only_the_noisy_signal():
    """About a tenth of the samples are replaced by a gross spike, and the
    returned clean signal is untouched; the mutation this catches is
    contaminating `clean` instead of `y`, which would move the baseline the
    plant is scored against off the plant's true response."""
    plant = PLANTS[0]
    rng = np.random.default_rng(0)
    t = np.linspace(0, plant["T"], plant["n"])
    true_clean = plant["func"](t, *[plant["true"][k] for k in sorted(plant["true"])])
    _, y, clean = gen_plant(plant, rng, noise=0.0, outliers=0.1)
    assert np.allclose(clean, true_clean)
    frac_hit = float(np.mean(np.abs(y - clean) > 1e-9))
    assert 0.05 < frac_hit < 0.15


def test_block_adapter_forwards_adaptive_window():
    """`adaptive_window=False` must reach `ImageFilter`, keeping the window
    at `window_size` from the first measurement; the mutation this catches
    is the adapter dropping the keyword on the way through, which would
    make a fixed-window and an adaptive-window run indistinguishable."""
    plant = PLANTS[0]
    ad = EAAd(plant, adaptive_window=False)
    assert ad.f.adaptive_window is False
    assert ad.f.W == plant["window"]


def test_drive_scores_past_warmup_only():
    """The RMSE is computed only from `warm` onward; the mutation this
    catches is dropping the warm-up slice, which would let the estimator's
    opening transient (far from the truth) leak into the score."""
    plant = PLANTS[2]  # first_order: a large early transient off p0
    rng = np.random.default_rng(0)
    t, y, clean = gen_plant(plant, rng, noise=0.01)
    warm = plant["window"] + 15
    rmse, _, track = drive(LegAd(plant), t, y, clean, warm)
    manual = float(np.sqrt(np.mean((track[warm:] - clean[warm:]) ** 2)))
    assert rmse == pytest.approx(manual, rel=1e-9)
    full = float(np.sqrt(np.nanmean((track - clean) ** 2)))
    assert not np.isclose(rmse, full)


def test_perr_is_percent_over_every_truth_key():
    """The mean relative error is taken over every key of `true`; the
    mutation this catches is a truth key lost from the mean, which would
    understate the error whenever the dropped parameter is the badly
    recovered one."""
    true = {"A": 2.0, "w": 2.5, "z": 0.12}
    params = {"A": 2.0, "w": 2.5, "z": 0.06}  # z off by 50%, the rest exact
    e = perr(params, true)
    assert e == pytest.approx(100.0 * 0.5 / 3)


def test_footprint_rows_mcu_fit_table():
    """A 3-axis block-filter tracker at W=60 fits the Cortex-M4F's SRAM and
    only just fits the ATmega328's; the mutation this catches is inverting
    the SRAM comparison, which would swap which MCUs the table calls a fit."""
    lat = {"dtfit block filter": 1.0, "dtfit Legendre filter": 1.0,
           "EKF (params-as-state)": 1.0, "RLS (AR predictor)": 1.0}
    rows = footprint_rows(lat, n=3, W=60)
    by_mcu = {r["MCU"]: r["fits"] for r in rows["mcu"]}
    assert by_mcu["ARM Cortex-M4F (STM32F4)"] == "yes"
    assert by_mcu["AVR ATmega328 (Uno)"] == "tight"


def test_mcus_is_the_five_field_form():
    """`MCUS` carries name, SRAM bytes, clock MHz, FPU flag and MFLOP/s for
    five parts including the rig's nRF52840; the mutation this catches is a
    row losing a field during a widen, which unpacks silently into the wrong
    column instead of raising."""
    assert len(MCUS) == 5
    assert MCUS[-1][0] == "nRF52840"
    for name, sram_bytes, clock_mhz, fpu, mflops in MCUS:
        assert isinstance(name, str)
        assert sram_bytes > 0
        assert clock_mhz > 0
        assert isinstance(fpu, bool)
        assert mflops > 0


def test_footprint_rows_mcu_table_carries_clock_and_mflops():
    """`footprint_rows`'s MCU table exposes the datasheet fields `MCUS` now
    carries; the mutation this catches is `footprint_rows` unpacking `MCUS`
    positionally without naming the new fields, which would drop them from
    the row instead of failing."""
    lat = {"dtfit block filter": 1.0, "dtfit Legendre filter": 1.0,
           "EKF (params-as-state)": 1.0, "RLS (AR predictor)": 1.0}
    rows = footprint_rows(lat, n=3, W=60)
    by_mcu = {r["MCU"]: r for r in rows["mcu"]}
    nrf = by_mcu["nRF52840"]
    assert nrf["clock_MHz"] == 64
    assert nrf["MFLOPs"] == 10.0
    assert nrf["FPU"] == "yes"
    assert by_mcu["AVR ATmega328 (Uno)"]["FPU"] == "no (soft)"


def test_make_multi_shapes_and_finite():
    """`make_multi` returns a stream and a fault split matched in shape and
    fully finite; the mutation this catches is dropping the `// 2` from
    `half`, which would move the fault to the end of the run instead of the
    midpoint."""
    rng = np.random.default_rng(0)
    t, noisy, clean, half = make_multi(rng, n=200)
    assert t.shape == (200,)
    assert noisy.shape == clean.shape == (200, 3)
    assert half == 100
    assert np.all(np.isfinite(noisy))
    assert np.all(np.isfinite(clean))


def test_make_multi_fault_scales_the_damping_jump():
    """`fault=0.0` leaves the second half on the first half's damping and
    `fault=1.0` is the default stream; the mutation this catches is `fault`
    scaling the final damping instead of the jump, which would undamp the
    second half at `fault=0.0`."""
    _t, _noisy, clean0, half = make_multi(np.random.default_rng(0), n=200,
                                          fault=0.0)
    _t, _noisy, clean1, _half = make_multi(np.random.default_rng(0), n=200)
    _t, _noisy, full, _half = make_multi(np.random.default_rng(0), n=200,
                                         fault=1.0)
    assert np.array_equal(clean1, full)
    assert np.array_equal(clean0[:half], clean1[:half])
    # undamped, the second half would swing at the full amplitude; on the
    # first half's damping it has decayed well below the first half's peak
    assert (np.abs(clean0[half:]).max(axis=0)
            < 0.5 * np.abs(clean0[:half]).max(axis=0)).all()
    assert np.abs(clean0[half:]).max() > np.abs(clean1[half:]).max()


def test_merged_tracker_step_predict_shape():
    """One `step` then `predict` returns one finite value per axis; the
    mutation this catches is swapping the detector's `update(t, y)`
    argument order, which a `FilterBank` signature change would also
    trigger."""
    tr = MergedTracker(3, [2.0, 2.5, 0.1])
    tr.step(0, 0.0, np.array([2.0, 1.5, 2.5]))
    pred = tr.predict(0.0)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


def test_run_tracker_no_false_alarms_before_the_fault():
    """Over a fault-free first half the fused detector raises no flags; the
    mutation this catches is counting flags at or after `half` as false
    alarms instead of the ones before it, which would call a correct late
    detection a false alarm."""
    rng = np.random.default_rng(0)
    t, Y, clean, half = make_multi(rng, n=900)
    tr, pred, rmse, fp, lat, warm = run_tracker(t, Y, clean, half, inflate=4.0)
    assert pred.shape == Y.shape
    assert np.isfinite(rmse)
    assert fp == 0


def test_kalman_multi_excludes_warmup_from_rmse():
    """The RMSE is computed only from `warm` onward, matching a manual
    recomputation; the mutation this catches is dropping the warm-up mask,
    which would let the filter's initial transient inflate the score."""
    rng = np.random.default_rng(2)
    t, Y, clean, half = make_multi(rng, n=200)
    warm = 30
    pred, rmse = kalman_multi(t, Y, clean, warm)
    valid = np.all(np.isfinite(pred), axis=1)
    manual = float(np.sqrt(np.mean((pred[valid][warm:] - clean[valid][warm:]) ** 2)))
    assert rmse == pytest.approx(manual, rel=1e-9)
    naive = float(np.sqrt(np.mean((pred[valid] - clean[valid]) ** 2)))
    assert not np.isclose(rmse, naive)


def test_mismatch_scores_correct_model_beats_wrong_model():
    """For every entry of `MISMATCH_PAIRS`, an EKF fitted to the plant that
    actually generated the stream scores a lower clean-RMSE than the same
    EKF fitted to the paired wrong plant; the mutation this catches is a
    copy-paste in the pair table that fits a plant against itself, which
    collapses the two scores to the same value instead of a clear gap."""
    by_key = {p["key"]: p for p in PLANTS}
    rng = np.random.default_rng(3)
    for true_key, wrong_key in MISMATCH_PAIRS:
        true_plant = by_key[true_key]
        t, y, clean = gen_plant(true_plant, rng, noise=0.05)
        right_rmse, _, _ = mismatch_scores(EKFAd(true_plant), t, y, clean, warm=20)
        wrong_rmse, _, _ = mismatch_scores(EKFAd(by_key[wrong_key]), t, y, clean, warm=20)
        assert right_rmse < wrong_rmse


def test_mismatch_control_names_are_public():
    """The mismatch control is reachable under the names a caller imports:
    `MISMATCH_PAIRS`, `MISMATCH_CEILING` and `mismatch_scores`, each listed
    in `__all__` and bound on the module; the mutation this catches is
    spelling one of them with a leading underscore, which leaves `__all__`
    advertising a name the module does not bind."""
    for name in ("MISMATCH_PAIRS", "MISMATCH_CEILING", "mismatch_scores"):
        assert name in plants.__all__
        assert hasattr(plants, name)
