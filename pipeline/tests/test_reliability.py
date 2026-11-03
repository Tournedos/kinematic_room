"""Tests of step 5 (reliability.py) on fake data. Run from the pipeline folder: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the code is one folder up (not installed)
import reliability  # noqa: E402

SETTINGS = {"n_splits": 100, "seed": 0, "correlation": "pearson", "spearman_brown": True}


def fake_trials(values, targets=None, task_id=1):
    """A trials table from values of shape (n_measures, n_units, n_reps): measure k of unit u, repetition r.
    targets: one label per repetition (same for every unit), or None = one target."""
    n_measures, n_units, n_reps = values.shape
    rows = []
    for u in range(n_units):
        for r in range(n_reps):
            row = {"trial_id": f"s{u:03d}_r{r:03d}", "session_id": f"s{u:03d}", "task_id": task_id, "task_name": "task",
                   "person": f"p{u}", "condition": "healthy", "visit": pd.NA, "side": "right",
                   "target": None if targets is None else targets[r], "trial_index": r + 2}
            row.update({f"PC{k + 1}": values[k, u, r] for k in range(n_measures)})
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_of(trials, n_measures, min_reps=4, settings=SETTINGS):
    """The whole step 5 on a fake trials table: units (one per session and arm), then the reliability table."""
    trials = reliability.add_units(trials, "separate")
    units = reliability.units_table(trials, min_reps)
    return reliability.split_half_reliability(trials, units, [f"PC{k + 1}" for k in range(n_measures)], settings)


def test_stable_number_is_reliable():
    rng = np.random.default_rng(1)
    per_unit = rng.normal(0, 1, (1, 30, 1))                      # a fixed value per unit
    values = per_unit + rng.normal(0, 0.05, (1, 30, 10))         # + small noise from one repetition to the next
    assert reliability_of(fake_trials(values), 1)["reliability"].iloc[0] > 0.95


def test_pure_noise_is_not_reliable():
    # Many units: with few units, chance alone can give +-0.4 (Spearman-Brown stretches small correlations).
    for seed in range(3):
        values = np.random.default_rng(seed).normal(0, 1, (1, 400, 10))
        assert abs(reliability_of(fake_trials(values), 1)["reliability"].iloc[0]) < 0.2


def test_both_halves_have_the_same_targets():
    targets = ["B", "B", "B", "C", "C", "D", "D", "D", "D"]     # 3 targets, odd numbers of trials
    trials = reliability.add_units(fake_trials(np.zeros((1, 5, 9)), targets, task_id=6), "separate")
    for seed in range(20):
        labels = reliability.half_labels(trials, seed)
        for _, unit in trials.assign(half=labels).groupby("unit"):
            in_a = unit.loc[unit["half"] == "A", "target"].value_counts().sort_index()
            in_b = unit.loc[unit["half"] == "B", "target"].value_counts().sort_index()
            assert in_a.to_dict() == in_b.to_dict() == {"B": 1, "C": 1, "D": 2}


def test_units_with_too_few_trials_are_left_out():
    values = np.random.default_rng(3).normal(0, 1, (1, 6, 6))
    trials = fake_trials(values)
    trials = trials[~((trials["session_id"] == "s000") & (trials["trial_index"] > 4))]   # s000: only 3 trials
    trials = reliability.add_units(trials, "separate")
    units = reliability.units_table(trials, min_reps=4)
    assert units.set_index("unit").loc["s000 right", "kept"] == False  # noqa: E712
    assert units["kept"].sum() == 5
    table = reliability.split_half_reliability(trials, units, ["PC1"], SETTINGS)
    assert table["n_units"].iloc[0] == 5


# ---- Same result as the NX-414 function (copied unchanged from nx414_LOUIS.ipynb, as the reference) ----

def compute_ceiling_splithalf(responses, folds=10, seed=0, spearman_brown=True, equalize_halves=True, clip_folds=False):
    x = np.asarray(responses, dtype=np.float64)
    unit_shape = x.shape[:-2]
    n_stim = x.shape[-2]
    n_reps = x.shape[-1]
    x_flat = x.reshape(-1, n_stim, n_reps)
    out = []
    for fold in range(folds):
        rng = np.random.default_rng(seed + fold)
        perm = rng.permutation(n_reps)
        if equalize_halves:
            k = n_reps // 2
            idx1 = perm[:k]
            idx2 = perm[k:2*k]
        else:
            k = n_reps // 2
            idx1 = perm[:k]
            idx2 = perm[k:]
        a = np.nanmean(x_flat[:, :, idx1], axis=-1)
        b = np.nanmean(x_flat[:, :, idx2], axis=-1)
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
        num = np.nansum(a * b, axis=1)
        den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
        r = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        if spearman_brown:
            r = (2 * r) / (1 + r)
        if clip_folds:
            r = np.clip(r, 0.0, 1.0)
        out.append(r)
    rel = np.nanmean(np.stack(out, axis=0), axis=0)
    rel = rel.reshape(unit_shape)
    rel = 100.0 * rel
    return rel


def test_same_as_the_old_eeg_function():
    rng = np.random.default_rng(4)
    # 3 numbers, 25 units, 9 repetitions (odd: one is dropped in each split), with some signal per unit
    values = rng.normal(0, 1, (3, 25, 1)) * np.array([2.0, 0.5, 0.0])[:, None, None] + rng.normal(0, 1, (3, 25, 9))
    old = compute_ceiling_splithalf(values, folds=50, seed=7) / 100   # old: in percent
    new = reliability_of(fake_trials(values), 3, settings={**SETTINGS, "n_splits": 50, "seed": 7})
    assert np.allclose(new["reliability"].to_numpy(), old, atol=1e-12)
