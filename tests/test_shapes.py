"""Tests of step 4 (shapes.py, pca.py) on fake data. Run from the repo folder: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the code is in the repo folder (not installed)
import pca  # noqa: E402
import shapes  # noqa: E402

SETTINGS = {"n_points": 50, "points": ["wrist"], "scale_by_arm_length": False, "shape_part": "whole"}


def test_stretch_keeps_ends_and_straight_lines():
    times = np.array([0.0, 0.1, 0.25, 0.3, 0.5])         # frames not evenly spaced
    values = np.column_stack([2 * times + 1, -times, 0.5 * times])   # a straight line at constant speed
    stretched = shapes.stretch(times, values, 20)
    assert np.allclose(stretched[0], values[0])
    assert np.allclose(stretched[-1], values[-1])
    target = np.linspace(0, 0.5, 20)
    assert np.allclose(stretched, np.column_stack([2 * target + 1, -target, 0.5 * target]))


def fake_inputs():
    """One trial of 30 frames and its row in the selected trials (arm length 0.5 m)."""
    t = np.linspace(0, 1, 30)
    movement_cut = pd.DataFrame({"trial_id": "fake", "time_s": t, "wrist_forward": 0.3 + 0.2 * t,
                                 "wrist_out": 0.1, "wrist_up": -0.4 + 0.5 * t})
    selected = pd.DataFrame([{"trial_id": "fake", "session_id": "s", "task_id": 1, "task_name": "task", "person": "P",
                              "condition": "unknown", "visit": 1, "side": "right",
                              "target": "90", "task_run": 1, "trial_index": 2, "facing_ok": True, "start_delay_s": 0.1,
                              "movement_duration_s": 1.0, "goal_time_s": 0.5, "arm_length_m": 0.5}])
    return movement_cut, selected


def test_scale_by_arm_length_divides_positions():
    movement_cut, selected = fake_inputs()
    in_metres = shapes.build_shape_table(movement_cut, selected, SETTINGS)
    in_arm_lengths = shapes.build_shape_table(movement_cut, selected, {**SETTINGS, "scale_by_arm_length": True})
    shape_columns = [c for c in in_metres.columns if pca.is_shape_column(c)]
    assert np.allclose(in_arm_lengths[shape_columns], in_metres[shape_columns] / 0.5)
    # start_delay_s and movement_duration_s are not changed
    assert in_arm_lengths["movement_duration_s"].iloc[0] == 1.0


def test_target_correction_gives_same_average_for_every_target():
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"target": ["B", "C", "D"] * 20, "side": "right"})
    offset = table["target"].map({"B": 0.0, "C": 0.2, "D": -0.3}).to_numpy()[:, None]   # each target: another place
    shapes_before = pd.DataFrame(rng.normal(0, 0.05, (60, 12)) + offset)
    corrected = pca.remove_target_average(shapes_before, table)
    averages = corrected.groupby(table["target"]).mean()
    assert np.allclose(averages.loc["B"], averages.loc["C"])
    assert np.allclose(averages.loc["B"], averages.loc["D"])
