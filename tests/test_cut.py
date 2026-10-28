"""Tests of step 3 (cut.py) on fake data. Run from the repo folder: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the code is in the repo folder (not installed)
import cut  # noqa: E402

SETTINGS = {"speed_threshold": 0.10, "min_frames_above": 3, "margin_frames": 0, "min_peak_speed": 0.15}
RATE = 30   # frames per second


def fake_trial(wrist_up):
    """One trial where the wrist only moves up and down, RATE frames per second."""
    t = np.arange(len(wrist_up)) / RATE
    return pd.DataFrame({"trial_id": "fake", "time_s": t, "wrist_forward": 0.3, "wrist_out": 0.0, "wrist_up": wrist_up})


def still_move_still():
    """1 s still, then a smooth 1 s movement up by 40 cm (starts at 1.0 s, ends at ~2.0 s), then 1 s still."""
    move = -0.3 + 0.4 * (1 - np.cos(np.linspace(0, np.pi, RATE))) / 2
    return fake_trial(np.r_[np.full(RATE, -0.3), move, np.full(RATE, 0.1)])


def test_start_and_end_found():
    _, cut_trials, removed = cut.cut_movements(still_move_still(), SETTINGS)
    assert removed.empty
    assert abs(cut_trials["start_s"].iloc[0] - 1.0) < 0.1
    assert abs(cut_trials["end_s"].iloc[0] - 2.0) < 0.1


def test_jitter_only_is_removed():
    jitter = -0.3 + np.random.default_rng(0).normal(0, 0.0005, 3 * RATE)   # 0.5 mm of camera noise, no movement
    _, cut_trials, removed = cut.cut_movements(fake_trial(jitter), SETTINGS)
    assert cut_trials.empty
    assert removed["trial_id"].tolist() == ["fake"]


def test_single_frame_spike_is_not_a_start():
    speed = np.zeros(60)
    speed[5] = 1.0          # one frame above the threshold: a jump of the camera
    speed[20:40] = 0.8      # the real movement
    start, end, reason = cut.movement_bounds(speed, SETTINGS)
    assert reason is None
    assert (start, end) == (20, 39)


def test_margin_keeps_extra_frames():
    exact, _, _ = cut.cut_movements(still_move_still(), SETTINGS)
    with_margin, _, _ = cut.cut_movements(still_move_still(), {**SETTINGS, "margin_frames": 2})
    assert len(with_margin) == len(exact) + 4   # 2 frames before the start, 2 after the end
    assert abs(with_margin["time_from_start_s"].iloc[0] + 2 / RATE) < 1e-9   # first frame: 2 frames before the start
