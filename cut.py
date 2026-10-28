"""Step 3: cut each trial to the movement. Used by 03_cut_movement.ipynb.

Each trial of step 2 still contains waiting before and after the movement. Here we find when the wrist really starts
and stops moving, from its speed, and keep only that part. Trials with no clear movement are removed.
The same rule for every task. Input: kinematics.parquet (step 2, not changed).
`settings` = the settings of the notebook, as a dict.
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_ROOT

WRIST = ["wrist_forward", "wrist_out", "wrist_up"]


def load_kinematics(processed_root=PROCESSED_ROOT):
    """The table of step 2: one row per moment of each trial."""
    return pd.read_parquet(processed_root / "kinematics.parquet")


def wrist_speed(trial):
    """Wrist speed (m/s) at each moment of one trial: distance moved per second, from the filtered wrist position.
    np.gradient uses the moments before and after, so a missing frame also makes its neighbours' speed missing."""
    if len(trial) < 2:
        return np.full(len(trial), np.nan)
    velocity = np.gradient(trial[WRIST].to_numpy(), trial["time_s"].to_numpy(), axis=0)
    return np.linalg.norm(velocity, axis=1)


def first_stretch_above(above, min_frames):
    """Position of the first frame of the first stretch of min_frames frames in a row that are all above the
    threshold (True), or None if there is no such stretch."""
    if len(above) < min_frames:
        return None
    # For each frame: are this frame and the next min_frames - 1 frames all above?
    full = np.convolve(above.astype(int), np.ones(min_frames, dtype=int), mode="valid") == min_frames
    return int(np.argmax(full)) if full.any() else None


def movement_bounds(speed, settings):
    """Where the movement starts and ends in one trial (row positions), or why it has no clear movement.
    Start = first frame where the speed is above speed_threshold x peak speed and stays above for min_frames_above
    frames in a row (so a single jump of the camera is not a start). End = the same rule read backwards from the end.
    Missing speed counts as below the threshold.
    Returns (start, end, reason): reason is None when the trial is kept."""
    if np.isnan(speed).all():
        return None, None, "no usable wrist data"
    peak = np.nanmax(speed)
    # The threshold is relative to the trial's own peak, so camera jitter alone would always pass it:
    # a trial whose peak is below min_peak_speed has no real movement.
    if peak < settings["min_peak_speed"]:
        return None, None, "peak speed too low"
    above = speed > settings["speed_threshold"] * peak   # missing speed (NaN) gives False = below
    start = first_stretch_above(above, settings["min_frames_above"])
    if start is None:
        return None, None, "no stretch above the threshold"
    end = len(above) - 1 - first_stretch_above(above[::-1], settings["min_frames_above"])
    if end <= start:
        return None, None, "end not after start"
    return start, end, None


def cut_trial(trial, start, end, margin_frames):
    """Keeps only the frames from start - margin_frames to end + margin_frames of one trial,
    and adds time_from_start_s (0 = movement start)."""
    first, last = max(start - margin_frames, 0), min(end + margin_frames, len(trial) - 1)
    piece = trial.iloc[first:last + 1].copy()
    piece["time_from_start_s"] = piece["time_s"] - trial["time_s"].iloc[start]
    return piece


def cut_movements(kinematics, settings):
    """Cuts every trial to its movement. Returns 3 tables:
    - movement_cut: the frames of the movement (same columns as kinematics + time_from_start_s), kept trials only
    - cut_trials: one row per kept trial: start_s, end_s, start_delay_s, movement_duration_s
    - removed: one row per removed trial, with the reason."""
    pieces, kept, removed = [], [], []
    for trial_id, trial in kinematics.groupby("trial_id", sort=False):
        trial = trial.reset_index(drop=True)
        start, end, reason = movement_bounds(wrist_speed(trial), settings)
        if reason is not None:
            removed.append({"trial_id": trial_id, "reason": reason})
            continue
        pieces.append(cut_trial(trial, start, end, settings["margin_frames"]))
        start_s, end_s = trial["time_s"].iloc[start], trial["time_s"].iloc[end]
        kept.append({"trial_id": trial_id, "start_s": start_s, "end_s": end_s,
                     # time_s = 0 when the person is in the start position, so the start time is also the delay
                     # before moving (like a reaction time)
                     "start_delay_s": start_s, "movement_duration_s": end_s - start_s})
    # (empty tables, with the right columns, when every trial is removed)
    movement_cut = pd.concat(pieces, ignore_index=True) if pieces else kinematics.iloc[:0].assign(time_from_start_s=[])
    cut_trials = pd.DataFrame(kept, columns=["trial_id", "start_s", "end_s", "start_delay_s", "movement_duration_s"])
    return movement_cut, cut_trials, pd.DataFrame(removed, columns=["trial_id", "reason"])


def save_cut(movement_cut, cut_trials, settings, processed_root=PROCESSED_ROOT):
    """Writes movement_cut.parquet, movement_cut_trials.parquet and the settings used (cut_settings.json)."""
    movement_cut.to_parquet(processed_root / "movement_cut.parquet", index=False)
    cut_trials.to_parquet(processed_root / "movement_cut_trials.parquet", index=False)
    with open(processed_root / "cut_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    for name in ["movement_cut.parquet", "movement_cut_trials.parquet", "cut_settings.json"]:
        print(f"{name:<30} {(processed_root / name).stat().st_size / 1e6:7.1f} MB")
