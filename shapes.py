"""Step 4a: the shape table. Used by 04_shapes_pca.ipynb.

Chooses the trials, stretches each cut movement (step 3) to n_points points evenly spaced in time, and builds ONE ROW
PER TRIAL: info columns, start_delay_s, movement_duration_s, then the shape: <point>_<direction>_000 ... (e.g.
wrist_forward_000 ... wrist_up_199). start_delay_s and movement_duration_s are kept apart: they do not go into the PCA.
`settings` = the settings of the notebook, as a dict.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PARTICIPANTS_CSV, PROCESSED_ROOT, SESSION_PARTICIPANT_CSV
from preprocess import add_participant_id, read_hand_filled_csv

DIRECTIONS = ["forward", "out", "up"]
ID_KINDS = ["named", "no ID", "test"]   # only for INCLUDE_NO_ID, INCLUDE_TESTS and the counts table
# Columns of the shape table before the shape columns.
INFO_COLUMNS = ["trial_id", "session_id", "task_id", "task_name", "person", "condition", "visit", "side", "target",
                "task_run", "trial_index", "facing_ok", "start_delay_s", "movement_duration_s"]
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e5e4df"


def load_inputs(processed_root=PROCESSED_ROOT):
    """The tables of steps 1-3: movement_cut (positions only), cut_trials, trials, sessions, kinematics_trials."""
    positions = [f"{point}_{direction}" for point in ("wrist", "elbow") for direction in DIRECTIONS]
    movement_cut = pd.read_parquet(processed_root / "movement_cut.parquet", columns=["trial_id", "time_s"] + positions)
    cut_trials = pd.read_parquet(processed_root / "movement_cut_trials.parquet")
    trials = pd.read_parquet(processed_root / "trials.parquet")
    sessions = pd.read_parquet(processed_root / "sessions.parquet")
    kinematics_trials = pd.read_parquet(processed_root / "kinematics_trials.parquet")
    return movement_cut, cut_trials, trials, sessions, kinematics_trials


# ---- Who is who ----

def session_people(sessions, participants_csv=PARTICIPANTS_CSV, session_participant_csv=SESSION_PARTICIPANT_CSV):
    """Who each session is, read again from the 2 hand-filled CSV files (so a correction there counts without
    re-running step 1).
    - person = participant_id when it is filled in, otherwise "noid_<date>_<time>".
    - condition = from participants.csv: "healthy", the diagnosis (e.g. "Stroke"), or "unknown" if not known yet.
      Sessions without an ID are healthy (rule of the project: the patients are those of the clinical team).
    - visit = 1 (first assessment) or 2 (reassessment), from session_participant.csv.
    - id_kind = "test" (participant_id starting with "test"), "no ID" or "named": only for the switches and counts."""
    people = add_participant_id(sessions[["session_id", "n_trials"]], session_participant_csv)
    conditions = read_hand_filled_csv(participants_csv).set_index("participant_id")["condition"]
    has_id = people["participant_id"].notna()
    people["person"] = people["participant_id"].where(has_id, "noid_" + people["session_id"].str.removeprefix("upper_limb_"))
    people["condition"] = people["participant_id"].map(conditions).where(has_id, "healthy").fillna("unknown")
    is_test = people["participant_id"].str.lower().str.startswith("test", na=False)
    people["id_kind"] = np.where(is_test, "test", np.where(has_id, "named", "no ID"))
    return people


def trial_info(trials, people, kinematics_trials, cut_trials):
    """One row per trial of the usable sessions (step 1): who, which task and arm, the checks, whether steps 2 and 3
    kept it, start_delay_s and movement_duration_s (step 3), arm_length_m (step 2), and goal_time_s."""
    info = trials[["trial_id", "session_id", "task_id", "task_name", "side", "target", "task_run", "trial_index",
                   "is_first_repetition", "passes_checks"]].copy()
    # goal_time (step 1) is on the software clock; time_s = 0 at the start position (pose locked, or trial start).
    info["goal_time_s"] = trials["goal_time"] - trials["pose_locked_time"].fillna(trials["start_time"])
    info = info.merge(people[["session_id", "person", "condition", "visit", "id_kind"]], on="session_id")
    info = info.merge(kinematics_trials[["trial_id", "facing_ok", "arm_length_m"]], on="trial_id", how="left")
    info = info.merge(cut_trials[["trial_id", "start_delay_s", "movement_duration_s"]], on="trial_id", how="left")
    info["in_step2"] = info["trial_id"].isin(kinematics_trials["trial_id"])
    info["in_step3"] = info["trial_id"].isin(cut_trials["trial_id"])
    return info


# ---- Which trials ----

def select_trials(info, settings):
    """The trials that go into the shape table: the trials kept by step 3, then the 4 switches in this order.
    Returns (selected, stages): stages = {name of the switch: trials left after it}, for the counts table."""
    selected = info[info["in_step3"]]
    stages = {}
    if not settings["include_no_id"]:
        selected = selected[selected["id_kind"] != "no ID"]
    stages[f"step 4: INCLUDE_NO_ID = {settings['include_no_id']}"] = selected
    if not settings["include_tests"]:
        selected = selected[selected["id_kind"] != "test"]
    stages[f"step 4: INCLUDE_TESTS = {settings['include_tests']}"] = selected
    if settings["only_passing_trials"]:
        selected = selected[selected["passes_checks"]]
    stages[f"step 4: ONLY_PASSING_TRIALS = {settings['only_passing_trials']}"] = selected
    if settings["skip_first_repetition"]:
        selected = selected[~selected["is_first_repetition"]]
    stages[f"step 4: SKIP_FIRST_REPETITION = {settings['skip_first_repetition']}"] = selected
    return selected, stages


def selection_table(people, info, stages, shape_table):
    """How many sessions and trials are left after each stage (steps 1 to 4), split into named / no ID / test.
    A session counts at a stage if at least one of its trials is left."""
    # Step 1, all recorded sessions: their trials are counted from the events (finished trials).
    rows = {"step 1: all recorded sessions": (people["id_kind"].value_counts(),
                                              people.groupby("id_kind")["n_trials"].sum())}
    stage_tables = {"step 1: usable sessions": info,
                    "step 2: kinematics computed": info[info["in_step2"]],
                    "step 3: clear movement (cut)": info[info["in_step3"]],
                    **stages,
                    "step 4: in the shape table": info[info["trial_id"].isin(shape_table["trial_id"])]}
    for stage, left in stage_tables.items():
        rows[stage] = (left.drop_duplicates("session_id")["id_kind"].value_counts(), left["id_kind"].value_counts())

    table = pd.DataFrame({stage: pd.concat({"sessions": sessions, "trials": trials})
                          for stage, (sessions, trials) in rows.items()}).T
    table = table.reindex(columns=pd.MultiIndex.from_product([["sessions", "trials"], ID_KINDS])).fillna(0).astype(int)
    for what in ["sessions", "trials"]:
        table[(what, "total")] = table[what].sum(axis=1)
    return table[[(what, c) for what in ["sessions", "trials"] for c in ID_KINDS + ["total"]]]


def per_task_table(info, stages, shape_table):
    """Per task: trials after step 3, how many trials each switch left out, then trials and persons in the shape table."""
    by_task = ["task_id", "task_name"]
    previous = info[info["in_step3"]]
    table = {"after step 3": previous.groupby(by_task).size()}
    for stage, left in stages.items():
        switch = stage.removeprefix("step 4: ").split(" =")[0]
        table[f"left out by {switch}"] = previous.groupby(by_task).size().sub(left.groupby(by_task).size(), fill_value=0)
        previous = left
    table["left out: no frames left"] = previous.groupby(by_task).size().sub(shape_table.groupby(by_task).size(),
                                                                             fill_value=0)
    table["trials"] = shape_table.groupby(by_task).size()
    table["persons"] = shape_table.groupby(by_task)["person"].nunique()
    return pd.DataFrame(table).fillna(0).astype(int)


# ---- Stretching ----

def stretch(times, values, n_points):
    """Positions at n_points moments evenly spaced in time, from the first to the last frame (np.interp: each value
    comes from the 2 frames around it). The first and last positions are kept; a straight line stays straight.
    values: one row per frame, one column per coordinate."""
    target = np.linspace(times[0], times[-1], n_points)
    return np.column_stack([np.interp(target, times, values[:, j]) for j in range(values.shape[1])])


def position_columns(points):
    """e.g. ["wrist_forward", "wrist_out", "wrist_up"] for points = ["wrist"]."""
    return [f"{point}_{direction}" for point in points for direction in DIRECTIONS]


def frames_used(frames, goal_time_s, settings):
    """The frames of one trial that make its shape. Frames with a missing position are left out (long gaps are then
    bridged by the stretching, as in the old code). shape_part "way_out": only up to the goal (goal_time_s)."""
    frames = frames.dropna(subset=position_columns(settings["points"]))
    if settings["shape_part"] == "way_out":
        frames = frames[frames["time_s"] <= goal_time_s]   # no goal time (NaN): nothing left
    elif settings["shape_part"] != "whole":
        raise ValueError(f'shape_part must be "whole" or "way_out", not {settings["shape_part"]!r}')
    return frames


def trial_shape(frames, goal_time_s, arm_length_m, settings):
    """The shape of one trial as one flat row: point by point, direction by direction, n_points values each
    (wrist_forward_000 ... wrist_up_199, then elbow_... if asked). None if fewer than 2 frames are left."""
    frames = frames_used(frames, goal_time_s, settings)
    if len(frames) < 2:
        return None
    values = frames[position_columns(settings["points"])].to_numpy()
    if settings["scale_by_arm_length"]:
        values = values / arm_length_m   # unit: arm lengths, so tall and small people can be compared
    return stretch(frames["time_s"].to_numpy(), values, settings["n_points"]).T.ravel()


def build_shape_table(movement_cut, selected, settings):
    """One row per selected trial: INFO_COLUMNS, then the shape columns. Trials with fewer than 2 frames left
    (e.g. no goal time with shape_part "way_out") are left out."""
    selected = selected.set_index("trial_id")
    kept, shapes = [], []
    for trial_id, frames in movement_cut[movement_cut["trial_id"].isin(selected.index)].groupby("trial_id", sort=False):
        trial = selected.loc[trial_id]
        shape = trial_shape(frames, trial["goal_time_s"], trial["arm_length_m"], settings)
        if shape is not None:
            kept.append(trial_id)
            shapes.append(shape)
    names = [f"{column}_{i:03d}" for column in position_columns(settings["points"]) for i in range(settings["n_points"])]
    info = selected.loc[kept].reset_index()[INFO_COLUMNS]
    return pd.concat([info, pd.DataFrame(np.vstack(shapes), columns=names)], axis=1)


# ---- Check and save ----

def plot_stretch_check(movement_cut, shape_table, info, trial_id, settings):
    """Wrist height of one trial: the cut movement (line) vs its stretched points (dots). The dots must sit on the line."""
    trial = info.set_index("trial_id").loc[trial_id]
    frames = frames_used(movement_cut[movement_cut["trial_id"] == trial_id], trial["goal_time_s"], settings)
    t = frames["time_s"].to_numpy()
    original = frames["wrist_up"].to_numpy()
    if settings["scale_by_arm_length"]:
        original = original / trial["arm_length_m"]
    n_points = settings["n_points"]
    stretched = shape_table.set_index("trial_id").loc[trial_id, [f"wrist_up_{i:03d}" for i in range(n_points)]]
    unit = "arm lengths" if settings["scale_by_arm_length"] else "m"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot((t - t[0]) / (t[-1] - t[0]) * 100, original, color=BLUE, linewidth=2, label=f"cut movement ({len(t)} frames)")
    ax.scatter(np.linspace(0, 100, n_points), stretched, color=ORANGE, s=10, zorder=3, label=f"stretched ({n_points} points)")
    ax.set_xlabel("progress of the movement (%)")
    ax.set_ylabel(f"wrist height from the shoulder ({unit})")
    ax.set_title(f"{trial_id}, shape_part = {settings['shape_part']}", fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.show()


def save_shape_table(shape_table, settings, processed_root=PROCESSED_ROOT):
    """Writes shape_tables.parquet and the settings used (shapes_settings.json)."""
    shape_table.to_parquet(processed_root / "shape_tables.parquet", index=False)
    with open(processed_root / "shapes_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    for name in ["shape_tables.parquet", "shapes_settings.json"]:
        print(f"{name:<24} {(processed_root / name).stat().st_size / 1e6:7.1f} MB")
