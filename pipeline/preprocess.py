"""Step 1: raw recordings -> 3 clean tables: sessions, trials, frames.

Preprocessing only organises and flags the data: no smoothing, no filtering, no change of coordinates.
Nothing is removed from a table: doubtful sessions and trials are only flagged (is_usable, passes_checks).

One session = 3 raw files in RAW_ROOT with the same name, upper_limb_<date>_<time>:
    .events.jsonl   what happened and when (one JSON event per line)
    .frames.csv     the body points, one row per frame
    .meta.json      the settings of the session (missing for a few sessions)
The raw files are only read, never modified.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from config import (MAX_DISTANCE_MAIN_PERSON_M, MAX_HEAD_HEIGHT_M, MAX_WRONG_PERSON_PCT, MIN_FRAMES_PER_TRIAL,
                    MIN_TRIALS_ONE_TASK, PARTICIPANTS_CSV, PROCESSED_ROOT, RAW_ROOT, SESSION_PARTICIPANT_CSV,
                    SETUP_KEYS)

# Events that mean "the person did what the task asked" (the goal differs per task).
GOAL_EVENTS = {"target_reached", "circle_completed", "cup_reached", "mouth_reached", "nose_reached"}

# The 14 body points kept, in the order of the raw file. "neck" is left out: it is always empty.
KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "body_center",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
             "left_hip", "right_hip"]
KEYPOINT_COLUMNS = [f"{point}_{value}" for point in KEYPOINTS for value in ("x", "y", "z", "conf")]

# Columns of participants.csv (filled in by hand). condition: "healthy", the diagnosis (e.g. "Stroke"), or "unknown".
# affected_side: the more affected arm, "left", "right", or empty = unknown. lesion_side (stroke): the brain hemisphere
# of the lesion, "left", "right", or empty = unknown. first_visit / second_visit: date and hour of the visits given by
# the clinical team, for reference only.
PARTICIPANT_COLUMNS = ["participant_id", "condition", "affected_side", "lesion_side", "first_visit", "second_visit",
                       "notes"]


# ---- Reading the raw files ----

def list_sessions(raw_root=RAW_ROOT):
    """All session IDs (e.g. "upper_limb_20260629_095022"), sorted by date.
    A session = an events file. Calibration recordings (r1_...) do not start with "upper_limb_", so they are left out."""
    return sorted(p.name.removesuffix(".events.jsonl") for p in raw_root.glob("upper_limb_*.events.jsonl"))


def read_events(session_id, raw_root=RAW_ROOT):
    """All events of one session, in file order (one dict per event). Blank lines are skipped."""
    with open(raw_root / f"{session_id}.events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_frames(session_id, columns, dtype=None, raw_root=RAW_ROOT):
    """The frames of one session, only the chosen columns (much faster than reading the whole file)."""
    return pd.read_csv(raw_root / f"{session_id}.frames.csv", usecols=columns, dtype=dtype)


def read_meta(session_id, raw_root=RAW_ROOT):
    """The settings of one session (a dict), or None if the session has no meta.json file."""
    path = raw_root / f"{session_id}.meta.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- Sessions ----

def build_sessions(raw_root=RAW_ROOT):
    """One row per session: when, typed ID, how much data, setup_version and is_usable. Every session is kept."""
    rows = []
    for session_id in list_sessions(raw_root):
        events = read_events(session_id, raw_root)
        meta = read_meta(session_id, raw_root)
        # Only 2 columns of the frames file are loaded (fast): enough to count the frames and to see if the person sat.
        frames = read_frames(session_id, ["nose_y", "trial_active"], raw_root=raw_root)
        # Head height (median of the nose during the repetitions, m): ~1.2 m seated, ~1.5-1.9 m standing.
        head_height = frames.loc[frames["trial_active"] == 1, "nose_y"].median()

        # Finished trials, and the tasks and arms really done (trial_end has the arm actually used).
        trial_ends = [e for e in events if e["type"] == "trial_end"]
        # Most repetitions of one task with one arm (e.g. 12): a real session has a few repetitions of at least one task.
        repetitions = pd.Series([(e["task_id"], e["side"]) for e in trial_ends]).value_counts()
        # When the first task started / the last task ended.
        task_times = [e["wall_time"] for e in events if e["type"] in ("task_start", "task_end")]
        # Date and start time come from the file name: upper_limb_<YYYYMMDD>_<HHMMSS>.
        date, time = session_id.removeprefix("upper_limb_").split("_")

        rows.append({
            "session_id": session_id,
            "date": pd.to_datetime(date).date(),
            "start_time": f"{time[:2]}:{time[2:4]}",
            "typed_id": typed_id(events),
            "n_frames": len(frames),
            "n_trials": len(trial_ends),
            "tasks": ", ".join(map(str, sorted({e["task_id"] for e in trial_ends}))),   # e.g. "1, 2, 7, 8"
            "arms": ", ".join(sorted({e["side"] for e in trial_ends})),                 # e.g. "left, right"
            # Time really spent on tasks (first task_start to last task_end), in minutes.
            "task_time_min": round((max(task_times) - min(task_times)) / 60, 1) if task_times else 0.0,
            "head_height_m": round(head_height, 2),
            "most_trials_one_task": int(repetitions.max()) if len(repetitions) else 0,
            "has_meta": meta is not None,
            "setup_version": setup_version(meta),
        })
    sessions = pd.DataFrame(rows)
    sessions["is_usable"] = is_usable(sessions)
    return sessions


def typed_id(events):
    """The ID typed by the operator during the session ("patient_id_set" events). If several were typed, the last one
    (the final correction). None if no ID was typed."""
    ids = [e["patient_id"] for e in events if e["type"] == "patient_id_set" and e.get("patient_id")]
    return ids[-1] if ids else None


def setup_version(meta):
    """Short fingerprint of the setup settings of a session (SETUP_KEYS in config.py): same settings = same fingerprint.
    A setting missing from meta.json counts as a value of its own (None). Returns None if there is no meta.json."""
    if meta is None:
        return None
    settings = {key: meta.get(key) for key in SETUP_KEYS}
    text = json.dumps(settings, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def is_usable(sessions, min_trials_one_task=MIN_TRIALS_ONE_TASK, max_head_height_m=MAX_HEAD_HEIGHT_M):
    """True for the sessions that go on to the trials and frames tables:
    - the session has frames and a meta.json file
    - at least min_trials_one_task repetitions of one task with one arm (quick recording tests have 1-5)
    - the person sat: head not higher than max_head_height_m (higher = staff standing to test the setup)."""
    return ((sessions["n_frames"] > 0)
            & sessions["has_meta"]
            & (sessions["most_trials_one_task"] >= min_trials_one_task)
            # written this way so that an unknown head height (NaN) does not count as standing
            & ~(sessions["head_height_m"] > max_head_height_m))


def setup_versions_table(sessions, raw_root=RAW_ROOT):
    """One row per setup_version, in order of first use: first and last date, number of sessions (all and usable),
    tasks recorded, and the settings that differ between versions (settings that never changed are not shown)."""
    sessions = sessions.dropna(subset=["setup_version"])
    by_version = sessions.groupby("setup_version", sort=False)
    table = by_version.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_sessions=("session_id", "size"),
        n_usable=("is_usable", "sum"),
        tasks=("tasks", lambda tasks: ", ".join(sorted({t for s in tasks for t in s.split(", ") if t}, key=int))),
    )
    # The settings of each version, read from its first session.
    settings = pd.DataFrame({version: {key: read_meta(session_id, raw_root).get(key) for key in SETUP_KEYS}
                             for version, session_id in by_version["session_id"].first().items()}).T
    changed = [key for key in SETUP_KEYS if settings[key].astype(str).nunique() > 1]
    return table.join(settings[changed])


# ---- Participants (2 CSV files filled in by hand) ----

def read_hand_filled_csv(path):
    """Reads one of the hand-filled CSV files. Everything is read as text (keeps IDs like "004"),
    and only empty cells count as missing (an ID like "NA" stays "NA")."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def make_participant_templates(sessions, session_participant_csv=SESSION_PARTICIPANT_CSV,
                               participants_csv=PARTICIPANTS_CSV):
    """Writes the 2 CSV files to fill in by hand, ONLY if they do not exist yet. An existing file is never touched.
    - session_participant.csv: one row per session; participant_id prefilled with the typed ID (empty if none),
      visit (1 = first assessment, 2 = reassessment) empty.
    - participants.csv: one row per participant_id of session_participant.csv; everything else empty."""
    session_participant_csv.parent.mkdir(parents=True, exist_ok=True)

    if session_participant_csv.exists():
        print(f"{session_participant_csv.name}: already exists, not changed")
    else:
        template = sessions[["session_id", "date", "typed_id"]].copy()
        template["participant_id"] = template["typed_id"]
        template["visit"] = None
        template["notes"] = None
        template.to_csv(session_participant_csv, index=False)
        print(f"{session_participant_csv.name}: template written ({len(template)} sessions)")

    if participants_csv.exists():
        print(f"{participants_csv.name}: already exists, not changed")
    else:
        ids = read_hand_filled_csv(session_participant_csv)["participant_id"].dropna().unique()
        template = pd.DataFrame(columns=PARTICIPANT_COLUMNS)
        template["participant_id"] = sorted(ids)
        template.to_csv(participants_csv, index=False)
        print(f"{participants_csv.name}: template written ({len(template)} participants)")


def add_participant_id(sessions, session_participant_csv=SESSION_PARTICIPANT_CSV):
    """Adds participant_id and visit to the sessions table, taken from session_participant.csv (empty where not
    filled in yet). Prints a warning listing the sessions that are not in the file (e.g. new recordings: add them by hand)."""
    table = read_hand_filled_csv(session_participant_csv)
    assert table["session_id"].is_unique, f"a session appears twice in {session_participant_csv.name}"

    missing = sorted(set(sessions["session_id"]) - set(table["session_id"]))
    if missing:
        print(f"WARNING: {len(missing)} sessions are not in {session_participant_csv.name}: add them by hand")
        print("\n".join(f"  {session_id}" for session_id in missing))

    sessions = sessions.copy()
    table = table.set_index("session_id")
    sessions["participant_id"] = sessions["session_id"].map(table["participant_id"])
    sessions["visit"] = pd.to_numeric(sessions["session_id"].map(table["visit"])).astype("Int64")
    return sessions


# ---- Trials ----

def end_type(reason):
    """Groups the ~12 different end reasons into 4 simple categories."""
    if reason == "start_returned":
        return "auto"          # the person came back to the start position: normal end
    if reason.startswith("manual"):
        return "manual"        # the operator pressed a key/button to go on
    if reason == "returned_to_a_without_target":
        return "failed"        # the person came back without reaching the target
    return "interrupted"       # shutdown, replaced, operator_stop


def trials_of_session(session_id, raw_root=RAW_ROOT):
    """The trials of one session (one dict per trial). We walk through its events in time order:
    a trial starts at "trial_start" and ends at the next "trial_end"; everything in between is noted on that trial."""
    events = read_events(session_id, raw_root)
    if not events:   # empty events file -> no trials
        return []
    # Times of all frames of the session (sorted), to count the frames inside each trial.
    frame_times = np.sort(read_frames(session_id, ["wall_time"], raw_root=raw_root)["wall_time"].to_numpy())

    rows = []
    task_runs = {}   # how many times each task has been started so far in this session
    current = None   # the trial being filled in (None = we are between two trials)
    for event in events:
        event_type = event["type"]

        if event_type == "task_start":
            # A task starts (again): count it, so repeated runs of the same task get task_run = 1, 2, ...
            task_runs[event["task_id"]] = task_runs.get(event["task_id"], 0) + 1

        elif event_type == "trial_start":
            # A new trial begins: open a new row with what we already know.
            current = {
                "session_id": session_id,
                "task_id": event["task_id"],
                "task_name": event["task_name"],
                "task_run": task_runs.get(event["task_id"], 1),
                "trial_index": event["trial_index"],
                "side": event["side"],
                "target": ((event.get("trial_spec") or {}).get("target") or {}).get("label"),  # "1"-"5" (task 4), "B"/"C"/"D" (task 6)
                "start_time": event["wall_time"],
                "pose_locked_time": np.nan,   # filled in below if the person reaches the start pose
                "goal_events": [],            # every goal event of the trial: type + time
            }

        elif current is None:
            # Event between two trials (countdown, task_start, operator keys, ...): not part of a trial.
            continue

        elif event_type == "trial_start_pose_locked":
            current["pose_locked_time"] = event["wall_time"]   # person in start position: the movement can begin

        elif event_type in GOAL_EVENTS:
            current["goal_events"].append({"type": event_type, "time": event["wall_time"]})

        elif event_type == "trial_end":
            # The trial is over: complete the row and store it.
            current["end_time"] = event["wall_time"]
            current["side"] = event["side"]   # arm actually used (the operator can switch arm during the countdown)
            current["duration_s"] = event["duration_s"]
            current["end_type"] = end_type(event["reason"])
            current["goal_reached"] = len(current["goal_events"]) > 0
            current["goal_time"] = current["goal_events"][0]["time"] if current["goal_events"] else np.nan
            # Number of frames with start_time <= wall_time <= end_time.
            current["n_frames"] = int(np.searchsorted(frame_times, current["end_time"], side="right")
                                      - np.searchsorted(frame_times, current["start_time"], side="left"))
            rows.append(current)
            current = None
    return rows


def build_trials(sessions, raw_root=RAW_ROOT):
    """One row per trial (= one repetition) of the given sessions. Every trial is kept (repetition 1, failed, ...)."""
    rows = []
    for session_id in sessions["session_id"]:
        rows += trials_of_session(session_id, raw_root)
    trials = pd.DataFrame(rows)

    # Unique trial ID = session ID + position of the trial in the session (001, 002, ...).
    trials["trial_id"] = trials["session_id"] + "_" + (trials.groupby("session_id").cumcount() + 1).map("{:03d}".format)
    # Repetition 1 of each run includes the ~5 s countdown and is often a practice movement.
    trials["is_first_repetition"] = trials["trial_index"] == 1

    trials = trials[[
        "trial_id", "session_id", "task_id", "task_name", "task_run", "trial_index", "is_first_repetition", "side", "target",
        "start_time", "pose_locked_time", "goal_time", "end_time", "duration_s",
        "end_type", "goal_reached", "goal_events", "n_frames",
    ]]
    assert trials["trial_id"].is_unique
    return trials


def add_quality_flags(trials, frames, max_wrong_person_pct=MAX_WRONG_PERSON_PCT, min_frames=MIN_FRAMES_PER_TRIAL):
    """Adds 2 columns to the trials table (flags only, nothing is removed):
    - wrong_person_pct: % of the trial's frames where the camera followed someone else (NaN if the trial has no frames)
    - passes_checks: goal reached, normal end (auto or manual), camera on the main person, and enough frames."""
    trials = trials.copy()
    wrong_person_pct = (~frames["is_main_person"]).groupby(frames["trial_id"]).mean() * 100
    trials["wrong_person_pct"] = trials["trial_id"].map(wrong_person_pct)
    trials["passes_checks"] = (
        trials["goal_reached"]
        & trials["end_type"].isin(["auto", "manual"])
        & (trials["wrong_person_pct"] <= max_wrong_person_pct)   # NaN (no frames) -> False
        & (trials["n_frames"] >= min_frames)
    )
    return trials


# ---- Frames ----

def build_frames(trials, raw_root=RAW_ROOT, max_distance_m=MAX_DISTANCE_MAIN_PERSON_M):
    """One row per frame inside a trial: trial_id, time_s, is_main_person and the x/y/z/conf of the 14 body points.
    Frames between trials are dropped. Body points are float32 (half the memory, still well below 1 mm precision)."""
    columns = ["pose_t", "wall_time"] + KEYPOINT_COLUMNS
    pieces = []
    for session_id, session_trials in trials.groupby("session_id"):
        raw = read_frames(session_id, columns, dtype={c: "float32" for c in KEYPOINT_COLUMNS}, raw_root=raw_root)
        wall_times = raw["wall_time"].to_numpy()

        for trial in session_trials.itertuples():
            # Frames with start_time <= wall_time <= end_time (same rule as n_frames in the trials table).
            first = np.searchsorted(wall_times, trial.start_time, side="left")
            last = np.searchsorted(wall_times, trial.end_time, side="right")
            piece = raw.iloc[first:last].copy()
            if piece.empty:
                continue
            piece["trial_id"] = trial.trial_id
            piece["time_s"] = time_since_start_pose(piece, trial)
            # The task run the frame belongs to: the chair can move between runs (used by is_main_person).
            piece["run"] = f"{session_id}_{trial.task_id}_{trial.task_run}"
            pieces.append(piece)

    frames = pd.concat(pieces, ignore_index=True)
    frames["is_main_person"] = is_main_person(frames, max_distance_m)
    frames = frames[["trial_id", "time_s", "is_main_person"] + KEYPOINT_COLUMNS]

    # Every trial has as many frames here as counted in the trials table.
    assert (frames.groupby("trial_id").size().reindex(trials["trial_id"], fill_value=0).to_numpy()
            == trials["n_frames"].to_numpy()).all()
    return frames


def time_since_start_pose(piece, trial):
    """time_s of the frames of one trial: 0 = person in start position (or start of the trial if that never happened).
    Negative = countdown / waiting before the movement.
    Frame times are on the camera clock (pose_t), event times on the software clock (wall_time). The reference time is
    moved to the camera clock by removing the usual delay between the two clocks in this trial (~0.013 s)."""
    reference_wall = trial.pose_locked_time if not np.isnan(trial.pose_locked_time) else trial.start_time
    delay = (piece["wall_time"] - piece["pose_t"]).median()
    return piece["pose_t"] - (reference_wall - delay)


def is_main_person(frames, max_distance_m=MAX_DISTANCE_MAIN_PERSON_M):
    """True for the frames of the main person, False when the camera followed someone else (e.g. the operator).
    The person sits still during a task, so a frame is the main person if the shoulder centre is within max_distance_m
    of its usual place in that task run. Usual place = where the shoulders are most often: the most frequent 10 cm box,
    then the median inside it. (Not the median of all frames: when 2 people take turns in front of the camera, that
    median falls between them and nobody would be the main person.)"""
    shoulder_centre = pd.DataFrame({c: (frames[f"left_shoulder_{c}"] + frames[f"right_shoulder_{c}"]) / 2 for c in "xyz"})
    box = shoulder_centre.div(0.10).round().astype("Int64").astype(str).agg("_".join, axis=1)   # e.g. "3_12_-5"
    has_shoulders = shoulder_centre.notna().all(axis=1)
    usual_box = box[has_shoulders].groupby(frames["run"][has_shoulders]).agg(lambda boxes: boxes.value_counts().index[0])
    in_usual_box = box == frames["run"].map(usual_box)
    usual_place = shoulder_centre[in_usual_box].groupby(frames["run"][in_usual_box]).median()
    distance = np.sqrt(((shoulder_centre - usual_place.reindex(frames["run"]).to_numpy()) ** 2).sum(axis=1, min_count=3))
    return distance <= max_distance_m


# ---- Save ----

def save_tables(sessions, trials, frames, processed_root=PROCESSED_ROOT):
    """Writes the 3 tables as parquet files in processed_root (overwrites them)."""
    processed_root.mkdir(parents=True, exist_ok=True)
    for name, table in [("sessions", sessions), ("trials", trials), ("frames", frames)]:
        path = processed_root / f"{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"{path.name:<18} {len(table):>9,} rows  {path.stat().st_size / 1e6:7.1f} MB")
