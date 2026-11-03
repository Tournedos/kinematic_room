"""Step 2: kinematics of the arm that did the task. Used by 02_movement_table.ipynb.

For every trial: the shoulder, elbow and wrist of that arm on a regular time grid, low-pass filtered, measured from
the shoulder in the person's own directions (forward / out / up), with 2 joint angles and their speeds.
Input: trials.parquet and frames.parquet (step 1). No trial is selected or removed here: later steps do that.
`settings` = the settings of the notebook, as a dict.
"""

import json

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import PROCESSED_ROOT

# The 4 body points used: both shoulders (their line gives the direction of the body) + elbow and wrist of the task arm.
POINTS = ["left_shoulder", "right_shoulder", "elbow", "wrist"]
XYZ = [f"{point}_{axis}" for point in POINTS for axis in "xyz"]
# Output positions: elbow and wrist from the shoulder, in the person's own directions.
POSITIONS = [f"{point}_{direction}" for point in ("elbow", "wrist") for direction in ("forward", "out", "up")]


def load_inputs(processed_root=PROCESSED_ROOT):
    """The trials table, and the frames (only the columns needed: shoulders, elbows and wrists)."""
    trials = pd.read_parquet(processed_root / "trials.parquet")
    points = [f"{side}_{point}_{value}" for side in ("left", "right") for point in ("shoulder", "elbow", "wrist")
              for value in ("x", "y", "z", "conf")]
    frames = pd.read_parquet(processed_root / "frames.parquet", columns=["trial_id", "time_s", "is_main_person"] + points)
    return trials, frames


# ---- 1. The frames used ----

def arm_frames(frames, trials, settings):
    """The movement frames (time_s >= 0) with the 4 points used: both shoulders, and the elbow and wrist of the arm
    that did the task (`side` in trials). A point counts as missing (NaN) when the camera followed someone else
    (is_main_person False) or, if min_confidence is set, when its confidence (_conf) is lower."""
    frames = frames[frames["time_s"] >= 0].merge(trials[["trial_id", "side"]], on="trial_id")
    is_right = (frames["side"] == "right").to_numpy()

    def column(point, value):
        # Shoulders: both, as they are. Elbow and wrist: those of the arm that did the task.
        if point.endswith("shoulder"):
            return frames[f"{point}_{value}"].to_numpy()
        return np.where(is_right, frames[f"right_{point}_{value}"], frames[f"left_{point}_{value}"])

    arm = frames[["trial_id", "side", "time_s"]].copy()
    for point in POINTS:
        missing = ~frames["is_main_person"].to_numpy()
        if settings["min_confidence"] is not None:
            missing |= column(point, "conf") < settings["min_confidence"]
        for axis in "xyz":
            arm[f"{point}_{axis}"] = np.where(missing, np.nan, column(point, axis).astype(float))
    return arm


# ---- 2. Regular time grid ----

def on_regular_grid(trial, max_gap_frames):
    """One trial on evenly spaced times, at the camera's median frame rate in this trial (time_s is the camera clock).
    A usable frame = all 4 points present. Values between usable frames are interpolated, but gaps of more than
    max_gap_frames missing frames are not bridged: they stay missing (NaN).
    Returns the grid table and a dict: frame_rate_hz, n_missing_filled, has_long_gap."""
    t = trial["time_s"].to_numpy()
    dt = np.median(np.diff(t))   # usual time between 2 frames
    usable = trial[XYZ].notna().all(axis=1).to_numpy()
    t_usable, values = t[usable], trial.loc[usable, XYZ].to_numpy()

    # Frames missing between 2 consecutive usable frames (missing points, or frames the camera did not record).
    # (at least 0: 2 frames can arrive closer together than usual)
    n_missing = np.maximum(np.round(np.diff(t_usable) / dt).astype(int) - 1, 0)
    long_gap = n_missing > max_gap_frames
    # Continuous pieces: a long gap starts a new piece. Interpolation only inside a piece.
    piece = np.r_[0, np.cumsum(long_gap)]

    n_points = int(np.floor((t_usable[-1] - t_usable[0]) / dt + 1e-9)) + 1
    grid_t = t_usable[0] + np.arange(n_points) * dt
    grid_values = np.full((n_points, len(XYZ)), np.nan)
    tolerance = 1e-3 * dt   # rounding errors: a grid time can land a tiny bit past the last frame of a piece
    for p in np.unique(piece):
        t_piece, values_piece = t_usable[piece == p], values[piece == p]
        inside = (grid_t >= t_piece[0] - tolerance) & (grid_t <= t_piece[-1] + tolerance)
        for j in range(len(XYZ)):
            grid_values[inside, j] = np.interp(grid_t[inside], t_piece, values_piece[:, j])

    grid = pd.DataFrame(grid_values, columns=XYZ)
    grid.insert(0, "time_s", grid_t)
    info = {"frame_rate_hz": 1 / dt, "n_missing_filled": int(n_missing[~long_gap].sum()),
            "has_long_gap": bool(long_gap.any())}
    return grid, info


# ---- 3. Filter ----

def low_pass(grid, cutoff_hz, frame_rate_hz):
    """Zero-phase low-pass Butterworth filter (order 4, run forward then backward with filtfilt, so peaks are not
    shifted in time) on each coordinate. filtfilt cannot handle missing values: each continuous piece is filtered on
    its own, and pieces too short to filter are left as they are.
    Returns the filtered table and the number of frames left unfiltered."""
    filtered = grid.copy()
    if cutoff_hz is None:
        return filtered, 0
    if cutoff_hz >= frame_rate_hz / 2:   # a filter cannot go above half the frame rate
        return filtered, len(grid)

    b, a = butter(4, cutoff_hz, fs=frame_rate_hz)
    min_length = 3 * max(len(a), len(b)) + 1   # shortest piece filtfilt accepts (16 frames)
    has_data = grid[XYZ].notna().all(axis=1).to_numpy()
    piece = np.cumsum(~has_data)   # consecutive rows with data share the same number
    n_unfiltered = 0
    for _, rows in pd.Series(np.flatnonzero(has_data)).groupby(piece[has_data]):
        rows = rows.to_numpy()
        if len(rows) < min_length:
            n_unfiltered += len(rows)
            continue
        filtered.loc[rows, XYZ] = filtfilt(b, a, grid.loc[rows, XYZ].to_numpy(), axis=0)
    return filtered, n_unfiltered


# ---- 4. Body frame ----

def facing_angle(grid, body_frame_from, start_frames):
    """How much the body is turned in this trial (degrees): direction of the line from the left to the right shoulder,
    seen from above. 0 = shoulders along the room's z axis (the usual case).
    body_frame_from: "start" = from the first start_frames frames (start posture), "median" = over the whole trial."""
    shoulders = grid.dropna(subset=XYZ)
    if body_frame_from == "start":
        shoulders = shoulders.head(start_frames)
    elif body_frame_from != "median":
        raise ValueError(f'body_frame_from must be "start" or "median", not {body_frame_from!r}')
    dx = (shoulders["right_shoulder_x"] - shoulders["left_shoulder_x"]).median()
    dz = (shoulders["right_shoulder_z"] - shoulders["left_shoulder_z"]).median()
    return float(np.degrees(np.arctan2(dx, dz)))


def to_body_frame(grid, side, facing_deg, mirror_left):
    """Elbow and wrist measured from the shoulder of the task arm, turned by facing_deg into the person's own
    directions: forward = in front of the chest, out = along the shoulder line (to the right), up = up (camera y).
    Some people sit turned: without this, the same arm raise would be "forward" for one and "forward + sideways" for
    another. mirror_left: "out" flipped for the left arm, so it is positive away from the body for both arms."""
    angle = np.radians(facing_deg)
    body = pd.DataFrame(index=grid.index)
    for point in ("elbow", "wrist"):
        dx = grid[f"{point}_x"] - grid[f"{side}_shoulder_x"]
        dy = grid[f"{point}_y"] - grid[f"{side}_shoulder_y"]
        dz = grid[f"{point}_z"] - grid[f"{side}_shoulder_z"]
        out = dx * np.sin(angle) + dz * np.cos(angle)
        body[f"{point}_forward"] = dx * np.cos(angle) - dz * np.sin(angle)
        body[f"{point}_out"] = -out if (mirror_left and side == "left") else out
        body[f"{point}_up"] = dy
    return body


# ---- 5. Angles, 6. speeds ----

def angle_between(u, v):
    """Angle in degrees between the vectors of each row of u and v."""
    cos = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(body):
    """elbow_angle_deg: angle at the elbow between upper arm and forearm (180 = straight arm).
    arm_elevation_deg: angle between the upper arm and straight down (0 = hanging, 90 = horizontal, 180 = straight up)."""
    upper_arm = body[["elbow_forward", "elbow_out", "elbow_up"]].to_numpy()          # shoulder -> elbow
    forearm = body[["wrist_forward", "wrist_out", "wrist_up"]].to_numpy() - upper_arm  # elbow -> wrist
    down = np.array([[0.0, 0.0, -1.0]])
    return pd.DataFrame({"elbow_angle_deg": angle_between(-upper_arm, forearm),
                         "arm_elevation_deg": angle_between(upper_arm, down)}, index=body.index)


def speeds(body, angles, frame_rate_hz):
    """Speeds on the regular grid (np.gradient). wrist_speed (m/s) = length of the wrist velocity.
    elbow_angle_speed, arm_elevation_speed (deg/s): + = the elbow opens / the arm goes up."""
    dt = 1 / frame_rate_hz
    wrist_velocity = np.gradient(body[["wrist_forward", "wrist_out", "wrist_up"]].to_numpy(), dt, axis=0)
    return pd.DataFrame({"wrist_speed": np.linalg.norm(wrist_velocity, axis=1),
                         "elbow_angle_speed": np.gradient(angles["elbow_angle_deg"].to_numpy(), dt),
                         "arm_elevation_speed": np.gradient(angles["arm_elevation_deg"].to_numpy(), dt)},
                        index=body.index)


def segment_lengths(body):
    """Median upper arm, forearm and arm length over the trial (m). Almost constant if the tracking is good;
    arm length is kept for a later size normalisation."""
    upper_arm = body[["elbow_forward", "elbow_out", "elbow_up"]].to_numpy()
    forearm = body[["wrist_forward", "wrist_out", "wrist_up"]].to_numpy() - upper_arm
    upper_length, forearm_length = np.linalg.norm(upper_arm, axis=1), np.linalg.norm(forearm, axis=1)
    return {"upper_arm_length_m": np.nanmedian(upper_length), "forearm_length_m": np.nanmedian(forearm_length),
            "arm_length_m": np.nanmedian(upper_length + forearm_length)}


# ---- All together ----

def trial_kinematics(trial, side, settings):
    """Steps 2 to 6 for one trial. Returns the frame table (one row per grid time) and a dict about the trial."""
    grid, info = on_regular_grid(trial, settings["max_gap_frames"])
    filtered, info["n_frames_unfiltered"] = low_pass(grid, settings["filter_cutoff_hz"], info["frame_rate_hz"])
    # The rotation angle comes from the filtered shoulders; the same rotation is used for the unfiltered copy.
    facing_deg = facing_angle(filtered, settings["body_frame_from"], settings["start_frames"])
    body = to_body_frame(filtered, side, facing_deg, settings["mirror_left"])

    parts = [grid[["time_s"]], body]
    if settings["keep_unfiltered"]:
        parts.append(to_body_frame(grid, side, facing_deg, settings["mirror_left"]).add_prefix("raw_"))
    angles = joint_angles(body)
    parts += [angles, speeds(body, angles, info["frame_rate_hz"])]

    info.update({"facing_deg": facing_deg, "n_frames": len(grid), **segment_lengths(body)})
    return pd.concat(parts, axis=1), info


def build_kinematics(frames, trials, settings):
    """Kinematics of every trial with at least 2 usable frames (all 4 points present).
    Returns (kinematics: one row per frame on the regular grid, kinematics_trials: one row per trial)."""
    arm = arm_frames(frames, trials, settings)
    sides = trials.set_index("trial_id")["side"]
    tables, rows = [], []
    for trial_id, trial in arm.groupby("trial_id", sort=False):
        if trial[XYZ].notna().all(axis=1).sum() < 2:
            continue
        table, info = trial_kinematics(trial, sides[trial_id], settings)
        table.insert(0, "trial_id", trial_id)
        tables.append(table)
        rows.append({"trial_id": trial_id, "side": sides[trial_id], **info})

    kinematics = pd.concat(tables, ignore_index=True)
    kinematics_trials = pd.DataFrame(rows)
    # Body turned by more than max_facing_deg: probably a tracking problem. Flag only, nothing is removed.
    kinematics_trials["facing_ok"] = kinematics_trials["facing_deg"].abs() <= settings["max_facing_deg"]
    kinematics_trials = kinematics_trials[[
        "trial_id", "side", "facing_deg", "facing_ok", "frame_rate_hz", "n_frames", "n_missing_filled",
        "has_long_gap", "n_frames_unfiltered", "upper_arm_length_m", "forearm_length_m", "arm_length_m",
    ]]
    return kinematics, kinematics_trials


def save_kinematics(kinematics, kinematics_trials, settings, processed_root=PROCESSED_ROOT):
    """Writes kinematics.parquet, kinematics_trials.parquet and the settings used (kinematics_settings.json)."""
    kinematics.to_parquet(processed_root / "kinematics.parquet", index=False)
    kinematics_trials.to_parquet(processed_root / "kinematics_trials.parquet", index=False)
    with open(processed_root / "kinematics_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    for name in ["kinematics.parquet", "kinematics_trials.parquet", "kinematics_settings.json"]:
        print(f"{name:<28} {(processed_root / name).stat().st_size / 1e6:7.1f} MB")
