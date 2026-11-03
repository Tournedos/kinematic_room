"""Measures of a movement, and the labels to compare visits (pre / post) and arms (contralesional / ipsilesional,
affected / less affected). Used by exploration/01_patient_visits.ipynb; the pipeline will reuse the measures later.

Input: the cut movements of step 3 (movement_cut.parquet): filtered positions in the person's own directions
(forward / out / up, metres), measured from the shoulder (shoulder = 0), left arm mirrored, on the trial's regular
time grid. Per moment: 2 joint angles. Per trial: how the 2 angles move together (coupling), their range, and how
smooth the wrist path is (jerk).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import shapes
from config import PARTICIPANTS_CSV, PROCESSED_ROOT, SESSION_PARTICIPANT_CSV
from preprocess import read_hand_filled_csv

ELBOW = ["elbow_forward", "elbow_out", "elbow_up"]
WRIST = ["wrist_forward", "wrist_out", "wrist_up"]
ANGLES = ["elbow_angle_deg", "shoulder_elevation_deg"]
VISIT_LABELS = {1: "pre", 2: "post"}   # a treatment was given between the 2 visits


# ---- Data and labels ----

def load_data(processed_root=PROCESSED_ROOT, participants_csv=PARTICIPANTS_CSV):
    """The cut movements (positions per moment, step 3) and one row per trial kept by step 3, with: who (person,
    condition, affected_side, lesion_side), visit (1, 2) and visit_label (pre, post), arm_label, the checks of step 1
    (passes_checks, is_first_repetition, end_type), frame_rate_hz and arm_length_m (step 2), movement_duration_s (step 3)."""
    movement_cut, cut_trials, trials, sessions, kinematics_trials = shapes.load_inputs(processed_root)
    info = shapes.trial_info(trials, shapes.session_people(sessions), kinematics_trials, cut_trials)
    info = (info[info["in_step3"]]
            .merge(kinematics_trials[["trial_id", "frame_rate_hz"]], on="trial_id")
            .merge(trials[["trial_id", "end_type"]], on="trial_id"))
    participants = read_hand_filled_csv(participants_csv).set_index("participant_id")
    for column in ["affected_side", "lesion_side"]:   # filled by hand; empty = unknown
        info[column] = info["person"].map(participants[column]) if column in participants else np.nan
    info["arm_label"] = arm_labels(info)
    info["visit_label"] = info["visit"].map(VISIT_LABELS)
    return movement_cut, info


def _side(column):
    """A side column of participants.csv as "left", "right" or "" (unknown), whatever the case or spaces."""
    return column.astype("string").str.strip().str.lower().fillna("")


def arm_labels(trials):
    """The arm of each trial (participants.csv), in this order of preference:
    - lesion_side known (stroke: the brain hemisphere of the lesion): "contralesional" = the arm on the other side
      (usually the affected one) / "ipsilesional" = the arm on the same side;
    - otherwise affected_side known: "affected" / "less affected";
    - otherwise: "left" / "right".
    Healthy participants: always "left" / "right"."""
    lesion, affected = _side(trials["lesion_side"]), _side(trials["affected_side"])
    patient = trials["condition"] != "healthy"
    by_lesion = np.where(trials["side"] == lesion, "ipsilesional", "contralesional")
    by_affected = np.where(trials["side"] == affected, "affected", "less affected")
    return pd.Series(np.select([patient & lesion.isin(["left", "right"]), patient & affected.isin(["left", "right"])],
                               [by_lesion, by_affected], trials["side"]), index=trials.index)


def select_trials(info, settings):
    """The trials looked at: only those that pass the checks of step 1 (only_passing_trials) and without
    repetition 1, the practice trial (skip_first_repetition)."""
    keep = pd.Series(True, index=info.index)
    if settings["only_passing_trials"]:
        keep &= info["passes_checks"]
    if settings["skip_first_repetition"]:
        keep &= ~info["is_first_repetition"]
    return info[keep]


def in_tasks(trials, tasks):
    """True for the trials of the chosen tasks: a list (e.g. [1, 2]), one task (e.g. 2), or None = all tasks."""
    return trials["task_id"].isin(np.atleast_1d(tasks)) if tasks is not None else pd.Series(True, index=trials.index)


def of_task(trials, task_id):
    """The trials of one task."""
    return trials[trials["task_id"] == task_id]


def task_title(info, task_id):
    """e.g. "Task 2 (Shoulder abduction with elbow extended)"."""
    return f"Task {task_id} ({info.loc[info['task_id'] == task_id, 'task_name'].iloc[0]})"


def patient_trials(selected, patients, tasks=None):
    """The selected trials of the chosen patients (one participant_id or a list) in the chosen tasks (list, one task,
    or None = all), in sessions with a known visit (pre or post)."""
    keep = selected["person"].isin(np.atleast_1d(patients)) & in_tasks(selected, tasks) & selected["visit_label"].notna()
    return selected[keep]


def describe_choice(info, patients, tasks):
    """Prints what the settings PATIENTS and TASKS select: for each patient, the condition and the visits found
    (a warning for an unknown participant_id), and the names of the tasks."""
    for patient in map(str, np.atleast_1d(patients)):
        own = info[info["person"] == patient]
        if own.empty:
            print(f"WARNING: no trials for '{patient}': check the participant_id (session_participant.csv)")
            continue
        visits = [v for v in VISIT_LABELS.values() if v in set(own["visit_label"])]
        print(f"{patient} ({own['condition'].iloc[0]}): visits {', '.join(visits) or 'none (visit column empty)'}")
    for task_id in np.atleast_1d(tasks) if tasks is not None else sorted(info["task_id"].unique()):
        print(task_title(info, task_id))


def healthy_trials(selected, tasks=None, show=True):
    """The selected trials of the healthy participants in the chosen tasks (list, one task, or None = all), both arms
    (the left arm is mirrored, so both are comparable), as a grey reference. Empty if show is False."""
    keep = (selected["condition"] == "healthy") & in_tasks(selected, tasks) & show
    return selected[keep]


def healthy_overview(healthy):
    """Per task, what the healthy reference is made of: number of trials and sessions, which arm, the camera frame
    rate and how the trials were ended (% by the operator)."""
    return (healthy.groupby(["task_id", "task_name"])
            .agg(trials=("trial_id", "size"), sessions=("session_id", "nunique"),
                 right_arm=("side", lambda sides: (sides == "right").sum()),
                 left_arm=("side", lambda sides: (sides == "left").sum()),
                 frames_per_second=("frame_rate_hz", "median"),
                 ended_by_operator_pct=("end_type", lambda ends: 100 * (ends == "manual").mean()))
            .round({"frames_per_second": 1, "ended_by_operator_pct": 0}))


def trials_per_task(patients):
    """Number of trials per task (rows) and per patient and visit (columns)."""
    visit = pd.Categorical(patients["visit_label"], list(VISIT_LABELS.values()), ordered=True)
    return pd.crosstab([patients["task_id"], patients["task_name"]], [patients["person"], visit],
                       colnames=["person", "visit"])


def visits_table(info, patients, session_participant_csv=SESSION_PARTICIPANT_CSV):
    """The sessions of the chosen patients, as used here: which visit, when it was recorded, the ID typed
    during the recording, the number of trials (kept by step 3, all tasks), the camera frame rate, how the trials
    were ended (% by the operator), and the notes of session_participant.csv (where the assignments made by hand,
    to be confirmed, are written)."""
    hand = read_hand_filled_csv(session_participant_csv).set_index("session_id")
    rows = info[info["person"].isin(np.atleast_1d(patients)) & info["visit_label"].notna()].copy()
    rows["visit_label"] = pd.Categorical(rows["visit_label"], list(VISIT_LABELS.values()), ordered=True)
    table = (rows.groupby(["person", "condition", "visit_label", "session_id"], observed=True)
             .agg(trials=("trial_id", "size"), frames_per_second=("frame_rate_hz", "median"),
                  ended_by_operator_pct=("end_type", lambda ends: 100 * (ends == "manual").mean()))
             .reset_index())
    stamp = table["session_id"].str.removeprefix("upper_limb_")   # YYYYMMDD_HHMMSS
    table.insert(4, "recorded", stamp.str[6:8] + "/" + stamp.str[4:6] + " " + stamp.str[9:11] + ":" + stamp.str[11:13])
    table["typed_id"] = table["session_id"].map(hand["typed_id"]).fillna("(none)")
    table["notes"] = table["session_id"].map(hand["notes"]).fillna("")
    return table.drop(columns="session_id").round({"frames_per_second": 1, "ended_by_operator_pct": 0})


# ---- Angles (per moment) ----

def angle_between(u, v):
    """Angle in degrees between the vectors of each row of u and v."""
    cos = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(positions):
    """Per moment, from the elbow and wrist positions (shoulder = 0):
    - elbow_angle_deg: angle at the elbow between the upper arm and the forearm. 180 = straight arm.
    - shoulder_elevation_deg: angle between the upper arm and straight down. 0 = arm hanging, 90 = horizontal,
      180 = straight up.
    The shoulder is at 0, so upper arm = elbow position, forearm = wrist - elbow."""
    upper_arm = positions[ELBOW].to_numpy()
    forearm = positions[WRIST].to_numpy() - upper_arm
    down = np.array([[0.0, 0.0, -1.0]])   # (forward, out, up)
    return pd.DataFrame({"elbow_angle_deg": angle_between(-upper_arm, forearm),
                         "shoulder_elevation_deg": angle_between(upper_arm, down)}, index=positions.index)


def add_angles(movement_cut):
    """The cut movements with the 2 angles added to every moment."""
    return pd.concat([movement_cut, joint_angles(movement_cut)], axis=1)


# ---- Per trial: angles ----

def angle_measures(trial):
    """For one trial (its moments, with the 2 angles):
    - coupling_r: Pearson correlation between shoulder_elevation_deg and elbow_angle_deg over the movement. Does the
      elbow open (+) or bend (-) while the arm goes up? NaN if one of the angles does not move.
    - elbow_range_deg, shoulder_range_deg: largest - smallest angle in the movement.
    - max_elbow_angle_deg: the largest elbow angle (how straight the arm gets)."""
    angles = trial[ANGLES].dropna()
    elbow, shoulder = angles["elbow_angle_deg"], angles["shoulder_elevation_deg"]
    moves = len(angles) >= 3 and elbow.std() > 0 and shoulder.std() > 0
    return {"coupling_r": float(np.corrcoef(shoulder, elbow)[0, 1]) if moves else np.nan,
            "elbow_range_deg": elbow.max() - elbow.min(), "shoulder_range_deg": shoulder.max() - shoulder.min(),
            "max_elbow_angle_deg": elbow.max()}


# ---- Per trial: jerk (smoothness of the wrist path) ----

def derivatives(times, positions):
    """Velocity, acceleration and jerk of a 3D path: np.gradient on time_s, applied 3 times.
    A missing moment (NaN) also makes its neighbours missing, so a gap is left out instead of being bridged."""
    velocity = np.gradient(positions, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    jerk = np.gradient(acceleration, times, axis=0)
    return velocity, acceleration, jerk


def mean_squared_jerk(times, positions):
    """Mean over the movement of |jerk|^2 (jerk = third derivative of the position, m/s^3; result in (m/s^3)^2).
    Depends on the duration and the size of the movement, and on the camera rate: compare with care."""
    _, _, jerk = derivatives(times, positions)
    squared = (jerk ** 2).sum(axis=1)   # NaN where a coordinate is missing
    return float(np.nanmean(squared)) if np.isfinite(squared).any() else np.nan


def ldlj(times, positions):
    """Log dimensionless jerk, velocity-based (Balasubramanian, Melendez-Calderon, Roby-Brami & Burdet 2015,
    "On the analysis of movement smoothness", Journal of NeuroEngineering and Rehabilitation 12:112):

        LDLJ = -ln( (t2 - t1)^3 / v_peak^2 * integral from t1 to t2 of |d2v/dt2|^2 dt )

    v = wrist velocity (3D), v_peak = largest speed, t2 - t1 = duration of the movement, |d2v/dt2| = length of the
    3D jerk (multi-dimensional version, Melendez-Calderon et al. 2021). Multiplying by duration^3 / v_peak^2 removes
    the units: it does not depend on how long or how big the movement is, so movements of different lengths can be
    compared. Always negative; closer to 0 = smoother (fewer speed bumps).
    On the regular time grid, the integral = sum of |d2v/dt2|^2 x time step (missing moments left out).
    NaN if it cannot be computed (no movement).
    Careful: the derivatives are computed from the frames, so with few frames per movement (low camera rate, short
    movement) the value is pulled towards 0. E.g. a perfectly smooth 0.8 s movement: -5.32 (exact), -5.01 at 30
    frames per second, -4.57 at 17. Compare trials filmed at the same rate."""
    velocity, _, jerk = derivatives(times, positions)
    speed = np.linalg.norm(velocity, axis=1)
    squared = (jerk ** 2).sum(axis=1)
    if not np.isfinite(speed).any() or not np.isfinite(squared).any():
        return np.nan
    duration = times[-1] - times[0]
    v_peak = np.nanmax(speed)
    integral = np.nansum(squared) * np.median(np.diff(times))
    if v_peak <= 0 or integral <= 0:
        return np.nan
    return float(-np.log(duration ** 3 / v_peak ** 2 * integral))


def trial_measures(moments, info):
    """One row per trial (all trials of info): the angle measures and the jerk of the wrist, next to the trial's
    info (who, visit, arm, task, frame_rate_hz, movement_duration_s, checks). frame_rate_hz is kept because jerk is
    sensitive to the camera rate, which changes between sessions (12 to 38 frames per second)."""
    rows = []
    for trial_id, trial in moments[moments["trial_id"].isin(info["trial_id"])].groupby("trial_id", sort=False):
        times, wrist = trial["time_s"].to_numpy(), trial[WRIST].to_numpy()
        row = {"trial_id": trial_id, **angle_measures(trial)}
        enough = len(trial) >= 4   # the third derivative needs a few moments
        row["mean_squared_jerk"] = mean_squared_jerk(times, wrist) if enough else np.nan
        row["ldlj"] = ldlj(times, wrist) if enough else np.nan
        rows.append(row)
    return info.merge(pd.DataFrame(rows), on="trial_id")


# ---- Curves on a common x axis (for the plots) ----

TIME_STEP_S = 0.02   # seconds between points when the movements are not time-normalized


def curves(moments, trials, columns, settings):
    """Each trial's chosen columns on a common x axis, one row per trial and point, so trials can be averaged:
    - time_normalize True: x = 0 ... 100 % of the movement, n_points points evenly spaced in time;
      False: x = seconds since the movement start, every TIME_STEP_S (each trial stops at its own duration).
    - size_normalize True: positions divided by the trial's arm length (unit: arm lengths). Angles do not change.
    trials: the trials to take (with trial_id and arm_length_m)."""
    arm_length = trials.set_index("trial_id")["arm_length_m"]
    is_position = np.array([c in ELBOW + WRIST for c in columns])
    pieces = []
    for trial_id, trial in moments[moments["trial_id"].isin(trials["trial_id"])].groupby("trial_id", sort=False):
        trial = trial.dropna(subset=columns)
        if len(trial) < 2:
            continue
        t = (trial["time_s"] - trial["time_s"].iloc[0]).to_numpy()
        values = trial[columns].to_numpy(dtype=float)
        if settings["size_normalize"]:
            values[:, is_position] /= arm_length[trial_id]
        if settings["time_normalize"]:
            x, at = np.linspace(0, 100, settings["n_points"]), np.linspace(0, t[-1], settings["n_points"])
        else:
            x = np.round(np.arange(0, t[-1] + 1e-9, TIME_STEP_S), 3)
            at = x
        piece = pd.DataFrame({c: np.interp(at, t, values[:, j]) for j, c in enumerate(columns)})
        piece.insert(0, "x", x)
        piece.insert(0, "trial_id", trial_id)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["trial_id", "x"] + columns)


# ---- Plots for exploration/01_patient_visits.ipynb ----

# Reference palette: pre = categorical slot 1 (blue), post = slot 2 (orange); healthy = grey (a reference, not a series).
VISIT_COLOURS = {"pre": "#2a78d6", "post": "#eb6834"}
HEALTHY_LINE, HEALTHY_FILL = "#898781", "#e1e0d9"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _clean(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _arm_order(patient):
    """The patient's arm labels in a fixed order (contralesional or affected first)."""
    order = ["contralesional", "ipsilesional", "affected", "less affected", "left", "right"]
    return sorted(patient["arm_label"].unique(), key=order.index)


def _visit_order(patient):
    return [v for v in VISIT_COLOURS if v in set(patient["visit_label"])]


def _x_label(settings):
    return "progress of the movement (%)" if settings["time_normalize"] else "time since the movement start (s)"


def _y_label(column, settings):
    if column.endswith("_deg"):
        return f"{column.removesuffix('_deg').replace('_', ' ')} (degrees)"
    unit = "arm lengths" if settings["size_normalize"] else "m"
    return f"{column.replace('_', ' ')} ({unit})"


NUMBER_LABELS = {
    "movement_duration_s": "movement duration (s)",
    "shoulder_range_deg": "shoulder range (degrees)",
    "elbow_range_deg": "elbow range (degrees)",
    "max_elbow_angle_deg": "largest elbow angle (degrees)",
    "coupling_r": "shoulder-elbow coupling (correlation)",
    "mean_squared_jerk": "mean squared jerk ((m/s3)2)",
    "ldlj": "LDLJ (closer to 0 = smoother)",
}


def _number_label(measure):
    """Readable name of a per-trial number, with its unit."""
    return NUMBER_LABELS.get(measure, measure.replace("_", " "))


def _nothing_to_draw(trials, title):
    """True (and prints a message) when there is no trial to draw, e.g. a patient without trials in this task."""
    if trials.empty:
        print(f"Nothing to draw for {title}: no trials (check PATIENTS, TASKS and the trial selection).")
        return True
    return False


def _most_trials(grouped, n_trials):
    """Only the x values where at least half of the trials are still going (in seconds, trials stop at different
    times: the tail of a mean made of 1 or 2 long trials would be misleading)."""
    return grouped.count() >= max(1, n_trials / 2)


def _curve_panel(ax, curve_table, patient, arm, healthy, column):
    """One panel: the healthy median and 25-75 % band (grey), then for pre and post: every repetition of this arm as
    a thin line and their mean as a thick line. healthy = the rows of curve_table of the healthy trials."""
    n_healthy = healthy["trial_id"].nunique()
    if n_healthy:
        grouped = healthy.groupby("x")[column]
        band = grouped.quantile([0.25, 0.5, 0.75]).unstack()[_most_trials(grouped, n_healthy)]
        ax.fill_between(band.index, band[0.25], band[0.75], color=HEALTHY_FILL, linewidth=0)
        ax.plot(band.index, band[0.5], color=HEALTHY_LINE, linewidth=1.5)
    for visit in _visit_order(patient):
        ids = patient.loc[(patient["arm_label"] == arm) & (patient["visit_label"] == visit), "trial_id"]
        own = curve_table[curve_table["trial_id"].isin(ids)]
        for _, line in own.groupby("trial_id"):
            ax.plot(line["x"], line[column], color=VISIT_COLOURS[visit], linewidth=0.8, alpha=0.35)
        if len(own):
            grouped = own.groupby("x")[column]
            mean = grouped.mean()[_most_trials(grouped, own["trial_id"].nunique())]
            ax.plot(mean.index, mean, color=VISIT_COLOURS[visit], linewidth=2.5)
    _clean(ax)


def plot_curves_patients(curve_table, patients, healthy_ids, column, settings, title):
    """The same as plot_curves for one column, for several patients: one row per patient, one panel per arm
    (affected first if known, otherwise left then right). Same y scale in every panel, to compare the patients."""
    if _nothing_to_draw(patients, title):
        return
    persons = list(dict.fromkeys(patients.sort_values("person")["person"]))
    fig, panels = plt.subplots(len(persons), 2, figsize=(12, 3 * len(persons)), squeeze=False, sharex=True, sharey=True)
    healthy = curve_table[curve_table["trial_id"].isin(healthy_ids)]
    for i, person in enumerate(persons):
        own = patients[patients["person"] == person]
        arms = _arm_order(own)
        for j in range(2):
            ax = panels[i][j]
            if j >= len(arms):   # only one arm recorded for this patient in this task
                ax.set_visible(False)
                continue
            _curve_panel(ax, curve_table, own, arms[j], healthy, column)
            ax.set_title(f"{person} ({own['condition'].iloc[0]}), {arms[j]} arm", fontsize=10, color=INK)
            ax.set_ylabel(_y_label(column, settings), fontsize=9, color=INK_SECONDARY)
            if i == len(persons) - 1:
                ax.set_xlabel(_x_label(settings), fontsize=9, color=INK_SECONDARY)
    _legend(fig, patients, healthy["trial_id"].nunique())
    fig.suptitle(f"{title}: {_y_label(column, settings)}", fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


def _legend(fig, patients, n_healthy):
    """Legend of the curve plot: pre and post (thin = each repetition, thick = mean), and the healthy reference."""
    handles = [Line2D([], [], color=VISIT_COLOURS[v], linewidth=2.5, label=f"{v}: thin = each repetition, thick = mean")
               for v in _visit_order(patients)]
    if n_healthy:
        handles.append(Line2D([], [], color=HEALTHY_LINE, linewidth=1.5, label=f"healthy median ({n_healthy} trials)"))
        handles.append(Patch(color=HEALTHY_FILL, label="healthy 25-75 %"))
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 0.95))


def _values_of(values, patient, arm, visit):
    """The values (one per trial) of one patient's trials with this arm at this visit."""
    ids = patient.loc[(patient["arm_label"] == arm) & (patient["visit_label"] == visit), "trial_id"]
    return values.reindex(ids).dropna()


def _draw_points(ax, groups):
    """groups = [(label, colour, values), ...]: one column of points per group (small sideways jitter so points do
    not hide each other, the same every run), a short black line at the median, the number of trials under the label."""
    rng = np.random.default_rng(0)
    for k, (label, colour, group) in enumerate(groups):
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(group)), group, s=30, color=colour, alpha=0.8,
                   edgecolor="white", linewidth=0.5)
        if len(group):
            ax.plot([k - 0.25, k + 0.25], [group.median()] * 2, color=INK, linewidth=2)
    ax.set_xticks(range(len(groups)), [f"{label}\n({len(group)})" for label, _, group in groups], fontsize=8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _clean(ax)


def plot_measures_patients(per_trial, patients, healthy_ids, measure_names, title, log_measures=("mean_squared_jerk",)):
    """Several patients side by side: one row per measure, one column per patient. In each panel, for each arm:
    pre (blue) and post (orange), then the healthy trials (grey). Each point = one trial, black line = median.
    Same y scale along a row, to compare the patients."""
    if _nothing_to_draw(patients, title):
        return
    persons = list(dict.fromkeys(patients.sort_values("person")["person"]))
    fig, panels = plt.subplots(len(measure_names), len(persons), figsize=(4.6 * len(persons), 3 * len(measure_names)),
                               squeeze=False, sharey="row")
    for i, measure in enumerate(measure_names):
        values = per_trial.set_index("trial_id")[measure]
        healthy = values.reindex(healthy_ids).dropna()
        for j, person in enumerate(persons):
            own = patients[patients["person"] == person]
            groups = [(f"{arm}\n{visit}", VISIT_COLOURS[visit], _values_of(values, own, arm, visit))
                      for arm in _arm_order(own) for visit in _visit_order(own)]
            if len(healthy):
                groups.append(("healthy", HEALTHY_LINE, healthy))
            ax = panels[i][j]
            _draw_points(ax, groups)
            if measure in log_measures:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"{person} ({own['condition'].iloc[0]})", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(_number_label(measure), fontsize=9, color=INK_SECONDARY)
    fig.suptitle(f"{title} (each point = one trial, black line = median)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the title above the panel titles
    plt.show()


ARM_LINES = {"contralesional": "--", "ipsilesional": "-", "affected": "--", "less affected": "-", "left": "--",
             "right": "-"}
# One colour per patient, avoiding blue and orange (= pre and post in the other plots). Checked for colour-blind
# readers (Machado 2009 simulation, OKLab): worst normal-vision difference 15.6 (>= 15), worst colour-blind 9.1 (>= 8).
# Valid for these 3 colours only: with more patients the colours repeat, and the markers keep them apart.
PATIENT_COLOURS = ["#1baf7a", "#eda100", "#008300"]   # aqua, yellow, green
PATIENT_MARKERS = ["o", "s", "D", "^", "v", "P"]


def _patient_styles(patients):
    """{person: (colour, marker, label)}, in alphabetical order, e.g. {"DBB": ("#1baf7a", "o", "DBB (Stroke)"), ...}."""
    people = patients[["person", "condition"]].drop_duplicates("person").sort_values("person")
    return {row.person: (PATIENT_COLOURS[i % len(PATIENT_COLOURS)], PATIENT_MARKERS[i % len(PATIENT_MARKERS)],
                         f"{row.person} ({row.condition})") for i, row in enumerate(people.itertuples())}


def plot_visits_all_tasks(per_trial, patients, healthy, measure, log_scale=False):
    """All tasks at a glance for one number: one panel per task, x = visit (pre, post), y = the median of `measure`
    over the trials of each patient and arm: one line each (colour and marker = patient; solid = ipsilesional, less
    affected or right arm, dashed = contralesional, affected or left arm). Grey: the healthy trials of the task
    (median line and 25-75 % band). patients and healthy: their trials in all tasks."""
    if _nothing_to_draw(patients, measure):
        return
    styles = _patient_styles(patients)
    values = per_trial.set_index("trial_id")[measure]
    medians = (patients.assign(value=patients["trial_id"].map(values))
               .groupby(["task_id", "person", "arm_label", "visit_label"])["value"].median())
    tasks = patients[["task_id", "task_name"]].drop_duplicates().sort_values("task_id")
    n_columns = min(4, len(tasks))
    n_rows = int(np.ceil(len(tasks) / n_columns))
    fig, panels = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 3.75 * n_rows), squeeze=False)
    for ax in panels.ravel()[len(tasks):]:
        ax.set_visible(False)
    for ax, task in zip(panels.ravel(), tasks.itertuples()):
        reference = values.reindex(healthy.loc[healthy["task_id"] == task.task_id, "trial_id"]).dropna()
        if len(reference):
            ax.axhspan(reference.quantile(0.25), reference.quantile(0.75), color=HEALTHY_FILL, linewidth=0)
            ax.axhline(reference.median(), color=HEALTHY_LINE, linewidth=1.5)
        if task.task_id in medians.index.get_level_values("task_id"):
            for (person, arm), line in medians.loc[task.task_id].groupby(level=["person", "arm_label"]):
                line = line.droplevel(["person", "arm_label"]).reindex(list(VISIT_LABELS.values()))
                colour, marker, _ = styles[person]
                ax.plot([0, 1], line.to_numpy(), color=colour, marker=marker, markersize=8, markeredgecolor="white",
                        linewidth=2, linestyle=ARM_LINES.get(arm, "-"))
        ax.set_xticks([0, 1], list(VISIT_LABELS.values()))
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{task.task_id}. {task.task_name}", fontsize=9, color=INK)
        if log_scale:
            ax.set_yscale("log")
        _clean(ax)
    for ax in panels[:, 0]:
        ax.set_ylabel(f"median {_number_label(measure)}", fontsize=9, color=INK_SECONDARY)

    handles = [Line2D([], [], color=colour, marker=marker, markersize=8, linewidth=2, label=label)
               for colour, marker, label in styles.values()]
    arms = sorted(set(patients["arm_label"]), key=list(ARM_LINES).index)
    handles += [Line2D([], [], color=INK_SECONDARY, linewidth=2, linestyle=ARM_LINES[arm], label=f"{arm} arm")
                for arm in arms]
    if len(healthy):
        handles += [Line2D([], [], color=HEALTHY_LINE, linewidth=1.5, label="healthy median"),
                    Patch(color=HEALTHY_FILL, label="healthy 25-75 %")]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 0.95))
    fig.suptitle(f"Median {_number_label(measure)} per visit, per patient and arm", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


