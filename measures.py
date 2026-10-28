"""Measures of a movement, and the labels to compare visits (pre / post) and arms (affected / less affected).
Used by exploration/01_patient_visits.ipynb; the pipeline will reuse them later.

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
    condition, affected_side), visit (1, 2) and visit_label (pre, post), arm_label, the checks of step 1
    (passes_checks, is_first_repetition, end_type), frame_rate_hz and arm_length_m (step 2), movement_duration_s (step 3)."""
    movement_cut, cut_trials, trials, sessions, kinematics_trials = shapes.load_inputs(processed_root)
    info = shapes.trial_info(trials, shapes.session_people(sessions), kinematics_trials, cut_trials)
    info = (info[info["in_step3"]]
            .merge(kinematics_trials[["trial_id", "frame_rate_hz"]], on="trial_id")
            .merge(trials[["trial_id", "end_type"]], on="trial_id"))
    participants = read_hand_filled_csv(participants_csv).set_index("participant_id")
    if "affected_side" in participants:
        info["affected_side"] = info["person"].map(participants["affected_side"])
    else:
        info["affected_side"] = np.nan
    info["arm_label"] = arm_labels(info)
    info["visit_label"] = info["visit"].map(VISIT_LABELS)
    return movement_cut, info


def arm_labels(trials):
    """The arm of each trial: "affected" / "less affected" when the person's affected side is known
    (affected_side in participants.csv), otherwise "left" / "right". Healthy participants: always "left" / "right"."""
    affected_side = trials["affected_side"].astype("string").str.strip().str.lower().fillna("")   # "" = unknown
    known = affected_side.isin(["left", "right"]) & (trials["condition"] != "healthy")
    label = np.where(trials["side"] == affected_side, "affected", "less affected")
    return pd.Series(np.where(known, label, trials["side"]), index=trials.index)


def select_trials(info, settings):
    """The trials looked at: only those that pass the checks of step 1 (only_passing_trials) and without
    repetition 1, the practice trial (skip_first_repetition)."""
    keep = pd.Series(True, index=info.index)
    if settings["only_passing_trials"]:
        keep &= info["passes_checks"]
    if settings["skip_first_repetition"]:
        keep &= ~info["is_first_repetition"]
    return info[keep]


def patient_trials(selected, patient, task_id):
    """The selected trials of one patient in one task, in sessions with a known visit (pre or post)."""
    keep = (selected["person"] == patient) & (selected["task_id"] == task_id) & selected["visit_label"].notna()
    return selected[keep]


def healthy_trials(selected, task_id, show=True):
    """The selected trials of the healthy participants in one task (both arms: the left arm is mirrored, so both are
    comparable), as a grey reference. Empty if show is False."""
    keep = (selected["condition"] == "healthy") & (selected["task_id"] == task_id) & show
    return selected[keep]


def patients_with_two_visits(info):
    """Persons who have both visit 1 and visit 2 (from session_participant.csv)."""
    visits = info.dropna(subset=["visit"]).groupby("person")["visit"].agg(set)
    return sorted(person for person, done in visits.items() if {1, 2} <= done)


def two_visit_trials(selected, info, task_id):
    """The selected trials, in one task, of all the patients who have 2 visits (pre and post)."""
    keep = (selected["person"].isin(patients_with_two_visits(info)) & (selected["task_id"] == task_id)
            & selected["visit_label"].notna())
    return selected[keep]


def visits_table(info, session_participant_csv=SESSION_PARTICIPANT_CSV):
    """The sessions of the patients who have 2 visits, as used here: which visit, when it was recorded, the ID typed
    during the recording, the number of trials (kept by step 3, all tasks), the camera frame rate, how the trials
    were ended (% by the operator), and the notes of session_participant.csv (where the assignments made by hand,
    to be confirmed, are written)."""
    hand = read_hand_filled_csv(session_participant_csv).set_index("session_id")
    rows = info[info["person"].isin(patients_with_two_visits(info))].copy()
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


def visits_overview(selected, info, patient):
    """For one patient: number of trials per visit and arm (rows) and task (columns), and the median camera frame
    rate of each visit (jerk depends on it). Prints a message if the patient has only one visit."""
    own = info[info["person"] == patient]
    visits = sorted(own["visit_label"].dropna().unique(), key=list(VISIT_LABELS.values()).index)
    if len(visits) < 2:
        print(f"{patient} has only one visit ({', '.join(visits) or 'none'}): only this visit is shown.")
    rates = own.groupby("visit_label")["frame_rate_hz"].median().round(1)
    print("Camera frame rate (median, frames per second): "
          + ", ".join(f"{visit} {rates[visit]}" for visit in visits))
    table = pd.crosstab([selected.loc[selected["person"] == patient, "visit_label"],
                         selected.loc[selected["person"] == patient, "arm_label"]],
                        selected.loc[selected["person"] == patient, "task_id"])
    return table.reindex(visits, level=0)


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
    """The patient's arm labels in a fixed order (affected first)."""
    order = ["affected", "less affected", "left", "right"]
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


def plot_curves(curve_table, patient, healthy_ids, columns, settings, title):
    """One row per column, one panel per arm label. Every repetition = thin line, mean = thick line, pre (blue) and
    post (orange). Healthy participants (same task, both arms): median (grey line) and 25-75 % band (grey area)."""
    arms = _arm_order(patient)
    fig, panels = plt.subplots(len(columns), len(arms), figsize=(6 * len(arms), 3.1 * len(columns)),
                               squeeze=False, sharex=True)
    healthy = curve_table[curve_table["trial_id"].isin(healthy_ids)]
    for j, arm in enumerate(arms):
        for i, column in enumerate(columns):
            ax = panels[i][j]
            _curve_panel(ax, curve_table, patient, arm, healthy, column)
            ax.set_ylabel(_y_label(column, settings), fontsize=9, color=INK_SECONDARY)
            if i == 0:
                ax.set_title(f"{arm} arm", fontsize=10, color=INK)
            if i == len(columns) - 1:
                ax.set_xlabel(_x_label(settings), fontsize=9, color=INK_SECONDARY)
    _legend(fig, patient, healthy["trial_id"].nunique(), with_mean=True)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


def plot_curves_patients(curve_table, patients, healthy_ids, column, settings, title):
    """The same as plot_curves for one column, for several patients: one row per patient, one panel per arm
    (affected first if known, otherwise left then right). Same y scale in every panel, to compare the patients."""
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
    _legend(fig, patients, healthy["trial_id"].nunique(), with_mean=True, counts=False)
    fig.suptitle(f"{title}: {_y_label(column, settings)}", fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


def _legend(fig, patient, n_healthy, with_mean, counts=True):
    """Legend: pre and post (with their number of trials, both arms together, if counts), and the healthy reference.
    with_mean: the curve plots (thin = each repetition, thick = mean; healthy median and 25-75 % band);
    otherwise one line per trial (healthy trials as light grey lines)."""
    detail = ": thin = each repetition, thick = mean" if with_mean else ""
    handles = [Line2D([], [], color=VISIT_COLOURS[v], linewidth=2.5 if with_mean else 1.5,
                      label=(f"{v} ({(patient['visit_label'] == v).sum()} trials, both arms)" if counts else v) + detail)
               for v in _visit_order(patient)]
    if n_healthy and with_mean:
        handles.append(Line2D([], [], color=HEALTHY_LINE, linewidth=1.5, label=f"healthy median ({n_healthy} trials)"))
        handles.append(Patch(color=HEALTHY_FILL, label="healthy 25-75 %"))
    elif n_healthy:
        handles.append(Line2D([], [], color=HEALTHY_FILL, linewidth=1.5, label=f"healthy ({n_healthy} trials)"))
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 0.95))


def plot_elbow_distribution(moments, patient, healthy_ids, bin_width_deg=5):
    """Histogram of elbow_angle_deg over all moments of the movement, pre vs post, one panel per arm label.
    Each histogram is scaled to an area of 1 (visits do not have the same number of moments). Healthy: grey area."""
    arms = _arm_order(patient)
    bins = np.arange(0, 180 + bin_width_deg, bin_width_deg)
    healthy = moments.loc[moments["trial_id"].isin(healthy_ids), "elbow_angle_deg"].dropna()
    fig, panels = plt.subplots(1, len(arms), figsize=(6 * len(arms), 3.4), squeeze=False)
    for ax, arm in zip(panels[0], arms):
        if len(healthy):
            ax.hist(healthy, bins=bins, density=True, color=HEALTHY_FILL, label="healthy")
        for visit in _visit_order(patient):
            ids = patient.loc[(patient["arm_label"] == arm) & (patient["visit_label"] == visit), "trial_id"]
            values = moments.loc[moments["trial_id"].isin(ids), "elbow_angle_deg"].dropna()
            if len(values):
                ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2, color=VISIT_COLOURS[visit],
                        label=visit)
        ax.set_title(f"{arm} arm", fontsize=10, color=INK)
        ax.set_xlabel("elbow angle (degrees, 180 = straight arm)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("share of the moments (per degree)", fontsize=9, color=INK_SECONDARY)
        _clean(ax)
    panels[0][-1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Elbow angle over all moments of the movement", fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


def plot_trial_points(per_trial, patient, healthy_ids, measure, title, log_scale=False):
    """One value per trial as a point: pre, post and healthy side by side, one panel per arm label.
    Short black line = median. log_scale: for numbers that span several orders of magnitude (e.g. jerk)."""
    arms = _arm_order(patient)
    values = per_trial.set_index("trial_id")[measure]
    healthy = values.reindex(healthy_ids).dropna()
    fig, panels = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.6), squeeze=False, sharey=True)
    for ax, arm in zip(panels[0], arms):
        groups = [(visit, VISIT_COLOURS[visit], _values_of(values, patient, arm, visit)) for visit in _visit_order(patient)]
        if len(healthy):
            groups.append(("healthy", HEALTHY_LINE, healthy))
        _draw_points(ax, groups)
        ax.set_title(f"{arm} arm", fontsize=10, color=INK)
        if log_scale:
            ax.set_yscale("log")
    panels[0][0].set_ylabel(measure.replace("_", " "), fontsize=9, color=INK_SECONDARY)
    fig.suptitle(f"{title} (each point = one trial, black line = median)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the title above the panel titles
    plt.show()


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
                ax.set_ylabel(measure.replace("_", " "), fontsize=9, color=INK_SECONDARY)
    fig.suptitle(f"{title} (each point = one trial, black line = median)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the title above the panel titles
    plt.show()


def plot_angle_paths(moments, patient, healthy_ids):
    """For each trial: shoulder elevation (x) against elbow angle (y) over the movement, one thin line per trial,
    pre and post in 2 colours, one panel per arm label. Healthy trials: light grey lines behind."""
    arms = _arm_order(patient)
    fig, panels = plt.subplots(1, len(arms), figsize=(5.5 * len(arms), 4.6), squeeze=False, sharex=True, sharey=True)
    for ax, arm in zip(panels[0], arms):
        for _, trial in moments[moments["trial_id"].isin(healthy_ids)].groupby("trial_id"):
            ax.plot(trial["shoulder_elevation_deg"], trial["elbow_angle_deg"], color=HEALTHY_FILL, linewidth=0.8)
        for visit in _visit_order(patient):
            ids = patient.loc[(patient["arm_label"] == arm) & (patient["visit_label"] == visit), "trial_id"]
            for _, trial in moments[moments["trial_id"].isin(ids)].groupby("trial_id"):
                ax.plot(trial["shoulder_elevation_deg"], trial["elbow_angle_deg"], color=VISIT_COLOURS[visit],
                        linewidth=1, alpha=0.6)
        ax.set_title(f"{arm} arm", fontsize=10, color=INK)
        ax.set_xlabel("shoulder elevation (degrees, 0 = hanging, 90 = horizontal)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("elbow angle (degrees, 180 = straight)", fontsize=9, color=INK_SECONDARY)
        _clean(ax)
    _legend(fig, patient, len(set(healthy_ids)), with_mean=False)
    fig.suptitle("Coupling: elbow angle against shoulder elevation, one line per trial", fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


def _ldlj_rate_correlation(trials):
    """Spearman correlation between ldlj and frame_rate_hz (rank-based: any monotonic relation counts)."""
    return trials["ldlj"].corr(trials["frame_rate_hz"], method="spearman")


def frame_rate_check(per_trial):
    """Is the jerk influenced by the camera rate? ldlj against frame_rate_hz for all trials (grey points), and the
    Spearman correlation per task and over all trials. Returns the table of correlations.
    |r| >= 0.5 = strongly related: then the camera rate influences the jerk, and visits filmed at different rates
    should not be compared on jerk."""
    data = per_trial.dropna(subset=["ldlj", "frame_rate_hz"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(data["frame_rate_hz"], data["ldlj"], s=10, color=HEALTHY_LINE, alpha=0.4, linewidth=0)
    ax.set_xlabel("camera frame rate of the trial (frames per second)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("ldlj (closer to 0 = smoother)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(f"All trials ({len(data)}): smoothness against camera rate", fontsize=11, color=INK)
    _clean(ax)
    fig.tight_layout()
    plt.show()

    table = data.groupby(["task_id", "task_name"]).apply(_ldlj_rate_correlation, include_groups=False)
    table = table.rename("spearman_r").to_frame()
    table.loc[("all", "all tasks"), "spearman_r"] = _ldlj_rate_correlation(data)
    strong = table["spearman_r"].abs() >= 0.5
    if strong.any():
        print(f"Strongly related (|r| >= 0.5) in {strong.sum()} of {len(table)} rows: the camera rate influences the jerk. "
              "Compare visits on jerk only if they were filmed at the same rate.")
    else:
        print("No strong relation (all |r| < 0.5) between the camera rate and the jerk.")
    return table.round(2)


def summary_table(per_trial, patients):
    """Per patient, visit and arm label: number of trials and the medians of: movement duration, largest elbow angle,
    range of the shoulder elevation, coupling correlation, ldlj, and the camera frame rate (the jerk depends on it).
    patients: the trials of one or several patients."""
    rows = per_trial[per_trial["trial_id"].isin(patients["trial_id"])].copy()
    rows["visit_label"] = pd.Categorical(rows["visit_label"], list(VISIT_LABELS.values()), ordered=True)
    return (rows.groupby(["person", "visit_label", "arm_label"], observed=True)
            .agg(trials=("trial_id", "size"), movement_duration_s=("movement_duration_s", "median"),
                 max_elbow_angle_deg=("max_elbow_angle_deg", "median"),
                 shoulder_range_deg=("shoulder_range_deg", "median"), coupling_r=("coupling_r", "median"),
                 ldlj=("ldlj", "median"), frame_rate_hz=("frame_rate_hz", "median"))
            .round(2))
