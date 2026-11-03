"""Plots to check the data: the checks of 02_movement_table.ipynb and the exploration notebooks.
(The PCA plots are in pca_tools.py.)"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

from config import PROCESSED_ROOT
from movement import SHAPE_COLUMNS, TASK_NAMES, prepare_trial, smoothed_wrist, wrist_speed

BLUE, ORANGE, INK, INK_LIGHT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
GREY, GREY_LIGHT, GRID = "#b5b4ad", "#d9d8d2", "#e5e4df"


def _clean(panel):
    panel.grid(color=GRID, linewidth=0.8)
    panel.spines[["top", "right"]].set_visible(False)


# ---- 02_movement_table.ipynb: checks of step 4 ----

def plot_resampling_check(movement, shape_table, trial_id, settings, title):
    """Movement on the time grid (line) vs resampled points (dots), wrist height. The dots should sit on the line."""
    task_id = shape_table.set_index("trial_id").loc[trial_id, "task_id"]
    original = prepare_trial(movement[movement["trial_id"] == trial_id], settings, task_id)   # same preparation as step 4
    t = original["time_s"]
    progress = (t - t.iloc[0]) / (t.iloc[-1] - t.iloc[0])
    n_points = settings["n_points"]
    resampled = shape_table.loc[shape_table["trial_id"] == trial_id, [f"wrist_up_{i:03d}" for i in range(n_points)]].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(progress * 100, original["wrist_up"], color=BLUE, linewidth=2, label=f"before resampling ({len(original)} moments, {settings['rate_hz']} per second)")
    ax.scatter(np.linspace(0, 100, n_points), resampled, color=ORANGE, s=10, zorder=3, label=f"resampled ({n_points} points)")
    ax.set_xlabel("progress of the movement (%)")
    ax.set_ylabel("wrist height from the shoulder (m)")
    ax.set_title(f"{title} ({trial_id})", fontsize=10)
    _clean(ax)
    ax.legend(frameon=False)
    plt.show()


def plot_clinical_check(movement, shape_table, trial_id, settings, title):
    """Left: wrist speed over time, speed bumps and peak speed marked. Right: distance from the start, range marked."""
    task_id = shape_table.set_index("trial_id").loc[trial_id, "task_id"]
    original = prepare_trial(movement[movement["trial_id"] == trial_id], settings, task_id)
    xyz = smoothed_wrist(original, settings)   # same smoothing as clinical_measures
    t = original["time_s"].to_numpy() - original["time_s"].iloc[0]
    speed, _ = wrist_speed(xyz, t)
    bumps = find_peaks(speed, prominence=settings["speed_peak_prominence"] * speed.max())[0]
    from_start = np.linalg.norm(xyz - xyz[0], axis=1)
    measures = shape_table.set_index("trial_id").loc[trial_id]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4))
    left.plot(t[1:], speed, color=BLUE, linewidth=2)
    left.scatter(t[1:][bumps], speed[bumps], color=ORANGE, s=40, zorder=3, label=f"speed bumps: {measures['n_speed_peaks']:.0f}")
    left.axhline(measures["peak_speed_m_s"], color=INK_LIGHT, linestyle="--", linewidth=1,
                 label=f"peak speed: {measures['peak_speed_m_s']:.2f} m/s")
    left.set_xlabel("time (s)")
    left.set_ylabel("wrist speed (m/s)")
    left.set_title(f"speed (mean speed {measures['mean_speed_m_s']:.2f} m/s, duration {measures['movement_duration_s']:.2f} s)", fontsize=10)
    right.plot(t, from_start, color=BLUE, linewidth=2)
    right.axhline(measures["range_m"], color=INK_LIGHT, linestyle="--", linewidth=1, label=f"range: {measures['range_m']:.2f} m")
    right.set_xlabel("time (s)")
    right.set_ylabel("distance from the start (m)")
    right.set_title(f"distance from the start (path length {measures['path_length_m']:.2f} m)", fontsize=10)
    for panel in (left, right):
        _clean(panel)
        panel.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{title}: clinical measures", fontsize=11)
    fig.tight_layout()
    plt.show()


# ---- exploration/first_repetition.ipynb ----

def plot_first_repetition(frames, repetitions, side, title):
    """All repetitions of one task, wrist position over time (countdown removed): black = 1st repetition, grey = others."""
    fig, panels = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for panel, axis, label in zip(panels, "yxz", ["height", "forward / back", "side to side"]):
        for rep in repetitions.itertuples():
            path = frames[(frames["trial_id"] == rep.trial_id) & (frames["time_s"] >= 0) & frames["is_patient"]]
            first = rep.trial_index == 1
            panel.plot(path["time_s"], path[f"{side}_wrist_{axis}"],
                       color=INK if first else GREY, linewidth=2.5 if first else 1, zorder=3 if first else 2)
        panel.set_ylabel(f"wrist {label} (m)")
        _clean(panel)
    panels[-1].set_xlabel("time since the patient is in start position (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.show()


# ---- Two groups of persons compared: wrist paths in 3D (exploration/compare_groups.ipynb) ----

def group_paths(table, conditions, arm, per_session, only_way_up, n_points=100):
    """Wrist paths (from the shoulder: forward / out / up) of the trials of these conditions with this arm.
    only_way_up: each path from its start to its furthest point only. per_session: one average path per session.
    Returns (paths: array paths x n_points x 3, the trials used)."""
    rows = table[table["condition"].isin(conditions) & (table["side"] == arm)]
    paths = np.stack([rows.filter(like=f"{c}_").to_numpy() for c in SHAPE_COLUMNS], axis=-1)   # trials x points x 3
    resampled = []
    for path in paths:
        if only_way_up:
            path = path[:np.linalg.norm(path - path[0], axis=1).argmax() + 1]   # start -> furthest point
        # Same number of points for every path (needed to average them): value at evenly spaced moments.
        moments = np.linspace(0, len(path) - 1, n_points)
        resampled.append(np.column_stack([np.interp(moments, np.arange(len(path)), path[:, k]) for k in range(3)]))
    paths = np.array(resampled).reshape(-1, n_points, 3)
    if per_session:
        sessions = rows["trial_id"].str.rsplit("_", n=1).str[0].to_numpy()
        paths = np.array([paths[sessions == s].mean(axis=0) for s in pd.unique(sessions)]).reshape(-1, n_points, 3)
    return paths, rows


def plot_group_paths(task_id, group_a, group_b, arm="right", per_session=False, only_way_up=True,
                     processed_root=PROCESSED_ROOT):
    """3D plot (turn it with the mouse) of the wrist paths of 2 groups, e.g. group_a="parkinson", group_b="healthy".
    A group = one condition or a list of conditions. Thin lines = one trial (or one session), thick line = the group
    average. The shoulder is the fixed point (0, 0, 0). Returns the plotly figure."""
    table = pd.read_parquet(processed_root / "shape_tables.parquet", filters=[("task_id", "==", task_id)])
    groups = [[group_a] if isinstance(group_a, str) else list(group_a), [group_b] if isinstance(group_b, str) else list(group_b)]
    if set(groups[0]) & set(groups[1]):
        raise ValueError(f"A condition is in both groups: {set(groups[0]) & set(groups[1])}")

    fig = go.Figure()
    for conditions, colour in zip(groups, [ORANGE, BLUE]):
        paths, rows = group_paths(table, conditions, arm, per_session, only_way_up)
        name = " + ".join(conditions)
        if len(paths) == 0:
            print(f"{name}: no trial of task {task_id} with the {arm} arm")
            continue
        sessions = rows["trial_id"].str.rsplit("_", n=1).str[0].nunique()
        persons = ", ".join(sorted(rows["patient_id"].unique())) if rows["patient_id"].nunique() <= 4 else f"{rows['patient_id'].nunique()} persons"
        print(f"{name}: {len(rows)} trials, {sessions} session{'s' if sessions > 1 else ''} ({persons})")
        # All the thin lines in one trace (gaps = None between them): much faster than one trace per line.
        gap = np.full((len(paths), 1, 3), np.nan)
        lines = np.concatenate([paths, gap], axis=1).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=lines[:, 1], y=lines[:, 0], z=lines[:, 2], mode="lines", opacity=0.35,
                                   line=dict(color=colour, width=2), name=f"{name}: {len(paths)} {'sessions' if per_session else 'trials'}",
                                   legendgroup=name, hoverinfo="skip", connectgaps=False))
        average = paths.mean(axis=0)
        fig.add_trace(go.Scatter3d(x=average[:, 1], y=average[:, 0], z=average[:, 2], mode="lines",
                                   line=dict(color=colour, width=9), name=f"{name}: average", legendgroup=name,
                                   hovertemplate="out %{x:.2f} m · forward %{y:.2f} m · up %{z:.2f} m<extra>" + name + "</extra>"))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["shoulder"], textposition="top center",
                               marker=dict(size=6, color=INK), showlegend=False, hoverinfo="skip"))
    axis = dict(backgroundcolor="#ffffff", gridcolor=GRID, zerolinecolor=GREY)
    part = "start → furthest point" if only_way_up else "whole movement"
    fig.update_layout(
        title=f"Task {task_id}: {TASK_NAMES.get(task_id, '')}, {arm} arm, wrist path from the shoulder ({part})",
        scene=dict(xaxis=dict(title="out (m)", **axis), yaxis=dict(title="forward (m)", **axis),
                   zaxis=dict(title="up (m)", **axis), aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=1.2, z=0.5))),
        legend=dict(orientation="h", y=0.5, x=0, yanchor="bottom"),
        height=750, margin=dict(l=0, r=0, t=80, b=0))
    return fig


def plot_vs_healthy(task_id, arm="right", per_session=False, only_way_up=True, processed_root=PROCESSED_ROOT):
    """3D plot (turn it with the mouse): the healthy persons as a grey cloud (thin lines + their average), and one
    condition at a time over it, in colour (buttons above the plot to switch, the view stays the same).
    Returns the plotly figure."""
    table = pd.read_parquet(processed_root / "shape_tables.parquet", filters=[("task_id", "==", task_id)])
    table = table[table["side"] == arm]
    conditions = sorted(c for c in table["condition"].unique() if c != "healthy")

    def path_traces(paths, colour, name, visible, thin_opacity, thin_width, average_width):
        """Thin lines (all paths in one trace, gaps between them) + the average path."""
        gap = np.full((len(paths), 1, 3), np.nan)
        lines = np.concatenate([paths, gap], axis=1).reshape(-1, 3)
        average = paths.mean(axis=0)
        return [go.Scatter3d(x=lines[:, 1], y=lines[:, 0], z=lines[:, 2], mode="lines", opacity=thin_opacity,
                             line=dict(color=colour, width=thin_width), name=f"{name}: {len(paths)} {'sessions' if per_session else 'trials'}",
                             legendgroup=name, hoverinfo="skip", connectgaps=False, visible=visible),
                go.Scatter3d(x=average[:, 1], y=average[:, 0], z=average[:, 2], mode="lines", visible=visible,
                             line=dict(color=colour, width=average_width), name=f"{name}: average", legendgroup=name,
                             hovertemplate="out %{x:.2f} m · forward %{y:.2f} m · up %{z:.2f} m<extra>" + name + "</extra>")]

    # The healthy cloud: always visible.
    paths, rows = group_paths(table, ["healthy"], arm, per_session, only_way_up)
    n_healthy = rows["patient_id"].nunique()
    print(f"Healthy cloud: {len(rows)} trials from {n_healthy} persons")
    if n_healthy < 8:
        print(f"  ⚠️ only {n_healthy} healthy persons: the cloud is a few persons' habits, not 'what is normal'")
    traces = path_traces(paths, GREY, "healthy", True, 0.35, 2, 9)

    # One condition at a time: its traces are visible only when its button is chosen.
    buttons = []
    for k, condition in enumerate(conditions):
        paths, rows = group_paths(table, [condition], arm, per_session, only_way_up)
        persons = ", ".join(sorted(rows["patient_id"].unique()))
        sessions = rows["trial_id"].str.rsplit("_", n=1).str[0].nunique()
        print(f"{condition}: {len(rows)} trials, {sessions} session{'s' if sessions > 1 else ''} ({persons})")
        traces += path_traces(paths, ORANGE, f"{condition} ({persons})", k == 0, 0.5, 3, 10)
        visible = [True, True] + [False] * 2 * len(conditions)
        visible[2 + 2 * k], visible[3 + 2 * k] = True, True
        buttons.append(dict(label=f"{condition} ({persons})", method="restyle", args=[{"visible": visible}]))

    fig = go.Figure(traces)
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["shoulder"], textposition="top center",
                               marker=dict(size=6, color=INK), showlegend=False, hoverinfo="skip"))
    for button in buttons:
        button["args"][0]["visible"].append(True)   # the shoulder dot stays visible
    axis = dict(backgroundcolor="#ffffff", gridcolor=GRID, zerolinecolor=GREY)
    part = "start → furthest point" if only_way_up else "whole movement"
    fig.update_layout(
        title=f"Task {task_id}: {TASK_NAMES.get(task_id, '')}, {arm} arm, one condition vs the healthy cloud ({part})",
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.0, xanchor="left", yanchor="bottom", buttons=buttons)],
        scene=dict(xaxis=dict(title="out (m)", **axis), yaxis=dict(title="forward (m)", **axis),
                   zaxis=dict(title="up (m)", **axis), aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=1.2, z=0.5))),
        uirevision="keep the view",   # switching condition does not reset the view
        legend=dict(orientation="v", x=1.0, y=0.9),
        height=750, margin=dict(l=0, r=0, t=110, b=0))
    return fig
