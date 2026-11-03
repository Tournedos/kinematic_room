"""Step 4b: one PCA per task on the movement shapes (the shape table of shapes.py). Used by 04_shapes_pca.ipynb.

The old pca_tools.py, adapted for: shapes made of one or more body points (wrist, or wrist + elbow), positions in
metres or in arm lengths (`unit`), and the map: colour = condition, marker = person.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import PROCESSED_ROOT

DIRECTIONS = ["forward", "out", "up"]                  # order of the shape columns for each point
POINT_STYLES = {"wrist": "-", "elbow": "--"}           # line style of each body point in the plots

# Map: colour = condition (healthy in black, unknown in grey), marker = person (fixed in every task, so the 2 visits
# of a patient have the same marker).
CONDITION_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PERSON_MARKERS = ["s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">", "d"]
HEALTHY_COLOUR, UNKNOWN_COLOUR = "#0b0b0b", "#9a9993"
BACKGROUND_COLOUR = "#d9d8d2"   # light grey background dots in inspect_trial
AVERAGE_COLOUR, PLUS_COLOUR, MINUS_COLOUR = "#8a8983", "#2a78d6", "#eb6834"
GRID_COLOUR = "#e5e4df"


def is_shape_column(column):
    """True for shape columns like "wrist_forward_012" (not for speed_000 or info columns)."""
    parts = column.split("_")
    return len(parts) == 3 and parts[1] in DIRECTIONS and parts[2].isdigit()


def task_table(shape_table, task_id):
    """One task's rows of the shape table. Returns (table, shape_columns, info_columns, points, n_points)."""
    table = shape_table[shape_table["task_id"] == task_id].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"No trials for task {task_id} in the shape table")
    shape_columns = [c for c in table.columns if is_shape_column(c)]
    info_columns = [c for c in table.columns if c not in shape_columns and not c.startswith("speed_")]
    points = list(dict.fromkeys(c.split("_")[0] for c in shape_columns))   # e.g. ["wrist"] or ["wrist", "elbow"]
    n_points = len(shape_columns) // (len(points) * len(DIRECTIONS))
    return table, shape_columns, info_columns, points, n_points


def plot_styles(shape_table):
    """Colour and marker of each person who is not healthy, the same in every task: colour = condition (one colour
    per condition, unknown in grey), marker = person. Healthy persons are all drawn together, in black.
    Returns {person: (colour, marker, label)}, e.g. {"DBB": ("#2a78d6", "s", "DBB (Stroke)"), ...}."""
    people = shape_table.loc[shape_table["condition"] != "healthy", ["person", "condition"]].drop_duplicates("person")
    people = people.sort_values(["condition", "person"])
    conditions = sorted(set(people["condition"]) - {"unknown"})
    colours = {condition: CONDITION_COLOURS[i % len(CONDITION_COLOURS)] for i, condition in enumerate(conditions)}
    colours["unknown"] = UNKNOWN_COLOUR
    return {row.person: (colours[row.condition], PERSON_MARKERS[i % len(PERSON_MARKERS)], f"{row.person} ({row.condition})")
            for i, row in enumerate(people.itertuples())}


def remove_target_average(shapes, table):
    """Tasks with several targets (4 and 6): each trial's movement minus the average movement of everyone for the same
    target and arm (+ the overall average). So the PCA does not separate the targets, only how people differ for
    the same target. Tasks with one target: unchanged."""
    if "target" in table.columns and table["target"].nunique() > 1:
        same_target_average = shapes.groupby([table["target"], table["side"]]).transform("mean")
        shapes = shapes - same_target_average + shapes.mean()
    return shapes


def run_pca(table, shape_columns, info_columns, n_keep=10):
    """PCA on the shape columns (no rescaling: everything is in the same unit), after remove_target_average.
    Returns (pca, scores): scores = info columns + PC1 ... PC<n_keep>, one row per trial."""
    shapes = remove_target_average(table[shape_columns], table)
    pca = PCA()
    pc_values = pca.fit_transform(shapes.to_numpy())
    scores = table[info_columns].copy()
    for i in range(min(n_keep, pc_values.shape[1])):
        scores[f"PC{i + 1}"] = pc_values[:, i]
    return pca, scores


def split(vector, points, n_points):
    """{(point, direction): its n_points values} from one flat shape vector (same order as the shape columns)."""
    return {(point, direction): vector[(i * len(DIRECTIONS) + j) * n_points:(i * len(DIRECTIONS) + j + 1) * n_points]
            for i, point in enumerate(points) for j, direction in enumerate(DIRECTIONS)}


def _clean_axes(ax):
    ax.grid(color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_explained(pca, title, n_shown=10):
    """Bars = share of the differences explained by each component, line = running total."""
    n_shown = min(n_shown, len(pca.explained_variance_ratio_))
    explained = pca.explained_variance_ratio_[:n_shown] * 100
    cumulative = np.cumsum(pca.explained_variance_ratio_)[:n_shown] * 100
    components = np.arange(1, n_shown + 1)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(components, explained, color=PLUS_COLOUR, width=0.6, label="this component")
    ax.plot(components, cumulative, color="#52514e", linewidth=2, marker="o", markersize=5, label="running total")
    for x, y in zip(components[:3], explained[:3]):
        ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(components, [f"PC{c}" for c in components])
    ax.set_ylabel("share of the differences (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title}: what each component explains", fontsize=11)
    _clean_axes(ax)
    ax.legend(frameon=False, loc="center right")
    plt.show()


def plot_components(pca, points, n_points, title, n_pcs=3, unit="m"):
    """For each component: average movement (grey), average + component (blue), average - component (orange).
    Wrist = solid line, elbow = dashed line (if in the shape)."""
    progress = np.linspace(0, 100, n_points)
    average = split(pca.mean_, points, n_points)
    n_pcs = min(n_pcs, len(pca.components_))
    fig, panels = plt.subplots(n_pcs, 4, figsize=(15, 3.4 * n_pcs), squeeze=False)
    for k in range(n_pcs):
        amount = 2 * np.sqrt(pca.explained_variance_[k])          # "a lot" = 2 x the typical amount
        plus = split(pca.mean_ + amount * pca.components_[k], points, n_points)
        minus = split(pca.mean_ - amount * pca.components_[k], points, n_points)
        row = panels[k]

        # Side view: path of each point (forward vs up), same scale on both axes.
        for shape, colour, label in [(average, AVERAGE_COLOUR, "average"), (plus, PLUS_COLOUR, "+ PC"),
                                     (minus, MINUS_COLOUR, "− PC")]:
            for point in points:
                row[0].plot(shape[(point, "forward")], shape[(point, "up")], color=colour, linewidth=2,
                            linestyle=POINT_STYLES.get(point, "-"), label=label if point == points[0] else None)
        row[0].scatter(0, 0, color="#0b0b0b", s=30, zorder=3)
        row[0].annotate("shoulder", (0, 0), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#52514e")
        row[0].set_aspect("equal")
        row[0].set_xlabel(f"forward ({unit})")
        row[0].set_ylabel(f"PC{k + 1} ({pca.explained_variance_ratio_[k] * 100:.0f}%)\n\nup ({unit})")
        row[0].set_title("side view: " + " and ".join(points) + " path", fontsize=10)

        # Each direction over the movement.
        for panel, direction in zip(row[1:], ["up", "forward", "out"]):
            for shape, colour in [(average, AVERAGE_COLOUR), (plus, PLUS_COLOUR), (minus, MINUS_COLOUR)]:
                for point in points:
                    panel.plot(progress, shape[(point, direction)], color=colour, linewidth=2,
                               linestyle=POINT_STYLES.get(point, "-"))
            panel.set_xlabel("progress of the movement (%)")
            panel.set_ylabel(f"{direction} ({unit})")
            panel.set_title(f"{direction} over the movement", fontsize=10)

        for panel in row:
            _clean_axes(panel)

    panels[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    styles = ", wrist = solid, elbow = dashed" if len(points) > 1 else ""
    fig.suptitle(f"{title}: what the first {n_pcs} components change "
                 f"(grey = average, blue = + component, orange = − component{styles})", fontsize=11)
    fig.tight_layout()
    plt.show()


def _groups(scores, styles):
    """(label, mask, colour, marker) for: healthy (black circles), then each other person (styles of plot_styles)."""
    groups = [("healthy", scores["condition"] == "healthy", HEALTHY_COLOUR, "o")]
    for person, (colour, marker, label) in styles.items():
        groups.append((label, scores["person"] == person, colour, marker))
    return [group for group in groups if group[1].any()]   # skip groups with no trial in this task


def plot_map(scores, title, styles, pairs=(("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3"))):
    """Each dot = one trial, placed by its amount of each component. Colour = condition, marker = person."""
    groups = _groups(scores, styles)
    fig, panels = plt.subplots(1, len(pairs), figsize=(5.3 * len(pairs), 4.8), squeeze=False)
    for panel, (a, b) in zip(panels[0], pairs):
        for label, mask, colour, marker in groups:
            panel.scatter(scores.loc[mask, a], scores.loc[mask, b], color=colour, marker=marker, s=28,
                          edgecolor="white", linewidth=0.5, label=f"{label} ({mask.sum()})")
        panel.set_xlabel(a)
        panel.set_ylabel(b)
        _clean_axes(panel)
    panels[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1),
                         title="trials", title_fontsize=8)
    fig.suptitle(f"{title}: each dot = one trial", fontsize=11)
    fig.tight_layout()
    plt.show()


def run_task(shape_table, task_id, styles, n_keep=10, n_pcs_shown=3, unit="m"):
    """One task: PCA, then the 3 plots. styles = plot_styles(shape_table). Returns a small summary dict."""
    table, shape_columns, info_columns, points, n_points = task_table(shape_table, task_id)
    pca, scores = run_pca(table, shape_columns, info_columns, n_keep)
    title = f"Task {task_id} ({table['task_name'].iloc[0]})"

    persons = table.groupby("condition")["person"].nunique()
    print(f"{title}: {len(table)} trials from {table['person'].nunique()} persons "
          f"({', '.join(f'{condition}: {n}' for condition, n in persons.items())}), "
          f"{n_points} points x {len(DIRECTIONS)} directions x {len(points)} point(s) ({', '.join(points)})")
    plot_explained(pca, title)
    plot_components(pca, points, n_points, title, n_pcs_shown, unit)
    plot_map(scores, title, styles)
    return summary_row(task_id, table, pca)


def summary_row(task_id, table, pca):
    """How many trials and persons, and how much the first 3 components explain."""
    ratio = pca.explained_variance_ratio_ * 100
    return {"task_id": task_id, "trials": len(table), "persons": table["person"].nunique(),
            "PC1_%": round(ratio[0], 1), "PC2_%": round(ratio[1], 1), "PC3_%": round(ratio[2], 1),
            "PC1-3_total_%": round(ratio[:3].sum(), 1)}


def summary_all_tasks(shape_table):
    """summary_row for every task of the shape table (the PCA of each task, without the plots)."""
    rows = []
    for task_id in sorted(shape_table["task_id"].unique()):
        table, shape_columns, info_columns, _, _ = task_table(shape_table, task_id)
        pca, _ = run_pca(table, shape_columns, info_columns)
        rows.append(summary_row(task_id, table, pca))
    return pd.DataFrame(rows).set_index("task_id")


def all_scores(shape_table, n_keep=10):
    """PC1 ... PC<n_keep> of every trial (one PCA per task), with trial_id and task_id: for the next steps."""
    pieces = []
    for task_id in sorted(shape_table["task_id"].unique()):
        table, shape_columns, info_columns, _, _ = task_table(shape_table, task_id)
        _, scores = run_pca(table, shape_columns, info_columns, n_keep)
        pieces.append(scores[["trial_id", "task_id"] + [c for c in scores.columns if c.startswith("PC")]])
    return pd.concat(pieces, ignore_index=True)


def save_scores(scores, processed_root=PROCESSED_ROOT):
    """Writes pca_scores.parquet."""
    scores.to_parquet(processed_root / "pca_scores.parquet", index=False)
    print(f"pca_scores.parquet       {len(scores):,} trials x {scores.shape[1]} columns")


def inspect_trial(shape_table, task_id, person, side, repetition, styles, pcs=("PC1", "PC2"), unit="m"):
    """Critical check: where are one person's trials on the PCA map, and what does one repetition look like?
    person = as in the "person" column (e.g. "LSR", "noid_20260629_095022"); repetition = the real repetition number
    (trial_index: 1 = first of the task, usually not in the table). If the task was done twice (or the person came
    twice), the last one is used.

    Left: the map (all trials light grey, this person's trials in colour, the chosen repetition circled).
    Right: the chosen repetition (black) over the person's other repetitions (thin, colour) and the task average (grey),
    for up / forward / out over the movement (wrist solid, elbow dashed).
    """
    table, shape_columns, info_columns, points, n_points = task_table(shape_table, task_id)
    pca, scores = run_pca(table, shape_columns, info_columns)

    own = scores[(scores["person"] == person) & (scores["side"] == side)]
    matches = own[own["trial_index"] == repetition]
    if matches.empty:
        raise ValueError(f"{person}, {side} arm, task {task_id}: no repetition {repetition} in the table. "
                         f"Repetitions available: {sorted(own['trial_index'].unique())}")
    chosen = matches.sort_values(["session_id", "task_run"]).index[-1]   # task done twice: the last one
    colour = styles[person][0] if person in styles else PLUS_COLOUR   # healthy person: blue (black is the chosen one)

    progress = np.linspace(0, 100, n_points)
    average = split(pca.mean_, points, n_points)
    shapes = table[shape_columns].to_numpy()

    fig, panels = plt.subplots(1, 4, figsize=(18, 4.3))
    a, b = pcs

    # 1. Map: everyone light grey, this person in colour, the chosen repetition circled.
    others = ~scores.index.isin(own.index)
    panels[0].scatter(scores.loc[others, a], scores.loc[others, b], color=BACKGROUND_COLOUR, s=18, label="other trials")
    panels[0].scatter(own[a], own[b], color=colour, s=40, edgecolor="white", linewidth=0.5,
                      label=f"{person}, {side} arm ({len(own)})")
    panels[0].scatter(scores.loc[chosen, a], scores.loc[chosen, b], s=180, facecolor="none",
                      edgecolor="#0b0b0b", linewidth=2, label=f"repetition {repetition}")
    panels[0].set_xlabel(a)
    panels[0].set_ylabel(b)
    panels[0].set_title("map of the trials", fontsize=10)
    panels[0].legend(frameon=False, fontsize=8)

    # 2-4. The movement: other repetitions (thin), chosen repetition (black), task average (grey dashed).
    for panel, direction in zip(panels[1:], ["up", "forward", "out"]):
        for point in points:
            style = POINT_STYLES.get(point, "-")
            for index in own.index:
                if index != chosen:
                    panel.plot(progress, split(shapes[index], points, n_points)[(point, direction)],
                               color=colour, linewidth=1, alpha=0.5, linestyle=style)
            panel.plot(progress, average[(point, direction)], color=AVERAGE_COLOUR, linewidth=2, linestyle=":")
            panel.plot(progress, split(shapes[chosen], points, n_points)[(point, direction)], color="#0b0b0b",
                       linewidth=2.5, linestyle=style)
        panel.set_xlabel("progress of the movement (%)")
        panel.set_ylabel(f"{direction} ({unit})")
        panel.set_title(f"{direction} over the movement", fontsize=10)

    for panel in panels:
        _clean_axes(panel)
    fig.suptitle(f"Task {task_id}, {person}, {side} arm, repetition {repetition} ({scores.loc[chosen, 'trial_id']}): "
                 f"black = this repetition, thin = other repetitions, grey dotted = task average", fontsize=11)
    fig.tight_layout()
    plt.show()

    # Numbers: this repetition vs the person's typical values (median over their trials).
    pc_names = [f"PC{i + 1}" for i in range(3)]
    print(pd.DataFrame({"this repetition": scores.loc[chosen, pc_names].astype(float),
                        f"{person} median": own[pc_names].median(),
                        f"{person} spread (std)": own[pc_names].std(),
                        "all trials spread (std)": scores[pc_names].std()}).round(2).to_string())
