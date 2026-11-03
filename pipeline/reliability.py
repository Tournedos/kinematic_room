"""Step 5: reliability (noise ceiling) of the numbers that describe a trial. Used by 05_reliability.ipynb.

For each task and each number (PC1..PC10 of the task's PCA, start_delay_s, movement_duration_s): is it a stable
property of the session, or does it jump from one repetition to the next? A number that is not stable can never
separate diseases, so this tells which numbers to keep for the next steps.

Split-half reliability, as compute_ceiling_splithalf of the NX-414 project (EEG noise ceiling), where
"EEG units" = the numbers, "stimuli" = the units here (one session x one arm), "repetitions" = the trials.
In each fold: the trials of each unit are split into 2 random halves of equal size, each half is averaged, and the
2 halves are correlated ACROSS units (Pearson or Spearman). The Spearman-Brown correction then gives the reliability
of the average of all the trials (each half has only half of them). Averaged over the folds.
`settings` = the settings of the notebook, as a dict.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import pca
from config import PROCESSED_ROOT

INFO_COLUMNS = ["trial_id", "session_id", "task_id", "task_name", "person", "condition", "visit", "side", "target",
                "trial_index", "start_delay_s", "movement_duration_s"]
MIN_UNITS = 3   # fewer units than this in a task: no correlation (with 2 units it is always +1 or -1)

# Chart colours (reference palette): one-hue blue ramp for the heatmap (light = 0, dark = 1), inks and grid.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK_SECONDARY, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARM_STYLES = {"right": "-", "left": "--"}


# ---- Inputs ----

def load_trials(processed_root=PROCESSED_ROOT):
    """One row per trial of the shape table (the trials chosen in step 4), with its info and its PCA scores
    (PC1..PC10 of its task, step 4)."""
    info = pd.read_parquet(processed_root / "shape_tables.parquet", columns=INFO_COLUMNS)
    scores = pd.read_parquet(processed_root / "pca_scores.parquet").drop(columns="task_id")
    return info.merge(scores, on="trial_id")


def step4_settings(processed_root=PROCESSED_ROOT):
    """The settings that made the shape table and the PCA scores (step 4), e.g. to know the SHAPE_PART used."""
    with open(processed_root / "shapes_settings.json") as f:
        return json.load(f)


def measure_names(n_pcs, extra_measures):
    """The numbers tested: PC1 ... PC<n_pcs>, then the extra measures (e.g. start_delay_s, movement_duration_s)."""
    return [f"PC{i}" for i in range(1, n_pcs + 1)] + list(extra_measures)


# ---- 1. Units ----

def add_units(trials, arms):
    """Adds 2 columns:
    - unit: what the halves are compared across (the "stimulus" of the EEG version): one session x one arm
      (arms "separate") or one session (arms "together").
    - split_group: the trials that are split together into the 2 halves. Same target (tasks 4 and 6 have several),
      and with arms "together" also the same arm: so both halves always contain the same targets (and arms), and a
      difference between the halves is not just "more trials to target B in this half"."""
    trials = trials.copy()
    target = trials["target"].fillna("-")
    if arms == "separate":
        trials["unit"] = trials["session_id"] + " " + trials["side"]
        trials["split_group"] = target
    elif arms == "together":
        trials["unit"] = trials["session_id"]
        trials["split_group"] = target + " " + trials["side"]
    else:
        raise ValueError(f'arms must be "separate" or "together", not {arms!r}')
    return trials


def splittable(trials):
    """True for the trials that can be split: their group (unit x split_group) has at least 2 trials.
    A target done only once by a unit cannot be in both halves, so it is left out."""
    return trials.groupby(["task_id", "unit", "split_group"])["trial_id"].transform("size") >= 2


def units_table(trials, min_reps):
    """One row per task x unit: its arm(s), number of trials, number that can be split, and kept = at least
    min_reps trials that can be split. Units with fewer are left out for that task (each half needs a few trials)."""
    trials = trials.assign(splittable=splittable(trials))
    units = (trials.groupby(["task_id", "task_name", "unit"])
             .agg(session_id=("session_id", "first"), person=("person", "first"), condition=("condition", "first"),
                  arms=("side", lambda sides: ", ".join(sorted(sides.unique()))),
                  n_trials=("trial_id", "size"), n_splittable=("splittable", "sum"))
             .reset_index())
    units["kept"] = units["n_splittable"] >= min_reps
    return units


def units_per_task(units):
    """Per task (and arm): number of units, how many are kept, how many were left out by MIN_REPS."""
    return (units.groupby(["task_id", "task_name", "arms"])
            .agg(units=("unit", "size"), kept=("kept", "sum"), left_out_by_min_reps=("kept", lambda kept: (~kept).sum()))
            .unstack("arms", fill_value=0))


# ---- 2. Split-half reliability ----

def half_labels(task, seed):
    """One random split of a task's trials: "A", "B" or "" (left out) for each trial.
    Within each split group (one unit, one target, see add_units): the trials in a random order, the first half -> A,
    the second half -> B; with an odd number, the last one is left out, so both halves have the same size.
    The random order comes from ONE permutation per fold (numpy default_rng(seed)), which every group uses for its
    own trials: with the same number of trials everywhere, this is exactly the split of compute_ceiling_splithalf."""
    groups = [task["unit"], task["split_group"]]
    position = task.groupby(groups).cumcount().to_numpy()                 # 0, 1, 2 ... within each group
    size = task.groupby(groups)["trial_id"].transform("size").to_numpy()
    permutation = np.random.default_rng(seed).permutation(size.max())
    # Where each trial's position comes in the permutation; the order of these within a group = its random order.
    place = pd.Series(np.argsort(permutation)[position], index=task.index)
    rank = place.groupby(groups).rank(method="first").to_numpy() - 1      # 0 ... size - 1
    half = size // 2
    return pd.Series(np.select([rank < half, rank < 2 * half], ["A", "B"], ""), index=task.index)


def correlate(a, b, method):
    """Correlation between 2 tables, column by column, across their rows (here: across units).
    "pearson" = linear; "spearman" = Pearson on the ranks (only the order of the units counts, so one extreme
    unit weighs less)."""
    if method == "spearman":
        a, b = a.rank(), b.rank()
    elif method != "pearson":
        raise ValueError(f'correlation must be "pearson" or "spearman", not {method!r}')
    a, b = a - a.mean(), b - b.mean()
    spread = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return ((a * b).sum() / spread).where(spread > 0)


def fold_reliability(task, labels, measures, correlation, spearman_brown):
    """The reliability of each number for one split: average of half A and of half B per unit, correlation of the
    2 halves across units, then the Spearman-Brown correction r_sb = 2r / (1 + r) (each half has only half of the
    trials; the corrected value is the reliability of the average of all of them)."""
    half_a = task[labels == "A"].groupby("unit")[measures].mean()
    half_b = task[labels == "B"].groupby("unit")[measures].mean().reindex(half_a.index)
    r = correlate(half_a, half_b, correlation)
    return 2 * r / (1 + r) if spearman_brown else r


def task_reliability(task, measures, settings):
    """The reliability of each number in one task, for each of the n_splits random splits (one row per split).
    Split k uses the seed seed + k (as in compute_ceiling_splithalf)."""
    task = task.sort_values(["unit", "split_group", "trial_id"]).reset_index(drop=True)
    return pd.DataFrame([fold_reliability(task, half_labels(task, settings["seed"] + fold), measures,
                                          settings["correlation"], settings["spearman_brown"])
                         for fold in range(settings["n_splits"])])


def split_half_reliability(trials, units, measures, settings):
    """One row per task x number: the mean reliability over the random splits, the 5th and 95th percentiles over
    the splits (how much it moves from one split to another), the number of units used, and the median number of
    trials per unit. Only the kept units (units_table) and the trials that can be split are used."""
    missing = [m for m in measures if m not in trials.columns]
    if missing:
        raise ValueError(f"Not in the table: {missing} (PCA scores: is N_KEEP of step 4 at least N_PCS?)")
    kept = units.loc[units["kept"], ["task_id", "unit"]]
    trials = trials[splittable(trials)].merge(kept, on=["task_id", "unit"])

    rows = []
    for (task_id, task_name), task in trials.groupby(["task_id", "task_name"]):
        n_units = task["unit"].nunique()
        folds = task_reliability(task, measures, settings) if n_units >= MIN_UNITS else pd.DataFrame(columns=measures)
        for measure in measures:
            values = folds[measure].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            rows.append({"task_id": task_id, "task_name": task_name, "measure": measure,
                         "reliability": values.mean() if len(values) else np.nan,
                         "reliability_p05": np.percentile(values, 5) if len(values) else np.nan,
                         "reliability_p95": np.percentile(values, 95) if len(values) else np.nan,
                         "n_units": n_units, "median_trials_per_unit": task.groupby("unit").size().median()})
    return pd.DataFrame(rows)


# ---- 3. Heatmap, 4. summary ----

def plot_heatmap(table, measures, threshold, title):
    """Rows = numbers, columns = tasks, colour and value = reliability (light = 0 or below, dark = 1).
    Cells at or above threshold get a frame, drawn a bit inside the cell (so neighbouring frames stay separate boxes)
    and in the same ink as the value (white on dark cells, black on light ones, so it is always visible)."""
    matrix = table.pivot(index="measure", columns="task_id", values="reliability").reindex(measures)
    names = table.drop_duplicates("task_id").set_index("task_id")["task_name"]
    fig, ax = plt.subplots(figsize=(1.1 * matrix.shape[1] + 3, 0.45 * matrix.shape[0] + 2.2))
    colours = LinearSegmentedColormap.from_list("reliability", BLUE_RAMP)
    image = ax.imshow(matrix.clip(lower=0).to_numpy(dtype=float), cmap=colours, vmin=0, vmax=1, aspect="auto")

    for (i, j), value in np.ndenumerate(matrix.to_numpy(dtype=float)):
        if np.isnan(value):
            ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=INK_MUTED)
            continue
        ink = "white" if value >= 0.5 else INK   # readable on the cell colour
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color=ink)
        if value >= threshold:
            ax.add_patch(Rectangle((j - 0.42, i - 0.36), 0.84, 0.72, fill=False, edgecolor=ink, linewidth=1.5))

    ax.set_xticks(range(matrix.shape[1]), [f"{t}. {names[t]}" for t in matrix.columns], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(matrix.shape[0]), [m.removesuffix("_s").replace("_", " ") for m in matrix.index], fontsize=9)
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("reliability (0 = random from one repetition to the next, 1 = stable)", color=INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    ax.set_title(f"{title}\nframed: reliability ≥ {threshold}", fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


def reliable_summary(table, threshold):
    """Per task: how many PCs have a reliability at or above threshold, which ones, and which extra measures."""
    reliable = table[table["reliability"] >= threshold]
    is_pc = reliable["measure"].str.fullmatch(r"PC\d+")
    by_task = table.drop_duplicates("task_id").set_index("task_id")[["task_name"]]
    by_task["n_reliable_PCs"] = reliable[is_pc].groupby("task_id").size()
    by_task["reliable_PCs"] = reliable[is_pc].groupby("task_id")["measure"].agg(", ".join)
    by_task["reliable_other"] = reliable[~is_pc].groupby("task_id")["measure"].agg(", ".join)
    return by_task.fillna({"n_reliable_PCs": 0, "reliable_PCs": "-", "reliable_other": "-"}).astype({"n_reliable_PCs": int})


# ---- 5. The patients' two visits (just to look) ----

def visit_medians(trials, measure):
    """Median of one number per task, person, arm and visit, for the persons who have both visit 1 and visit 2."""
    visits = trials.dropna(subset=["visit"]).groupby("person")["visit"].agg(set)
    persons = [person for person, done in visits.items() if {1, 2} <= done]
    two_visits = trials[trials["person"].isin(persons)]
    return (two_visits.groupby(["task_id", "task_name", "person", "condition", "side", "visit"])[measure]
            .median().reset_index())


def plot_visits(medians, measure, styles):
    """One panel per task: x = visit (1, 2), y = median of `measure`, one line per person and arm
    (colour = condition and marker = person, as on the PCA maps; solid = right arm, dashed = left arm).
    Prints a message and draws nothing if no person has 2 visits (session_participant.csv not filled in yet)."""
    if medians.empty:
        print("No person has both visit 1 and visit 2 yet (fill in the visit column of session_participant.csv).")
        return
    tasks = medians.drop_duplicates("task_id").sort_values("task_id")
    n_columns = 4
    n_rows = int(np.ceil(len(tasks) / n_columns))
    fig, panels = plt.subplots(n_rows, n_columns, figsize=(3.6 * n_columns, 3.2 * n_rows), squeeze=False)
    for panel, (_, task) in zip(panels.ravel(), tasks.iterrows()):
        for (person, side), line in medians[medians["task_id"] == task["task_id"]].groupby(["person", "side"]):
            colour, marker, _ = styles.get(person, (INK, "o", person))
            panel.plot(line["visit"].astype(int), line[measure], color=colour, marker=marker, markersize=8,
                       linewidth=2, linestyle=ARM_STYLES.get(side, "-"), markeredgecolor="white")
        panel.set_title(f"{task['task_id']}. {task['task_name']}", fontsize=9, color=INK)
        panel.set_xticks([1, 2], ["visit 1", "visit 2"])
        panel.set_xlim(0.7, 2.3)
        panel.set_ylabel(f"median {measure}", fontsize=9, color=INK_SECONDARY)
        panel.grid(color=GRID, linewidth=0.8)
        panel.set_axisbelow(True)
        panel.spines[["top", "right"]].set_visible(False)
    for panel in panels.ravel()[len(tasks):]:
        panel.set_visible(False)

    persons = medians.drop_duplicates("person")["person"]
    handles = [Line2D([], [], color=styles.get(p, (INK, "o", p))[0], marker=styles.get(p, (INK, "o", p))[1],
                      linewidth=2, markersize=8, label=styles.get(p, (INK, "o", p))[2]) for p in persons]
    handles += [Line2D([], [], color=INK_SECONDARY, linestyle=style, linewidth=2, label=f"{side} arm")
                for side, style in ARM_STYLES.items()]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 0.95))
    fig.suptitle(f"The patients' two visits: median {measure} per visit (just to look, no statistics)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    plt.show()


# ---- Save ----

def save_reliability(table, settings, processed_root=PROCESSED_ROOT):
    """Writes reliability.parquet and the settings used (reliability_settings.json)."""
    table.to_parquet(processed_root / "reliability.parquet", index=False)
    with open(processed_root / "reliability_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    for name in ["reliability.parquet", "reliability_settings.json"]:
        print(f"{name:<28} {(processed_root / name).stat().st_size / 1e6:7.2f} MB")
