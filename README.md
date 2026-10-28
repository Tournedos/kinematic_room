# kinematic_room

Upper-limb movement recordings from the Champalimaud Foundation (Krakauer lab). Patients (Parkinson's, stroke,
cerebellar ataxias, ...) and healthy participants do 8 arm tasks in a room where a markerless camera system tracks
15 body points in 3D.

Vocabulary: **session** = one recording · **task** = one of the 8 exercises · **trial** = one repetition ·
**frame** = one snapshot of the body points.

## Data (never committed)
`data/` holds patient identifiers and is ignored by git. Notebook outputs (patient codes, dates) are removed before
every commit by nbstripout.
- Raw recordings: read only, path `RAW_ROOT` in `upperlimb/config.py`.
- Filled in by hand: `data/session_participant.csv` (which participant each session is) and `data/participants.csv`
  (group, diagnosis, age, ...). Written once as templates by step 1, never overwritten.
- Outputs: `data/processed/`.

## Setup
```
conda env create -f environment.yml
conda activate upperlimb
pip install -e .
nbstripout --install      # once per clone: strips notebook outputs before each commit
```

## Step 1: preprocessing
Run `notebooks/01_preprocess.ipynb`. It builds 3 tables in `data/processed/`:
- `sessions.parquet`: one row per session (all sessions, with `is_usable`, `setup_version`, `participant_id`)
- `trials.parquet`: one row per trial of the usable sessions, with quality flags (`passes_checks`)
- `frames.parquet`: one row per frame inside a trial (x, y, z, conf of 14 body points)

Nothing is smoothed, filtered or moved: step 1 only organises and flags. Code: `upperlimb/`. Settings: `upperlimb/config.py`.

## Tests
`pytest` (runs on a tiny fake session).
