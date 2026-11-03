# kinematic_room

Upper-limb movement recordings from the Champalimaud Foundation (Krakauer lab). Patients (Parkinson's, stroke,
cerebellar ataxias, ...) and healthy participants do 8 arm tasks in a room where a markerless camera system tracks
15 body points in 3D.

**How the data is organised:** patient → session (`session_id`, normally one visit) → task (one of the 8 exercises,
done with each arm) → trial (one repetition) → frame (one snapshot of the body points).

## Folders
- **`exploration/`: start here.**
  - `DBB.ipynb`, `MAM.ipynb`, `JFR.ipynb`: the raw data of each patient with two visits, with no processing.
  - `01_patient_visits.ipynb`: first visit (pre) against second visit (post).
- **`pipeline/`**: the analysis pipeline: notebooks `01` to `05`, their code, the settings and paths (`config.py`), and
  the tests.
- **`data/`**: never committed (patient identifiers).
  - `upper_limb_recorded_data/`: a copy of the raw recordings.
  - `patients_raw/<patient>/visit<1 or 2>/`: the raw files of DBB, MAM and JFR, to check things by hand.
  - `session_participant.csv`, `participants.csv`: who each session is, filled in by hand.
  - `processed/`: the tables written by the pipeline.

## Running
Python 3.10, with pandas, numpy, pyarrow, scipy, scikit-learn, matplotlib, jupyter and pytest.
- Notebooks: open them and run all cells.
- Tests: `cd pipeline`, then `python -m pytest tests`.
