"""Paths and settings of the project. Change them here, nowhere else."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ---- Paths ----
# Raw recordings (read only, never modified).
RAW_ROOT = Path("/Users/louistschanz/Documents/EPFL/Champalimaud/immersive-kinematics/data/upper_limb_recorded_data/"
                "upper_limb_asessment/recordings")
# Tables written by step 1 (sessions, trials, frames).
PROCESSED_ROOT = REPO_ROOT / "data/processed"
# Tables of the old repo (only to compare with the old code).
OLD_PROCESSED_ROOT = Path("/Users/louistschanz/Documents/EPFL/Champalimaud/immersive-kinematics/data/processed")
# Filled in by hand: who is who (written once as templates, never overwritten by the code).
PARTICIPANTS_CSV = REPO_ROOT / "data/participants.csv"
SESSION_PARTICIPANT_CSV = REPO_ROOT / "data/session_participant.csv"

# ---- Which sessions are usable ----
MIN_TRIALS_ONE_TASK = 4    # at least this many repetitions of one task with one arm (quick recording tests have 1-5)
MAX_HEAD_HEIGHT_M = 1.42   # head higher than this = the person stood (staff testing the setup); seated ~1.1-1.3 m

# ---- Trial quality flags ----
MIN_FRAMES_PER_TRIAL = 10
MAX_WRONG_PERSON_PCT = 10          # max % of frames where the camera followed someone else
MAX_DISTANCE_MAIN_PERSON_M = 0.30  # a frame is the main person if the shoulders are this close to their usual place

# ---- Step 2: kinematics ----
MAX_FACING_DEG = 45   # body turned by more than this -> facing_ok = False (flag only)

# ---- setup_version: the meta.json settings that define a version of the setup ----
# Only settings that change how the tasks run or how the body is tracked. Session-specific values (times, IDs,
# file paths) and the infrastructure (servers, audio, projector) are left out.
SETUP_KEYS = [
    # how a trial ends (automatic when the arm is back, or by the operator)
    "automatic_trial_end_initial", "manual_trial_end",
    # task thresholds and targets
    "start_return_shoulder_tolerance_m", "pointing_ray_mode", "shoulder_ray_pitch_offset_deg",
    "task4_reach_length_m", "task4_forward_reach_m", "task4_diagonal_reach_m",
    "task4_table_height_m", "task4_table_edge_forward_offset_m",
    "task6_start_elbow_min_deg", "task6_start_elbow_max_deg",
    "task8_start_elbow_min_deg", "task8_start_elbow_max_deg", "task8_start_wrist_below_shoulder_m",
    "mouth_reach_radius_m", "nose_reach_radius_m",
    "nose_wrist_xz_radius_m", "nose_wrist_y_offset_m", "nose_wrist_y_tolerance_m",
    "task_table_absolute_a_x_m", "task_table_absolute_a_z_m",
    "table_left_wrist_offset_x_m", "table_left_wrist_offset_z_m",
    "table_right_wrist_offset_x_m", "table_right_wrist_offset_z_m",
    "table_wrist_yaw_variant_task_ids", "table_wrist_yaw_variants_deg",
    # body tracking
    "pose_frame_source_mode", "pose_triangulation_profile", "pose_y_correction_m", "max_pose_age_s",
]
