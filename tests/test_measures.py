"""Tests of measures.py on fake data. Run from the repo folder: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the code is in the repo folder (not installed)
import measures  # noqa: E402


def arm(elbow, wrist):
    """One moment: elbow and wrist positions (forward, out, up) from the shoulder."""
    return pd.DataFrame([dict(zip(measures.ELBOW + measures.WRIST, list(elbow) + list(wrist)))])


def test_angles():
    straight_forward = measures.joint_angles(arm((0.3, 0, 0), (0.6, 0, 0))).iloc[0]
    assert straight_forward["elbow_angle_deg"] == pytest.approx(180)
    assert straight_forward["shoulder_elevation_deg"] == pytest.approx(90)    # horizontal
    hanging = measures.joint_angles(arm((0, 0, -0.3), (0, 0, -0.6))).iloc[0]
    assert hanging["shoulder_elevation_deg"] == pytest.approx(0)
    straight_up = measures.joint_angles(arm((0, 0, 0.3), (0, 0, 0.6))).iloc[0]
    assert straight_up["shoulder_elevation_deg"] == pytest.approx(180)
    bent = measures.joint_angles(arm((0, 0, -0.3), (0.25, 0, -0.3))).iloc[0]   # forearm forward, upper arm down
    assert bent["elbow_angle_deg"] == pytest.approx(90)


def test_constant_speed_straight_line_has_no_jerk():
    times = np.arange(60) / 30
    positions = np.outer(times, [0.2, 0.1, 0.3])   # same direction, same speed all the time
    assert measures.mean_squared_jerk(times, positions) == pytest.approx(0, abs=1e-20)


def minimum_jerk(duration_s, size_m, rate_hz=200):
    """A minimum-jerk movement (the smoothest point-to-point movement) along one 3D direction."""
    times = np.arange(0, duration_s + 1e-9, 1 / rate_hz)
    s = times / duration_s
    distance = size_m * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    return times, np.outer(distance, np.array([0.6, 0.0, 0.8]))


def test_ldlj_does_not_depend_on_duration_or_size():
    # Sampled finely (1000 per second), so the numerical derivatives are accurate: the formula itself does not depend
    # on duration or size. (At camera rates, 12-38 per second, the numerical error grows: fewer frames per movement
    # = closer to 0. That is the frame-rate check of the notebook.)
    short_small = measures.ldlj(*minimum_jerk(0.8, 0.15, rate_hz=1000))
    long_big = measures.ldlj(*minimum_jerk(2.5, 0.60, rate_hz=1000))
    assert short_small == pytest.approx(long_big, abs=0.01)
    assert short_small == pytest.approx(-np.log(720 / 1.875 ** 2), abs=0.01)   # exact value for a minimum-jerk movement
    # a movement with a speed bump is less smooth (further from 0)
    times, positions = minimum_jerk(1.0, 0.3, rate_hz=1000)
    bumpy = positions + 0.01 * np.sin(2 * np.pi * 5 * times)[:, None]
    assert measures.ldlj(times, bumpy) < short_small


def test_coupling_is_minus_one_when_the_elbow_closes_as_the_arm_goes_up():
    elevation = np.linspace(20, 120, 50)
    trial = pd.DataFrame({"shoulder_elevation_deg": elevation, "elbow_angle_deg": 180 - elevation})
    assert measures.angle_measures(trial)["coupling_r"] == pytest.approx(-1)


def test_arm_label():
    trials = pd.DataFrame({"side": ["left", "right", "left", "left"],
                           "affected_side": ["left", "left", None, "left"],
                           "lesion_side": [None, None, None, None],
                           "condition": ["Stroke", "Stroke", "Stroke", "healthy"]})
    assert measures.arm_labels(trials).tolist() == ["affected", "less affected", "left", "left"]


def test_arm_label_from_the_lesion_side():
    # lesion in the right hemisphere: the left arm is contralesional, the right arm ipsilesional (whatever affected_side)
    trials = pd.DataFrame({"side": ["left", "right", "left"], "affected_side": ["left", "left", None],
                           "lesion_side": ["Right ", "right", "right"], "condition": ["Stroke", "Stroke", "healthy"]})
    assert measures.arm_labels(trials).tolist() == ["contralesional", "ipsilesional", "left"]
