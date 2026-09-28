"""A short final leg must propagate a reduced entry speed to earlier phases."""

import holistic_motion as hm
import numpy as np
import pytest


def test_short_final_deceleration_preserves_velocity_continuity():
    inputs = {
        "waypoints": [
            [-0.9354447474771082, -2.4134574688379695],
            [-0.5241088043619658, -3.268812656509193],
            [-0.3579563815953131, -3.6467652794396783],
        ],
        "max_velocity": [0.5498234556472988, 1.902897427993734],
        "max_acceleration": [9.847619614681955, 1.4945934331795436],
        "max_jerk": [7.53225475425155, 2.72928577422683],
        "blend_tolerance": 0.03926977684513886,
        "profile": "double_s",
    }
    trajectory = hm.RnTrajectory(**inputs)
    report = trajectory.constraint_report(10001)
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"]
    assert report["within_limits"]
    points = np.asarray(inputs["waypoints"])
    q = trajectory.sample([0.0, trajectory.duration])[0]
    np.testing.assert_allclose(q, points[[0, -1]], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("length", [1000.0, 10000.0])
@pytest.mark.parametrize("jerk", [1e4, 1e5, 1e6])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_short_jerk_ramps_after_long_cruise(length, jerk, direction):
    trajectory = hm.RnTrajectory(
        [[0.0], [direction * length]],
        [0.1],
        [1.0],
        [jerk],
        profile="double_s",
        blend_tolerance=0.005,
    )
    for slowdown in [1.0, 1.7]:
        if slowdown != 1.0:
            trajectory.set_minimum_duration(slowdown * trajectory.duration)
        report = trajectory.constraint_report(2)
        assert report["within_limits"]
        assert report["velocity_continuous"]
        assert report["acceleration_continuous"]
        assert np.max(report["maximum_acceleration_jump"]) < 1e-12
        assert np.max(report["maximum_velocity_jump"]) < 1e-12
        q, v, a, _ = trajectory.sample([0.0, trajectory.duration])
        np.testing.assert_allclose(q[:, 0], [0.0, direction * length], atol=1e-10, rtol=0)
        np.testing.assert_allclose(v, 0.0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a, 0.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("length", [1000.0, 10000.0])
@pytest.mark.parametrize("acceleration", [1e4, 1e5, 1e6])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_short_acceleration_ramps_after_long_cruise(length, acceleration, direction):
    trajectory = hm.RnTrajectory(
        [[0.0], [direction * length]],
        [0.1],
        [acceleration],
        [1.0],
        profile="trapezoidal",
        blend_tolerance=0.005,
    )
    for slowdown in [1.0, 1.7]:
        if slowdown != 1.0:
            trajectory.set_minimum_duration(slowdown * trajectory.duration)
        report = trajectory.constraint_report(2)
        assert report["within_limits"]
        assert report["velocity_continuous"]
        assert np.max(report["maximum_velocity_jump"]) < 1e-12
        q, v, _, _ = trajectory.sample([0.0, trajectory.duration])
        np.testing.assert_allclose(q[:, 0], [0.0, direction * length], atol=1e-10, rtol=0)
        np.testing.assert_allclose(v, 0.0, atol=1e-12, rtol=0)
