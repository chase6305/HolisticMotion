"""Large coordinate units must keep curve-cap sampling bounded."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("profile", ["trapezoidal", "double_s"])
@pytest.mark.parametrize("scale", [1.0, 1e4, 1e12, 1e100])
def test_large_coordinate_blended_trajectory(profile, scale):
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]) * scale
    limits = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]) * scale
    trajectory = hm.RnTrajectory(
        waypoints=points,
        max_velocity=limits[0],
        max_acceleration=limits[1],
        max_jerk=limits[2],
        blend_tolerance=0.1 * scale,
        profile=profile,
    )
    reference = hm.RnTrajectory(
        waypoints=points / scale,
        max_velocity=limits[0] / scale,
        max_acceleration=limits[1] / scale,
        max_jerk=limits[2] / scale,
        blend_tolerance=0.1,
        profile=profile,
    )
    assert trajectory.duration == pytest.approx(reference.duration, rel=1e-10)
    times = np.unique(
        np.concatenate(
            [
                np.linspace(start, end, 257)
                for start, end in zip(
                    trajectory.breakpoints[:-1], trajectory.breakpoints[1:]
                )
            ]
        )
    )
    states = trajectory.sample(times)
    assert all(np.isfinite(state).all() for state in states)
    np.testing.assert_allclose(states[0][[0, -1]] / scale, points[[0, -1]] / scale)
    for values, limit in zip(states[1:], limits):
        assert np.max(np.abs(values) / limit) <= 1.0 + 1e-8
    report = trajectory.constraint_report(10001)
    assert report["within_limits"]
    assert report["velocity_continuous"]
    if profile == "double_s":
        assert report["acceleration_continuous"]
