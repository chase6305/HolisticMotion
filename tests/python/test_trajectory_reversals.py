"""A blending request must preserve reversals as stopped waypoints."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("dof", [1, 2, 7])
@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("time_scale", [1.0, 1.7])
def test_blended_reversal_stops_at_exact_waypoint(dof, profile, time_scale):
    points = np.zeros((4, dof))
    points[:, 0] = [0.0, 1.0, 0.4, -0.2]
    trajectory = hm.RnTrajectory(
        points,
        np.ones(dof),
        np.ones(dof),
        np.ones(dof),
        blend_tolerance=0.01,
        profile=profile,
    )
    trajectory.set_minimum_duration(trajectory.duration * time_scale)
    q, dq, ddq, _ = trajectory.sample(trajectory.breakpoints)
    indices = np.flatnonzero(np.abs(q[:, 0] - 1.0) < 1e-12)
    assert len(indices) > 0
    np.testing.assert_allclose(dq[indices], 0.0, atol=1e-12)
    if profile == "double_s":
        np.testing.assert_allclose(ddq[indices], 0.0, atol=1e-12)
    _, positions, *_ = trajectory.sample_uniform(2001)
    assert np.min(positions[:, 0]) >= -0.2 - 1e-12
    assert np.max(positions[:, 0]) <= 1.0 + 1e-12
    np.testing.assert_allclose(positions[[0, -1]], points[[0, -1]], atol=1e-12)
    assert trajectory.constraint_report()["within_limits"]


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
def test_near_reversal_with_unresolvable_blend_stops_at_waypoint(profile):
    points = np.array([[0.0, 0.0], [0.001, 0.0], [0.0, 0.00001]])
    trajectory = hm.RnTrajectory(
        points,
        np.ones(2),
        np.ones(2),
        np.ones(2),
        blend_tolerance=0.00003,
        profile=profile,
    )
    q, dq, ddq, _ = trajectory.sample(trajectory.breakpoints)
    at_corner = np.linalg.norm(q - points[1], axis=1) < 1e-12
    assert np.any(at_corner)
    np.testing.assert_allclose(dq[at_corner], 0.0, atol=1e-12)
    if profile == "double_s":
        np.testing.assert_allclose(ddq[at_corner], 0.0, atol=1e-12)
    np.testing.assert_allclose(q[[0, -1]], points[[0, -1]], atol=1e-12)
    assert trajectory.constraint_report()["within_limits"]
