"""Filtering near-duplicate interior points must preserve requested endpoints."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("blend", [0.0, 1e-4])
@pytest.mark.parametrize("last", [0.2 + 2e-6, 0.2 - 2e-6, 0.2])
def test_near_duplicate_final_waypoint_is_retained(profile, blend, last):
    points = np.array([[0.0, 0.0], [0.1, 0.02], [0.2, 0.0], [last, 0.0]])
    trajectory = hm.RnTrajectory(
        points,
        np.ones(2),
        np.ones(2),
        np.ones(2),
        blend_tolerance=blend,
        profile=profile,
    )
    np.testing.assert_array_equal(trajectory.waypoints[[0, -1]], points[[0, -1]])
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), points[-1], rtol=0.0, atol=2e-14
    )
    for time in [trajectory.duration, 1.5 * trajectory.duration]:
        np.testing.assert_allclose(trajectory.velocity(time), 0.0, atol=1e-10)
    assert np.all(np.diff(trajectory.breakpoints) > 0.0)


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
def test_terminal_near_duplicate_does_not_erase_a_short_excursion(profile):
    points = [[0.0], [1.5e-5], [0.9e-5]]
    trajectory = hm.RnTrajectory(points, [1.0], [1.0], [1.0], profile=profile)
    np.testing.assert_allclose(trajectory.waypoints, points, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), points[-1], rtol=0.0, atol=1e-14
    )
    positions = trajectory.sample(trajectory.breakpoints)[0]
    assert positions.max() == pytest.approx(1.5e-5, abs=1e-14)


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("dof", [1, 7])
@pytest.mark.parametrize("displacement", [-2e-6, 2e-6])
def test_short_two_point_motion_is_distinct(profile, dof, displacement):
    points = np.zeros((2, dof))
    points[1, 0] = displacement
    limits = np.ones(dof)
    trajectory = hm.RnTrajectory(points, limits, limits, limits, profile=profile)
    q, dq, ddq, dddq = trajectory.sample(np.linspace(0.0, trajectory.duration, 101))
    assert trajectory.duration > 0.0
    np.testing.assert_allclose(q[[0, -1]], points, rtol=0.0, atol=1e-15)
    assert trajectory.path_length == pytest.approx(abs(displacement), abs=0.0)
    for derivative in [dq, ddq, dddq]:
        assert np.max(np.abs(derivative)) <= 1.0 + 1e-10
