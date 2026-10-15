"""Large finite waypoints must retain all requested path segments."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("dof", [2, 7])
@pytest.mark.parametrize("scale", [1e155, 1e200, 1e300])
@pytest.mark.parametrize("blend", [0.0, 0.01])
def test_large_waypoints_retain_initial_and_final_positions(dof, scale, blend):
    points = np.zeros((4, dof))
    points[:, :2] = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]) * scale
    trajectory = hm.RnTrajectory(
        points,
        np.full(dof, np.sqrt(scale)),
        np.ones(dof),
        np.ones(dof),
        blend_tolerance=blend * scale,
        profile="trapezoidal",
    )
    assert 2.9 < trajectory.path_length / scale <= 3.0
    assert np.isfinite(trajectory.duration) and trajectory.duration > 0.0
    np.testing.assert_allclose(
        trajectory.position(0.0) / scale, points[0] / scale, atol=0.0
    )
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration) / scale,
        points[-1] / scale,
        rtol=1e-13,
        atol=0.0,
    )
    samples = trajectory.sample(np.linspace(0.0, trajectory.duration, 101))
    assert all(np.isfinite(value).all() for value in samples)
    np.testing.assert_allclose(samples[0][0] / scale, points[0] / scale, atol=0.0)
    np.testing.assert_allclose(samples[0][-1] / scale, points[-1] / scale, rtol=1e-13)


@pytest.mark.parametrize("dof", [2, 7])
@pytest.mark.parametrize("blend", [0.0, 0.01])
def test_large_absolute_coordinate_norm_keeps_finite_motion(dof, blend):
    points = np.full((4, dof), 1.3e308)
    points[:, :2] += np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]) * 1e306
    trajectory = hm.RnTrajectory(
        points,
        np.full(dof, 1e153),
        np.ones(dof),
        np.ones(dof),
        blend_tolerance=blend * 1e306,
        profile="trapezoidal",
    )
    assert 2.9 < trajectory.path_length / 1e306 <= 3.01
    np.testing.assert_array_equal(trajectory.position(0.0), points[0])
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration) / 1e308,
        points[-1] / 1e308,
        rtol=1e-13,
    )
