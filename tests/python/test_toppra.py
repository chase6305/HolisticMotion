import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory, retime_path


def test_toppra_retimes_multijoint_path_with_zero_boundaries():
    trajectory = retime_path(
        [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
        [1.0, 0.8],
        [2.0, 1.5],
        grid_size=120,
    )
    times, position, velocity, acceleration = trajectory.sample_uniform(300)
    assert trajectory.duration > 0.0
    np.testing.assert_allclose(position[0], [0.0, 0.0])
    np.testing.assert_allclose(position[-1], [1.0, 0.5])
    np.testing.assert_allclose(velocity[[0, -1]], 0.0, atol=1e-9)
    assert np.max(np.abs(velocity), axis=0)[0] <= 1.0 + 1e-6
    assert np.max(np.abs(velocity), axis=0)[1] <= 0.8 + 1e-6
    assert np.all(np.max(np.abs(acceleration), axis=0) <= [2.0 + 1e-6, 1.5 + 1e-6])
    assert np.all(np.diff(times) > 0.0)


def test_toppra_validates_inputs_and_boundary_velocity():
    with pytest.raises(ValueError, match="distinct"):
        ToppraTrajectory([[0.0], [0.0]], [1.0], [1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        ToppraTrajectory([[0.0], [1.0]], [0.0], [1.0])
    with pytest.raises(ValueError, match="boundary"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], start_path_velocity=2.0)


def test_toppra_clamps_sample_times():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0])
    position, _, _ = trajectory.sample([-1.0, trajectory.duration + 1.0])
    np.testing.assert_allclose(position[:, 0], [0.0, 1.0])
