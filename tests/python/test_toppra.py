import numpy as np
import pytest
from holistic_motion.trajectory import ToppraResult, ToppraTrajectory, retime_path


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
    with pytest.raises(ValueError, match="finite"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], start_path_velocity=float("nan"))
    with pytest.raises(TypeError, match="grid_size"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], grid_size=20.5)


def test_toppra_clamps_sample_times():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0])
    position, _, _ = trajectory.sample([-1.0, trajectory.duration + 1.0])
    np.testing.assert_allclose(position[:, 0], [0.0, 1.0])


def test_toppra_result_is_an_immutable_snapshot():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0])
    expected, _, _ = trajectory.sample_uniform(10)[1:]

    with pytest.raises(ValueError, match="read-only"):
        trajectory.result.path_speeds[0] = 10.0
    with pytest.raises(ValueError, match="read-only"):
        trajectory.waypoints[0, 0] = 10.0
    with pytest.raises(ValueError, match="read-only"):
        trajectory.max_velocity[0] = 10.0
    with pytest.raises(AttributeError):
        trajectory.result = trajectory.result

    actual, _, _ = trajectory.sample_uniform(10)[1:]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gridpoints": [0.0, 0.0]}, "gridpoints"),
        ({"path_speeds": [0.0, -1.0]}, "non-negative"),
        ({"times": [0.0, -1.0], "duration": -1.0}, "duration"),
        ({"duration": 2.0}, "final time"),
        ({"path_accelerations": [1.0]}, "path dynamics"),
    ],
)
def test_toppra_result_rejects_inconsistent_snapshots(kwargs, message):
    values = {
        "gridpoints": [0.0, 1.0],
        "path_speeds": [1.0, 1.0],
        "path_accelerations": [0.0],
        "times": [0.0, 1.0],
        "duration": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ToppraResult(**values)


def test_toppra_sampling_uses_constant_acceleration_interval_dynamics():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0], grid_size=20)
    interval = 0
    start_time = trajectory.result.times[interval]
    end_time = trajectory.result.times[interval + 1]
    elapsed = 0.5 * (end_time - start_time)
    expected_s = (
        trajectory.result.gridpoints[interval]
        + trajectory.result.path_speeds[interval] * elapsed
        + 0.5 * trajectory.result.path_accelerations[interval] * elapsed**2
    )

    position, velocity, _ = trajectory.sample([start_time + elapsed])

    np.testing.assert_allclose(position[0, 0], expected_s, atol=1e-12)
    np.testing.assert_allclose(
        velocity[0, 0],
        trajectory.result.path_speeds[interval]
        + trajectory.result.path_accelerations[interval] * elapsed,
        atol=1e-12,
    )
