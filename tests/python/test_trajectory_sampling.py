"""Batch sampling preserves states and rejects unrepresentable array shapes."""

import numpy as np
import pytest


def _trajectory(kind, profile="double_s"):
    import holistic_motion as hm

    if kind == "cartesian":
        end = np.eye(4)
        angle = 0.4
        end[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]
        end[:3, 3] = [0.2, -0.1, 0.3]
        limits = np.ones(6)
        return hm.CartesianLineTrajectory(
            np.eye(4), end, limits, 2 * limits, 5 * limits, profile=profile
        )
    dof = int(kind)
    points = np.zeros((2, dof))
    points[1] = np.linspace(0.1, 0.3, dof)
    limits = np.ones(dof)
    return hm.RnTrajectory(points, limits, limits, limits, profile=profile)


@pytest.mark.parametrize("kind", ["cartesian", "1", "7", "32"])
def test_uniform_sampling_rejects_unrepresentable_arrays(kind):
    trajectory = _trajectory(kind)
    components = 16 if kind == "cartesian" else int(kind)
    maximum_index = np.iinfo(np.intp).max
    for samples in (
        maximum_index // (np.dtype(float).itemsize * components) + 1,
        maximum_index + 1,
        np.iinfo(np.uintp).max,
    ):
        with pytest.raises(ValueError, match="sample count is too large"):
            trajectory.sample_uniform(int(samples))


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("samples", [2, 31, 257])
def test_cartesian_uniform_sampling_matches_complete_states(profile, samples):
    trajectory = _trajectory("cartesian", profile)
    times, poses, velocity, acceleration, jerk = trajectory.sample_uniform(samples)
    assert poses.shape == (samples, 4, 4)
    np.testing.assert_allclose(
        times, np.linspace(0, trajectory.duration, samples),
        rtol=2 * np.finfo(float).eps, atol=0,
    )
    for index, time in enumerate(times):
        for actual, expected in zip(
            (poses[index], velocity[index], acceleration[index], jerk[index]),
            trajectory.state(time),
        ):
            np.testing.assert_array_equal(actual, expected)
    for count in (0, 1):
        with pytest.raises(ValueError, match="at least 2 samples"):
            trajectory.sample_uniform(count)


@pytest.mark.parametrize("kind", ["cartesian", "1", "7", "32"])
def test_sample_buffers_are_independent_writable_and_outlive_trajectory(kind):
    import gc

    trajectory = _trajectory(kind)
    first = trajectory.sample_uniform(31)
    second = trajectory.sample_uniform(31)
    expected = [array.copy() for array in second]
    for index, array in enumerate(first):
        assert array.flags.writeable
        assert array.flags.owndata
        if array.ndim == 2:
            assert array.flags.f_contiguous
        assert not np.shares_memory(array, second[index])
        array.flat[0] += 1
    del first, trajectory
    gc.collect()
    for actual, reference in zip(second, expected):
        np.testing.assert_array_equal(actual, reference)


def test_empty_joint_samples_keep_shape_and_array_ownership():
    trajectory = _trajectory("7")
    values = trajectory.sample(np.array([]))
    del trajectory
    for value in values:
        assert value.shape == (0, 7)
        assert value.flags.writeable
        assert value.flags.owndata
