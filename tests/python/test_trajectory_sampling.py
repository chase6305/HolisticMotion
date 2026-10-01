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


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("kind", ["cartesian", "1", "7", "32"])
def test_report_buffers_keep_ownership_and_outlive_trajectory(kind, profile):
    import gc

    trajectory = _trajectory(kind, profile)
    first = trajectory.constraint_report(65)
    second = trajectory.constraint_report(65)
    keys = [key for key, value in first.items() if isinstance(value, np.ndarray)]
    assert len(keys) == (3 if kind == "cartesian" else 8)
    dof = 6 if kind == "cartesian" else int(kind)
    expected = {key: second[key].copy() for key in keys}
    for index, key in enumerate(keys):
        array = first[key]
        assert array.shape == (dof,)
        assert array.dtype == np.dtype(np.float64)
        assert array.flags.writeable and array.flags.owndata
        assert array.flags.c_contiguous and array.flags.f_contiguous
        assert not np.shares_memory(array, second[key])
        for other_key in keys[index + 1:]:
            assert not np.shares_memory(array, first[other_key])
        array[0] += 1.0
        # Owning diagnostic buffers remain resizable independently of the report.
        array.resize((dof + 1,), refcheck=False)
    fresh = trajectory.constraint_report(65)
    for key in keys:
        np.testing.assert_array_equal(fresh[key], expected[key])
    views = {key: second[key][:] for key in keys}
    del first, second, fresh, trajectory
    gc.collect()
    for key, view in views.items():
        np.testing.assert_array_equal(view, expected[key])


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("kind", ["cartesian", "1", "3", "7", "32"])
def test_state_vector_buffers_are_independent_and_own_their_storage(kind, profile):
    import gc

    trajectory = _trajectory(kind, profile)
    time = trajectory.duration * 0.37
    state = trajectory.state(time)
    reference = [array.copy() for array in state]
    vectors = state[1:] if kind == "cartesian" else state
    for index, array in enumerate(vectors):
        assert array.dtype == np.dtype(np.float64)
        assert array.ndim == 1
        assert array.flags.owndata and array.flags.writeable
        assert array.flags.c_contiguous and array.flags.f_contiguous
        for other in vectors[index + 1:]:
            assert not np.shares_memory(array, other)
        array[:] = -123.0
        array.resize((array.size + 1,), refcheck=False)
    fresh = trajectory.state(time)
    views = [array[:] for array in fresh]
    del state, vectors, fresh, trajectory
    gc.collect()
    for actual, expected in zip(views, reference):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("kind", ["1", "3", "7", "32"])
@pytest.mark.parametrize("samples", [2, 3, 31, 257])
def test_joint_uniform_sampling_matches_its_returned_times(kind, profile, samples):
    trajectory = _trajectory(kind, profile)
    trajectory.set_minimum_duration(trajectory.duration * 2.3)
    times, *states = trajectory.sample_uniform(samples)
    assert times.flags.owndata and times.flags.writeable
    assert times[0] == 0.0 and times[-1] == trajectory.duration
    np.testing.assert_allclose(
        times, np.linspace(0.0, trajectory.duration, samples),
        rtol=2 * np.finfo(float).eps, atol=0.0,
    )
    for actual, expected in zip(states, trajectory.sample(times)):
        np.testing.assert_array_equal(actual, expected)
    # Changing the output time vector cannot modify any returned state buffer.
    expected = [array.copy() for array in states]
    times[:] = -1.0
    times.resize((samples + 1,), refcheck=False)
    for actual, reference in zip(states, expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("dof", [1, 2, 3, 7, 14, 32])
@pytest.mark.parametrize("blend", [0.0, 0.005])
def test_batch_sampling_preserves_scalar_states_at_shuffled_knots(profile, dof, blend):
    import holistic_motion as hm

    points = 0.3 * np.sin(
        0.4 * np.arange(8)[:, None] + 0.3 * np.arange(dof)[None, :]
    )
    limits = np.ones(dof)
    trajectory = hm.RnTrajectory(
        points, limits, limits, limits, blend_tolerance=blend, profile=profile,
    )
    rng = np.random.default_rng(20260929 + dof)
    for scale in (1.0, 2.3):
        trajectory.set_minimum_duration(trajectory.duration * scale)
        knots = trajectory.breakpoints
        times = np.concatenate((
            np.linspace(0.0, trajectory.duration, 65), knots,
            np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf),
            [-1e300, 1e300, 0.0, trajectory.duration],
        ))
        # Include non-monotone input, repeated knots and non-contiguous storage.
        rng.shuffle(times)
        times = times[::-1]
        actual = trajectory.sample(times)
        expected = [np.stack(values) for values in zip(
            *(trajectory.state(float(time)) for time in times)
        )]
        for array, reference in zip(actual, expected):
            np.testing.assert_array_equal(
                array.view(np.uint64), reference.view(np.uint64),
            )


@pytest.mark.parametrize("kind", ["1", "7", "32"])
def test_sample_layouts_survive_later_calls_with_different_shapes(kind):
    trajectory = _trajectory(kind)
    counts = (0, 1, 2, 33)
    batches = [
        trajectory.sample(np.linspace(0, trajectory.duration, count))
        for count in counts
    ]
    # Retain every result while subsequent calls reuse their stack descriptors.
    del trajectory
    for count, batch in zip(counts, batches):
        reference = np.empty((count, int(kind)), dtype=np.float64, order="F")
        for array in batch:
            assert array.shape == reference.shape
            assert array.strides == reference.strides
            assert array.dtype == reference.dtype
            assert array.base is None
            assert array.flags.owndata and array.flags.writeable
            assert array.flags.aligned
            assert array.flags.c_contiguous == reference.flags.c_contiguous
            assert array.flags.f_contiguous == reference.flags.f_contiguous


@pytest.mark.parametrize("kind", ["cartesian", "1", "7", "32"])
def test_uniform_sample_layouts_match_owned_numpy_arrays(kind):
    trajectory = _trajectory(kind)
    batches = [trajectory.sample_uniform(count) for count in (2, 33)]
    del trajectory
    for batch in batches:
        for array in batch:
            order = "F" if array.ndim == 2 else "C"
            reference = np.empty(array.shape, dtype=np.float64, order=order)
            assert array.strides == reference.strides
            assert array.dtype == reference.dtype
            assert array.base is None
            assert array.flags.owndata and array.flags.writeable
            assert array.flags.aligned
            assert array.flags.c_contiguous == reference.flags.c_contiguous
            assert array.flags.f_contiguous == reference.flags.f_contiguous
