"""Native trajectory arrays must retain their real-valued contract."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "factory_name",
    ["JointTrajectory3", "JointTrajectory", "RnTrajectory", "BaseTrajectory"],
)
@pytest.mark.parametrize(
    "field", ["waypoints", "max_velocity", "max_acceleration", "max_jerk"]
)
@pytest.mark.parametrize("representation", ["array", "list", "object"])
def test_native_trajectory_rejects_complex_inputs(factory_name, field, representation):
    import holistic_motion as hm

    inputs = {
        "waypoints": np.array([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1]]),
        "max_velocity": np.ones(3),
        "max_acceleration": np.ones(3),
        "max_jerk": np.ones(3),
    }
    bad = inputs[field].astype(complex)
    bad.flat[-1] += 4j
    if representation == "list":
        bad = bad.tolist()
    elif representation == "object":
        bad = bad.astype(object)
    inputs[field] = bad
    with pytest.raises(TypeError, match="real"):
        getattr(hm, factory_name)(**inputs)


@pytest.mark.parametrize("representation", ["array", "list", "object"])
def test_native_trajectory_rejects_complex_sample_times(representation):
    import holistic_motion as hm

    trajectory = hm.RnTrajectory([[0.0], [0.2]], [1.0], [1.0], [1.0])
    times = np.array([0.0, trajectory.duration + 4j])
    if representation == "list":
        times = times.tolist()
    elif representation == "object":
        times = times.astype(object)
    with pytest.raises(TypeError, match="real"):
        trajectory.sample(times)
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), [0.2], atol=1e-7
    )


@pytest.mark.parametrize(
    "factory_name",
    ["JointTrajectory3", "JointTrajectory", "RnTrajectory", "BaseTrajectory"],
)
@pytest.mark.parametrize("representation", ["array", "list", "object"])
def test_native_trajectory_preserves_real_input_forms(factory_name, representation):
    import holistic_motion as hm

    waypoints = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1]])
    limits = np.ones(3)
    if representation == "list":
        inputs = (waypoints.tolist(), limits.tolist())
    elif representation == "object":
        inputs = (waypoints.astype(object), limits.astype(object))
    else:
        inputs = (waypoints, limits)
    trajectory = getattr(hm, factory_name)(inputs[0], inputs[1], inputs[1], inputs[1])
    times = np.array([0.0, trajectory.duration])
    if representation == "list":
        times = times.tolist()
    elif representation == "object":
        times = times.astype(object)
    positions, velocities, accelerations, jerks = trajectory.sample(times)
    np.testing.assert_allclose(positions, waypoints, atol=1e-7)
    assert all(np.isfinite(state).all() for state in [velocities, accelerations, jerks])


@pytest.mark.parametrize(
    "factory_name,dof",
    [
        ("JointTrajectory1", 1),
        ("JointTrajectory3", 3),
        ("JointTrajectory32", 32),
        ("JointTrajectory", 3),
        ("RnTrajectory", 3),
        ("BaseTrajectory", 3),
    ],
)
@pytest.mark.parametrize("later_input", ["max_acceleration", "minimum_duration"])
def test_joint_trajectory_snapshots_inputs_before_later_conversion(
    factory_name, dof, later_input
):
    import holistic_motion as hm

    points = np.array([np.zeros(dof), np.full(dof, 0.2)])
    expected_points = points.copy()
    velocity = np.ones(dof)
    calls = []

    def mutate():
        points.fill(np.nan)
        velocity.fill(0.0)
        calls.append(True)

    class Acceleration:
        def __array__(self, dtype=None, copy=None):
            mutate()
            return np.ones(dof, dtype=dtype)

    class Duration:
        def __float__(self):
            mutate()
            return 0.0

    inputs = {"max_acceleration": np.ones(dof), "minimum_duration": 0.0}
    inputs[later_input] = (
        Acceleration() if later_input == "max_acceleration" else Duration()
    )
    trajectory = getattr(hm, factory_name)(
        points, velocity, max_jerk=np.ones(dof), **inputs
    )
    assert calls == [True]
    np.testing.assert_array_equal(trajectory.waypoints, expected_points)
    np.testing.assert_array_equal(trajectory.max_velocity, np.ones(dof))
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), expected_points[-1], atol=1e-7
    )


@pytest.mark.parametrize("later_input", ["max_acceleration", "max_jerk"])
def test_trajectory_constraints_snapshot_limits_before_later_conversion(later_input):
    import holistic_motion as hm

    velocity, acceleration = np.ones(3), np.ones(3)
    calls = []

    class Limit:
        def __array__(self, dtype=None, copy=None):
            velocity.fill(0.0)
            if later_input == "max_jerk":
                acceleration.fill(0.0)
            calls.append(True)
            return np.ones(3, dtype=dtype)

    inputs = {
        "max_velocity": velocity,
        "max_acceleration": acceleration,
        "max_jerk": np.ones(3),
    }
    inputs[later_input] = Limit()
    constraints = hm.TrajectoryConstraints(**inputs)
    assert calls == [True]
    assert constraints.valid


@pytest.mark.parametrize(
    "field", ["start", "end", "max_velocity", "max_acceleration", "max_jerk"]
)
@pytest.mark.parametrize("representation", ["array", "list", "object"])
def test_cartesian_trajectory_rejects_complex_inputs(field, representation):
    import holistic_motion as hm

    end = np.eye(4)
    end[0, 3] = 0.2
    inputs = {
        "start": np.eye(4),
        "end": end,
        "max_velocity": np.ones(6),
        "max_acceleration": np.ones(6),
        "max_jerk": np.ones(6),
    }
    bad = inputs[field].astype(complex)
    bad.flat[0] += 4j
    if representation == "list":
        bad = bad.tolist()
    elif representation == "object":
        bad = bad.astype(object)
    inputs[field] = bad
    with pytest.raises(TypeError, match="real"):
        hm.CartesianLineTrajectory(**inputs)


@pytest.mark.parametrize("representation", ["array", "list", "object"])
def test_cartesian_trajectory_preserves_real_input_forms(representation):
    import holistic_motion as hm

    start, end = np.eye(4), np.eye(4)
    end[0, 3] = 0.2
    inputs = [start, end, np.ones(6), np.ones(6), np.ones(6)]
    if representation == "list":
        inputs = [value.tolist() for value in inputs]
    elif representation == "object":
        inputs = [value.astype(object) for value in inputs]
    trajectory = hm.CartesianLineTrajectory(*inputs)
    np.testing.assert_allclose(trajectory.position(0), start, atol=1e-7)
    np.testing.assert_allclose(trajectory.position(trajectory.duration), end, atol=1e-7)


@pytest.mark.parametrize("later_input", ["end", "minimum_duration"])
def test_cartesian_trajectory_snapshots_poses_before_later_conversion(later_input):
    import holistic_motion as hm

    start, end = np.eye(4), np.eye(4)
    end[0, 3] = 0.2
    expected_start, expected_end = start.copy(), end.copy()
    calls = []

    class End:
        def __array__(self, dtype=None, copy=None):
            start.fill(np.nan)
            calls.append(True)
            return np.array(end, dtype=dtype)

    class Duration:
        def __float__(self):
            start.fill(np.nan)
            end.fill(np.nan)
            calls.append(True)
            return 0.0

    inputs = {"end": end, "minimum_duration": 0.0}
    inputs[later_input] = End() if later_input == "end" else Duration()
    trajectory = hm.CartesianLineTrajectory(
        start,
        max_velocity=np.ones(6),
        max_acceleration=np.ones(6),
        max_jerk=np.ones(6),
        **inputs,
    )
    assert calls == [True]
    np.testing.assert_allclose(trajectory.position(0.0), expected_start, atol=1e-7)
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), expected_end, atol=1e-7
    )
