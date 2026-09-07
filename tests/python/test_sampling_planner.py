import time

import holistic_motion as hm
import numpy as np
import pytest


def _options(algorithm=hm.SamplingAlgorithm.RRT_CONNECT):
    options = hm.PlanningOptions()
    options.algorithm = algorithm
    options.timeout_seconds = 0.15
    options.max_iterations = 3000
    options.extension_range = 0.15
    options.edge_resolution = 0.02
    options.random_seed = 7
    options.shortcut_attempts = 80
    return options


def _outside_wall(q):
    return not (-0.2 < q[0] < 0.2 and -0.75 < q[1] < 0.75)


@pytest.mark.parametrize(
    "algorithm",
    [
        hm.SamplingAlgorithm.RRT_CONNECT,
        hm.SamplingAlgorithm.RRT_STAR,
        hm.SamplingAlgorithm.INFORMED_RRT_STAR,
    ],
)
def test_sampling_planner_routes_around_obstacle(algorithm):
    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], _outside_wall)
    result = planner.plan([-0.8, 0.0], [0.8, 0.0], _options(algorithm))
    assert result.success, result.message
    np.testing.assert_allclose(result.path[0], [-0.8, 0.0])
    np.testing.assert_allclose(result.path[-1], [0.8, 0.0])
    assert all(_outside_wall(q) for q in result.path)
    assert result.statistics.final_path_length <= (
        result.statistics.initial_path_length + 1e-12
    )
    assert result.statistics.collision_checks > 0


def test_sampling_planner_reports_invalid_endpoints():
    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], _outside_wall)
    result = planner.plan([-2.0, 0.0], [0.8, 0.0], _options())
    assert not result.success
    assert result.status == hm.PlanningStatus.INVALID_START

    result = planner.plan([-0.8, 0.0], [0.0, 0.0], _options())
    assert not result.success
    assert result.status == hm.PlanningStatus.INVALID_GOAL


def test_rrt_star_keeps_a_goal_improved_by_rewiring():
    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], _outside_wall)
    options = _options(hm.SamplingAlgorithm.RRT_STAR)
    options.timeout_seconds = 5.0
    options.goal_bias = 0.0
    options.simplify_path = False
    options.random_seed = 0
    options.max_iterations = 155
    earlier = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
    options.max_iterations = 160
    later = planner.plan([-0.8, 0.0], [0.8, 0.0], options)

    assert earlier.success and later.success
    assert (
        later.statistics.final_path_length
        <= earlier.statistics.final_path_length + 1e-12
    )


def test_sampling_planner_is_deterministic_and_interpolates():
    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], _outside_wall)
    options = _options()
    options.interpolate_path = True
    options.interpolation_points = 25
    first = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
    second = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
    assert first.success and second.success
    assert len(first.path) == 25
    np.testing.assert_allclose(first.path, second.path)


def test_sampling_planner_interpolation_preserves_corners():
    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], _outside_wall)
    options = _options()
    options.shortcut_attempts = 0
    options.timeout_seconds = 2.0
    original = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
    options.interpolate_path = True
    options.interpolation_points = len(original.path) + 1
    interpolated = planner.plan([-0.8, 0.0], [0.8, 0.0], options)

    assert original.success and interpolated.success
    assert len(interpolated.path) == options.interpolation_points
    for corner in original.path:
        assert any(
            np.linalg.norm(corner - state) < 1e-12 for state in interpolated.path
        )
    assert interpolated.statistics.final_path_length == pytest.approx(
        original.statistics.final_path_length
    )


@pytest.mark.parametrize("expire_deadline", [False, True])
def test_sampling_planner_keeps_original_if_interpolation_fails(expire_deadline):
    checked = []

    def valid(state):
        checked.append(float(state[0]))
        if state[0] == 0.25:
            if expire_deadline:
                time.sleep(0.02)
                return True
            return False
        return True

    planner = hm.SamplingPlanner([0.0], [1.0], valid)
    options = _options()
    options.timeout_seconds = 0.01 if expire_deadline else 1.0
    options.edge_resolution = 0.6
    options.interpolate_path = True
    options.interpolation_points = 5
    result = planner.plan([0.0], [1.0], options)

    # The original grid checks 0, 0.5, and 1. The denser candidate is rejected
    # atomically when its first new sample fails or consumes the deadline.
    assert result.success, result.message
    assert 0.25 in checked
    np.testing.assert_allclose(result.path, [[0.0], [1.0]])


def test_sampling_planner_validates_options():
    planner = hm.SamplingPlanner([-1.0], [1.0])
    options = _options()
    options.goal_bias = 2.0
    result = planner.plan([0.0], [0.5], options)
    assert result.status == hm.PlanningStatus.INVALID_PROBLEM

    for name in ("timeout_seconds", "extension_range", "edge_resolution"):
        options = _options()
        setattr(options, name, float("nan"))
        result = planner.plan([0.0], [0.5], options)
        assert result.status == hm.PlanningStatus.INVALID_PROBLEM

    options = _options()
    options.interpolate_path = True
    options.interpolation_points = 1
    result = planner.plan([0.0], [0.5], options)
    assert result.status == hm.PlanningStatus.INVALID_PROBLEM

    with pytest.raises(ValueError):
        planner.set_joint_weights([-1.0])


def test_sampling_planner_timeout_includes_validation():
    def slow_validator(_state):
        time.sleep(0.02)
        return True

    planner = hm.SamplingPlanner([-1.0], [1.0], slow_validator)
    options = _options()
    options.timeout_seconds = 0.005
    result = planner.plan([0.0], [0.5], options)

    assert result.status == hm.PlanningStatus.TIMEOUT
    assert result.statistics.planning_time_ms >= 5.0


def test_sampling_planner_timeout_includes_single_segment_callback():
    calls = 0

    def slow_validator(_state):
        nonlocal calls
        calls += 1
        if calls == 3:
            time.sleep(0.01)
        return True

    planner = hm.SamplingPlanner([-1.0], [1.0], slow_validator)
    options = _options()
    options.timeout_seconds = 0.005
    options.edge_resolution = 2.0
    result = planner.plan([0.0], [0.5], options)

    assert result.status == hm.PlanningStatus.TIMEOUT
    assert calls == 3


def test_sampling_planner_safely_handles_unrepresentable_segment_count():
    planner = hm.SamplingPlanner([-1.0], [1.0], lambda _state: True)
    options = _options()
    options.timeout_seconds = 0.005
    options.edge_resolution = np.nextafter(0.0, 1.0)
    result = planner.plan([-0.5], [0.5], options)

    assert result.status == hm.PlanningStatus.TIMEOUT


def test_sampling_planner_only_counts_real_validator_calls():
    planner = hm.SamplingPlanner([-1.0], [1.0])
    result = planner.plan([-0.5], [0.5], _options())

    assert result.success
    assert result.statistics.collision_checks == 0


@pytest.mark.parametrize("periodic", [False, True])
def test_sampling_planner_without_validator_skips_edge_sampling(periodic):
    planner = hm.SamplingPlanner([-np.pi], [np.pi])
    if periodic:
        planner.set_continuous_joints([0])
    options = _options()
    options.timeout_seconds = 0.05
    options.edge_resolution = np.nextafter(0.0, 1.0)
    options.interpolate_path = True
    options.interpolation_points = 5
    result = planner.plan([-3.0], [3.0], options)

    assert result.success
    assert result.status == hm.PlanningStatus.EXACT_SOLUTION
    assert len(result.path) == 5
    assert result.statistics.collision_checks == 0
    np.testing.assert_allclose(result.path[0], [-3.0])
    np.testing.assert_allclose(result.path[-1], [3.0])
    assert result.statistics.final_path_length == pytest.approx(
        (2 * np.pi - 6.0 if periodic else 6.0) / (2 * np.pi)
    )


def test_sampling_planner_continuous_joint_update_is_transactional():
    planner = hm.SamplingPlanner([-np.pi], [np.pi])
    with pytest.raises(IndexError, match="continuous joint index"):
        planner.set_continuous_joints([0, 1])

    options = _options()
    options.interpolate_path = True
    options.interpolation_points = 3
    result = planner.plan([-3.0], [3.0], options)

    assert result.success
    np.testing.assert_allclose(result.path[1], [0.0], atol=1e-12)


def test_sampling_planner_weighted_periodic_seam():
    checked = []

    def valid(state):
        checked.append(np.asarray(state).copy())
        return abs(state[0]) >= 3.0 - 1e-12

    planner = hm.SamplingPlanner([-np.pi, -1.0], [np.pi, 1.0], valid)
    planner.set_continuous_joints([0])
    planner.set_joint_weights([4.0, 0.25])
    options = _options()
    options.interpolate_path = True
    options.interpolation_points = 31
    result = planner.plan([3.0, -0.4], [-3.0, 0.4], options)

    assert result.success, result.message
    assert len(checked) > 3
    assert all(abs(state[0]) >= 3.0 - 1e-12 for state in result.path)
    expected_length = np.sqrt(4.0 * (2.0 * np.pi - 6.0) ** 2 + 0.25 * 0.8**2)
    assert result.statistics.final_path_length == pytest.approx(expected_length)
    np.testing.assert_allclose(
        np.asarray(result.path)[:, 1], np.linspace(-0.4, 0.4, 31)
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        hm.SamplingAlgorithm.RRT_CONNECT,
        hm.SamplingAlgorithm.RRT_STAR,
        hm.SamplingAlgorithm.INFORMED_RRT_STAR,
    ],
)
def test_sampling_planner_searches_weighted_periodic_space(algorithm):
    planner = hm.SamplingPlanner([-np.pi, -1.0], [np.pi, 1.0], _outside_wall)
    planner.set_continuous_joints([0])
    planner.set_joint_weights([3.0, 0.2])
    options = _options(algorithm)
    options.timeout_seconds = 1.0
    options.extension_range = 0.35
    result = planner.plan([-0.8, 0.0], [0.8, 0.0], options)

    assert result.success, result.message
    assert result.statistics.iterations > 0
    length = 0.0
    for first, second in zip(result.path, result.path[1:]):
        delta = np.asarray(second) - first
        delta[0] = (delta[0] + np.pi) % (2.0 * np.pi) - np.pi
        length += np.sqrt(np.dot([3.0, 0.2], delta**2))
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / options.edge_resolution)))
        samples = first + np.linspace(0.0, 1.0, count + 1)[:, None] * delta
        samples[:, 0] = (samples[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
        assert all(_outside_wall(state) for state in samples)
    assert result.statistics.final_path_length == pytest.approx(length)


def test_sampling_planner_rejects_unrepresentable_default_weights():
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="default weights"):
        hm.SamplingPlanner([-maximum], [maximum])
    with pytest.raises(ValueError, match="default weights"):
        hm.SamplingPlanner([0.0], [np.nextafter(0.0, 1.0)])


def test_sampling_planner_saturates_extreme_finite_timeout():
    planner = hm.SamplingPlanner([-1.0], [1.0])
    options = _options()
    options.timeout_seconds = np.finfo(float).max

    result = planner.plan([-0.5], [0.5], options)

    assert result.success


@pytest.mark.parametrize(
    "algorithm",
    [
        hm.SamplingAlgorithm.RRT_CONNECT,
        hm.SamplingAlgorithm.RRT_STAR,
        hm.SamplingAlgorithm.INFORMED_RRT_STAR,
    ],
)
def test_sampling_planner_uses_shortest_direct_path(algorithm):
    checked = []

    def valid(state):
        checked.append(np.asarray(state).copy())
        return True

    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], valid)
    result = planner.plan([-0.8, -0.4], [0.8, 0.4], _options(algorithm))

    assert result.success
    assert result.message == "direct path found"
    assert result.statistics.iterations == 0
    assert result.statistics.sampled_states == 0
    assert result.statistics.tree_nodes == 2
    assert len(result.path) == 2
    np.testing.assert_allclose(result.path, [[-0.8, -0.4], [0.8, 0.4]])
    assert len(checked) == result.statistics.collision_checks
