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
