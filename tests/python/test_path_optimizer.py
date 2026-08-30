import math
import time

import holistic_motion as hm
import numpy as np
import pytest


def _options(**overrides):
    options = hm.PathOptimizationOptions()
    options.max_iterations = 80
    options.timeout_seconds = 0.2
    options.step_size = 0.4
    options.edge_resolution = 0.03
    options.length_weight = 1.0
    options.smoothness_weight = 0.5
    for name, value in overrides.items():
        setattr(options, name, value)
    return options


def test_path_optimizer_smooths_path_and_preserves_endpoints():
    path = np.array(
        [
            [-0.8, 0.0],
            [-0.4, 0.5],
            [0.0, -0.5],
            [0.4, 0.5],
            [0.8, 0.0],
        ]
    )
    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    result = optimizer.optimize(path, _options())

    assert result.success
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    np.testing.assert_allclose(result.path[0], path[0])
    np.testing.assert_allclose(result.path[-1], path[-1])
    assert result.statistics.accepted_updates > 0
    assert result.statistics.final_objective < result.statistics.initial_objective


def test_path_optimizer_backtracks_to_a_feasible_step():
    def outside_obstacle(q):
        return np.linalg.norm(q) >= 0.3

    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0], outside_obstacle)
    path = [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]

    single_step = _options(max_iterations=1, step_size=1.0, line_search_steps=1)
    assert (
        optimizer.optimize(path, single_step).status
        == hm.PathOptimizationStatus.UNCHANGED
    )

    backtracking = _options(
        max_iterations=1,
        step_size=1.0,
        line_search_steps=2,
        line_search_decay=0.5,
    )
    result = optimizer.optimize(path, backtracking)
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    assert result.statistics.line_search_evaluations == 2
    np.testing.assert_allclose(result.path[1], [0.0, 0.5], atol=1e-12)
    assert result.statistics.final_path_length <= result.statistics.initial_path_length


def test_path_optimizer_does_not_revalidate_known_edge_endpoints():
    optimizer = hm.PathOptimizer([-1.0], [1.0], lambda _q: True)
    options = _options(max_iterations=1, edge_resolution=2.0)
    result = optimizer.optimize([[-0.8], [0.4], [0.8]], options)

    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    # Three input states, then the candidate. Adjacent endpoints are already
    # known to be valid and must not trigger duplicate validator calls.
    assert result.statistics.collision_checks == 4


def test_path_optimizer_state_cost_pushes_path_away_from_obstacle():
    def clearance_cost(q):
        deficit = max(0.0, 0.6 - np.linalg.norm(q))
        return deficit * deficit

    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    optimizer.set_state_cost(clearance_cost)
    options = _options()
    options.length_weight = 0.1
    options.smoothness_weight = 0.1
    options.state_cost_weight = 10.0
    options.state_cost_step_size = 0.3
    result = optimizer.optimize([[-1.0, 0.0], [0.0, 0.2], [1.0, 0.0]], options)

    assert result.success
    assert result.path[1][1] > 0.2
    assert result.statistics.state_cost_evaluations > 0
    assert result.statistics.final_objective < result.statistics.initial_objective


def test_path_optimizer_analytic_gradient_avoids_finite_differences():
    def clearance_cost(q):
        radius = np.linalg.norm(q)
        deficit = max(0.0, 0.6 - radius)
        return deficit * deficit

    def clearance_gradient(q):
        q = np.asarray(q)
        radius = np.linalg.norm(q)
        if radius == 0.0 or radius >= 0.6:
            return np.zeros_like(q)
        return -2.0 * (0.6 - radius) * q / radius

    path = [[-1.0, 0.0], [0.0, 0.2], [1.0, 0.0]]
    options = _options()
    options.length_weight = 0.1
    options.smoothness_weight = 0.1
    options.state_cost_weight = 10.0
    options.state_cost_step_size = 0.3

    finite_difference = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    finite_difference.set_state_cost(clearance_cost)
    finite_result = finite_difference.optimize(path, options)

    analytic = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    analytic.set_state_cost(clearance_cost)
    analytic.set_state_cost_gradient(clearance_gradient)
    analytic_result = analytic.optimize(path, options)

    assert analytic_result.success
    assert analytic_result.path[1][1] > 0.2
    assert analytic_result.statistics.state_cost_gradient_evaluations > 0
    assert (
        analytic_result.statistics.state_cost_evaluations
        < finite_result.statistics.state_cost_evaluations
    )


def test_path_optimizer_direction_respects_enabled_objective_terms():
    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0])
    optimizer.set_state_cost(lambda q: (q[0] + 0.8) ** 2)
    optimizer.set_state_cost_gradient(lambda q: [2.0 * (q[0] + 0.8), 0.0])
    options = _options(
        max_iterations=1,
        step_size=0.2,
        length_weight=0.0,
        smoothness_weight=0.0,
        state_cost_weight=1.0,
        state_cost_step_size=1.0,
    )
    result = optimizer.optimize([[-1.0, 1.0], [0.2, 0.5], [1.0, 1.0]], options)

    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    assert result.path[1][0] < 0.2
    assert result.path[1][1] == pytest.approx(0.5)


def test_path_optimizer_preconditions_gradient_with_joint_weights():
    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0])
    optimizer.set_joint_weights([100.0, 1.0])
    optimizer.set_state_cost(lambda q: q[0] + q[1] + 4.0)
    optimizer.set_state_cost_gradient(lambda _q: [1.0, 1.0])
    options = _options(
        max_iterations=1,
        step_size=0.2,
        length_weight=0.0,
        smoothness_weight=0.0,
        state_cost_weight=1.0,
        state_cost_step_size=1.0,
    )
    result = optimizer.optimize([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], options)

    movement = np.abs(np.asarray(result.path[1]))
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    assert movement[1] > 50.0 * movement[0]


def test_path_optimizer_geometry_direction_matches_finite_difference():
    path = np.array([[-0.8, -0.2], [0.3, 0.7], [0.9, -0.1]])
    weights = np.array([2.0, 0.5])
    length_weight = 0.8
    smoothness_weight = 1.7

    def objective(middle):
        candidate = path.copy()
        candidate[1] = middle
        differences = np.diff(candidate, axis=0)
        length = np.sum(np.sqrt(np.sum(weights * differences**2, axis=1)))
        acceleration = differences[1] - differences[0]
        smoothness = np.sum(weights * acceleration**2)
        return length_weight * length + smoothness_weight * smoothness

    step = 1e-6
    gradient = np.empty(2)
    for index in range(2):
        positive = path[1].copy()
        negative = path[1].copy()
        positive[index] += step
        negative[index] -= step
        gradient[index] = (objective(positive) - objective(negative)) / (2 * step)
    expected_direction = -gradient / weights
    expected_direction /= np.max(np.abs(expected_direction))

    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0])
    optimizer.set_joint_weights(weights)
    options = _options(
        max_iterations=1,
        step_size=1e-4,
        line_search_steps=1,
        length_weight=length_weight,
        smoothness_weight=smoothness_weight,
    )
    result = optimizer.optimize(path, options)

    actual_direction = (np.asarray(result.path[1]) - path[1]) / options.step_size
    np.testing.assert_allclose(actual_direction, expected_direction, rtol=1e-5)


def test_path_optimizer_incremental_objective_matches_definition():
    x = np.linspace(-0.9, 0.9, 60)
    path = np.column_stack((x, 0.35 * np.sin(8.0 * x)))
    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    options = _options()
    options.max_iterations = 20
    result = optimizer.optimize(path, options)

    output = np.asarray(result.path)
    differences = np.diff(output, axis=0)
    weights = np.array([0.25, 0.25])
    length = np.sum(np.sqrt(np.sum(weights * differences**2, axis=1)))
    acceleration = np.diff(differences, axis=0)
    smoothness = np.sum(weights * acceleration**2)
    expected = options.length_weight * length + options.smoothness_weight * smoothness

    assert result.success
    assert result.statistics.attempted_updates >= len(path) - 2
    assert result.statistics.final_objective == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_path_optimizer_rejects_invalid_analytic_gradient():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(lambda _q: 1.0)
    optimizer.set_state_cost_gradient(lambda _q: [0.0, 1.0])
    options = _options()
    options.state_cost_weight = 1.0
    with pytest.raises(ValueError, match="gradient"):
        optimizer.optimize([[-0.5], [0.0], [0.5]], options)


def test_path_optimizer_rejects_unrepresentable_objective():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_joint_weights([np.finfo(float).max])
    options = _options(length_weight=0.0, smoothness_weight=1.0)
    result = optimizer.optimize([[-1.0], [1.0], [0.0]], options)

    assert not result.success
    assert result.status == hm.PathOptimizationStatus.INVALID_PATH
    assert len(result.path) == 0
    assert "objective is not finite" in result.message


def test_path_optimizer_prunes_unrepresentable_candidate_before_callbacks():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_joint_weights([np.finfo(float).max])
    optimizer.set_state_cost(lambda q: q[0] + 1.0)
    optimizer.set_state_cost_gradient(lambda _q: [1.0])
    options = _options(
        max_iterations=1,
        step_size=1.0,
        line_search_steps=1,
        length_weight=0.0,
        smoothness_weight=1.0,
        state_cost_weight=1.0,
        state_cost_step_size=1.0,
    )
    result = optimizer.optimize([[-1.0], [0.0], [1.0]], options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    assert result.statistics.line_search_evaluations == 1
    assert result.statistics.state_cost_evaluations == 1
    assert result.statistics.collision_checks == 0


def test_path_optimizer_uses_wrapped_continuous_joint_topology():
    optimizer = hm.PathOptimizer([-math.pi], [math.pi])
    optimizer.set_continuous_joints([0])
    result = optimizer.optimize([[3.0], [3.1], [-3.0]], _options())

    assert result.success
    assert abs(result.path[1][0]) > 3.0
    np.testing.assert_allclose(result.path[0], [3.0])
    np.testing.assert_allclose(result.path[-1], [-3.0])


def test_path_optimizer_edge_sampling_uses_cached_shortest_arc():
    samples = []

    def validator(q):
        samples.append(float(q[0]))
        return abs(q[0]) > 2.9

    optimizer = hm.PathOptimizer([-math.pi], [math.pi], validator)
    optimizer.set_continuous_joints([0])
    options = _options(edge_resolution=0.05)
    result = optimizer.optimize([[3.0], [-3.0]], options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    assert len(samples) > 2
    assert all(abs(sample) > 2.9 for sample in samples)


def test_path_optimizer_tiny_resolution_saturates_safely():
    optimizer = hm.PathOptimizer([-1.0], [1.0], lambda _q: True)
    options = _options(timeout_seconds=0.001, edge_resolution=np.nextafter(0.0, 1.0))
    result = optimizer.optimize([[-0.8], [0.8]], options)

    assert result.status == hm.PathOptimizationStatus.TIMEOUT
    assert not result.success


def test_path_optimizer_without_validator_skips_edge_sampling():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = _options(timeout_seconds=0.001, edge_resolution=np.nextafter(0.0, 1.0))
    result = optimizer.optimize([[-0.8], [0.8]], options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    assert result.statistics.collision_checks == 0


def test_path_optimizer_rejects_invalid_input_edge():
    def outside_wall(q):
        return not (-0.2 < q[0] < 0.2 and -0.8 < q[1] < 0.8)

    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0], outside_wall)
    result = optimizer.optimize([[-0.8, 0.0], [0.8, 0.0]], _options())

    assert not result.success
    assert result.status == hm.PathOptimizationStatus.INVALID_PATH
    assert len(result.path) == 0


def test_path_optimizer_timeout_returns_best_feasible_path():
    calls = 0

    def slow_validator(_state):
        nonlocal calls
        calls += 1
        if calls > 3:
            time.sleep(0.02)
        return True

    optimizer = hm.PathOptimizer([-1.0], [1.0], slow_validator)
    options = _options()
    options.timeout_seconds = 0.01
    options.edge_resolution = 2.0
    result = optimizer.optimize([[-0.8], [0.2], [0.8]], options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.TIMEOUT
    np.testing.assert_allclose(result.path[0], [-0.8])
    np.testing.assert_allclose(result.path[-1], [0.8])


def test_path_optimizer_stops_after_gradient_exhausts_timeout():
    validation_calls = 0

    def validator(_state):
        nonlocal validation_calls
        validation_calls += 1
        return True

    def slow_gradient(_state):
        time.sleep(0.02)
        return [1.0]

    optimizer = hm.PathOptimizer([-1.0], [1.0], validator)
    optimizer.set_state_cost(lambda _q: 1.0)
    optimizer.set_state_cost_gradient(slow_gradient)
    options = _options(
        timeout_seconds=0.01,
        edge_resolution=2.0,
        state_cost_weight=1.0,
    )
    path = [[-0.8], [0.2], [0.8]]
    result = optimizer.optimize(path, options)

    assert result.status == hm.PathOptimizationStatus.TIMEOUT
    np.testing.assert_allclose(result.path, path)
    assert validation_calls == 3


def test_path_optimizer_stops_during_initial_state_costs():
    def slow_cost(_state):
        time.sleep(0.02)
        return 1.0

    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(slow_cost)
    options = _options(timeout_seconds=0.01, state_cost_weight=1.0)
    path = [[-0.8], [-0.4], [0.0], [0.4], [0.8]]
    result = optimizer.optimize(path, options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.TIMEOUT
    assert result.statistics.state_cost_evaluations == 1
    assert math.isnan(result.statistics.initial_objective)
    assert math.isnan(result.statistics.final_objective)
    np.testing.assert_allclose(result.path, path)


def test_path_optimizer_prunes_impossible_state_cost_candidates():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(lambda _q: 0.0)
    optimizer.set_state_cost_gradient(lambda _q: [1.0])
    options = _options(max_iterations=1, edge_resolution=2.0, state_cost_weight=1.0)
    result = optimizer.optimize([[-0.8], [0.0], [0.8]], options)

    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    assert result.statistics.line_search_evaluations == options.line_search_steps
    assert result.statistics.state_cost_evaluations == 1
    assert result.statistics.collision_checks == 0


def test_path_optimizer_skips_zero_directions_and_clamped_steps():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = _options(max_iterations=1, edge_resolution=2.0)
    stationary = optimizer.optimize([[-0.8], [0.0], [0.8]], options)
    assert stationary.status == hm.PathOptimizationStatus.UNCHANGED
    assert stationary.statistics.iterations == 1
    assert stationary.statistics.line_search_evaluations == 0
    assert stationary.statistics.collision_checks == 0

    optimizer.set_state_cost(lambda q: q[0] + 1.0)
    optimizer.set_state_cost_gradient(lambda _q: [1.0])
    bounded = _options(
        max_iterations=1,
        edge_resolution=2.0,
        length_weight=0.0,
        smoothness_weight=0.0,
        state_cost_weight=1.0,
        state_cost_step_size=1.0,
    )
    clamped = optimizer.optimize([[-0.5], [-1.0], [0.5]], bounded)
    assert clamped.status == hm.PathOptimizationStatus.UNCHANGED
    assert clamped.statistics.line_search_evaluations == 1
    assert clamped.statistics.state_cost_evaluations == 1


def test_path_optimizer_reuses_current_cost_for_bounded_finite_difference():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(lambda q: (q[0] + 1.0) ** 2)
    options = _options(
        max_iterations=1,
        edge_resolution=2.0,
        length_weight=0.0,
        smoothness_weight=0.0,
        state_cost_weight=1.0,
        state_cost_step_size=1.0,
    )
    result = optimizer.optimize([[-0.5], [-1.0], [0.5]], options)

    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    # One initial cost and one inward finite-difference sample. The clamped
    # outward sample reuses the initial value.
    assert result.statistics.state_cost_evaluations == 2


def test_path_optimizer_two_point_path_uses_fast_path():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = _options(state_cost_weight=1.0)
    result = optimizer.optimize([[-0.8], [0.8]], options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    assert result.statistics.iterations == 0
    assert result.statistics.attempted_updates == 0
    assert result.statistics.state_cost_evaluations == 0
    assert result.statistics.initial_objective == pytest.approx(
        result.statistics.final_objective
    )


def test_path_optimizer_stops_between_finite_difference_sides():
    calls = 0

    def slow_after_initial(_state):
        nonlocal calls
        calls += 1
        if calls > 1:
            time.sleep(0.02)
        return 1.0

    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(slow_after_initial)
    options = _options(timeout_seconds=0.01, state_cost_weight=1.0)
    result = optimizer.optimize([[-0.8], [0.0], [0.8]], options)

    assert result.success
    assert result.status == hm.PathOptimizationStatus.TIMEOUT
    assert result.statistics.state_cost_evaluations == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("step_size", 0.0),
        ("step_size", 1.1),
        ("line_search_steps", 0),
        ("line_search_decay", 0.0),
        ("line_search_decay", 1.0),
        ("edge_resolution", 0.0),
        ("smoothness_weight", -1.0),
    ],
)
def test_path_optimizer_validates_options(name, value):
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = _options()
    setattr(options, name, value)
    result = optimizer.optimize([[-0.5], [0.0], [0.5]], options)
    assert result.status == hm.PathOptimizationStatus.INVALID_PATH


def test_path_optimizer_rejects_unrepresentable_default_weights():
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="default weights"):
        hm.PathOptimizer([-maximum], [maximum])
    with pytest.raises(ValueError, match="default weights"):
        hm.PathOptimizer([0.0], [np.nextafter(0.0, 1.0)])


def test_path_optimizer_requires_state_cost_callback_when_weighted():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = _options()
    options.state_cost_weight = 1.0
    result = optimizer.optimize([[-0.5], [0.0], [0.5]], options)
    assert not result.success
    assert result.status == hm.PathOptimizationStatus.INVALID_PATH
