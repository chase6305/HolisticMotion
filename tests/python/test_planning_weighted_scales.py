"""Weighted distances must not lose finite motion to intermediate squares."""

import math

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "delta,weight",
    [(1e-180, 1e300), (1e-160, 1e300), (1e-180, 1.0), (1e150, 1e-300), (1e160, 1e-320)],
)
def test_optimizer_retains_representable_weighted_path_length(delta, weight):
    bound = max(1.0, 2 * abs(delta))
    optimizer = hm.PathOptimizer([-bound], [bound])
    optimizer.set_joint_weights([weight])
    options = hm.PathOptimizationOptions()
    options.length_weight = 1.0
    options.smoothness_weight = 0.0
    options.max_iterations = 1
    options.edge_resolution = bound
    result = optimizer.optimize([[0.0], [delta]], options)
    expected = math.sqrt(weight) * abs(delta)
    assert result.success
    assert result.statistics.initial_path_length == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )
    assert result.statistics.initial_objective == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )
    np.testing.assert_array_equal(result.path, [[0.0], [delta]])


@pytest.mark.parametrize("factory", [hm.PathOptimizer, hm.SamplingPlanner])
@pytest.mark.parametrize("scale", [1e155, 1e160])
def test_planning_accepts_representable_inverse_squared_ranges(factory, scale):
    planner = factory([0.0], [scale])
    if factory is hm.PathOptimizer:
        result = planner.optimize([[0.0], [scale]])
    else:
        result = planner.plan([0.0], [scale])
    assert result.success
    assert result.statistics.initial_path_length == pytest.approx(1.0, rel=1e-4)
    np.testing.assert_array_equal(np.asarray(result.path)[[0, -1]], [[0.0], [scale]])


@pytest.mark.parametrize("factory", [hm.PathOptimizer, hm.SamplingPlanner])
def test_mixed_weights_do_not_hide_an_underflowed_dominant_component(factory):
    planner = factory([-1.0, -1.0], [1.0, 1.0])
    weights = np.array([1e300, 1e200])
    goal = np.array([1e-180, 1e-140])
    planner.set_joint_weights(weights)
    if factory is hm.PathOptimizer:
        result = planner.optimize([np.zeros(2), goal])
    else:
        result = planner.plan(np.zeros(2), goal)
    assert result.success
    expected = math.hypot(*(np.sqrt(weights) * goal))
    assert result.statistics.initial_path_length == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("scale,weight", [(1.0, 1.0), (1e150, 1e300)])
def test_sampling_can_detour_when_squared_distances_overflow(scale, weight):
    def valid(state):
        return not (0.4 * scale < state[0] < 0.6 * scale and state[1] < 0.2 * scale)

    planner = hm.SamplingPlanner([0.0, 0.0], [scale, scale], valid)
    planner.set_joint_weights([weight, weight])
    options = hm.PlanningOptions()
    options.timeout_seconds = 2.0
    options.max_iterations = 2000
    options.extension_range = 0.2 * scale * math.sqrt(weight)
    options.edge_resolution = 0.05 * scale
    options.goal_bias = 0.25
    options.random_seed = 7
    options.simplify_path = False
    result = planner.plan([0.0, 0.0], [scale, 0.0], options)
    assert result.success
    assert all(valid(state) for state in result.path)
    assert result.statistics.final_path_length > scale * math.sqrt(weight)
    np.testing.assert_array_equal(result.path[0], [0.0, 0.0])
    np.testing.assert_array_equal(result.path[-1], [scale, 0.0])


@pytest.mark.parametrize("weight", [1.0, 1e-40, 1e40, 1e-300, 1e300])
def test_length_optimization_retains_descent_under_global_weight_scaling(weight):
    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0])
    optimizer.set_joint_weights([weight, weight])
    options = hm.PathOptimizationOptions()
    options.smoothness_weight = 0.0
    options.minimum_improvement = 0.0
    options.max_iterations = 10
    result = optimizer.optimize([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], options)
    assert result.success
    assert result.statistics.accepted_updates > 0
    assert result.statistics.final_objective < 0.6 * result.statistics.initial_objective


@pytest.mark.parametrize("scale", [1.0, 1e-10, 1e-16, 1e-100, 1e-150])
def test_optimizer_respects_representable_steps_at_small_coordinate_scales(scale):
    optimizer = hm.PathOptimizer([-2 * scale] * 2, [2 * scale] * 2)
    options = hm.PathOptimizationOptions()
    options.smoothness_weight = 0.0
    options.minimum_improvement = 0.0
    options.step_size = 0.35 * scale
    options.edge_resolution = scale
    options.max_iterations = 10
    path = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]) * scale
    result = optimizer.optimize(path, options)
    assert result.success
    assert result.statistics.accepted_updates > 0
    assert result.statistics.final_objective < 0.6 * result.statistics.initial_objective
    np.testing.assert_array_equal(result.path[0], path[0])
    np.testing.assert_array_equal(result.path[-1], path[-1])


@pytest.mark.parametrize("scale", [1.0, 1e-100, 1e-150])
def test_state_cost_descent_survives_preconditioning_underflow(scale):
    optimizer = hm.PathOptimizer([-2 * scale] * 2, [2 * scale] * 2)
    optimizer.set_state_cost(lambda state: float(state @ state))
    optimizer.set_state_cost_gradient(lambda state: 2 * state)
    options = hm.PathOptimizationOptions()
    options.length_weight = 0.0
    options.smoothness_weight = 0.0
    options.state_cost_weight = 1.0
    options.step_size = 0.35 * scale
    options.edge_resolution = scale
    options.minimum_improvement = 0.0
    options.max_iterations = 10
    result = optimizer.optimize(np.array([[0, 0], [0.5, 1], [1, 0]]) * scale, options)
    assert result.success
    assert result.statistics.accepted_updates > 0
    assert result.statistics.final_objective < 0.1 * result.statistics.initial_objective


@pytest.mark.parametrize("weight", [1.0, 1e308])
def test_preconditioned_direction_keeps_subnormal_component_ratios(weight):
    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    optimizer.set_joint_weights([weight, weight])
    gradient = np.array([0.7e-15, 1e-15])
    optimizer.set_state_cost(lambda state: float(4e-15 + state @ gradient))
    optimizer.set_state_cost_gradient(lambda _state: gradient)
    options = hm.PathOptimizationOptions()
    options.length_weight = 0.0
    options.smoothness_weight = 0.0
    options.state_cost_weight = 1.0
    options.minimum_improvement = 0.0
    options.step_size = 0.2
    options.max_iterations = 1
    result = optimizer.optimize([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]], options)
    assert result.success
    assert result.statistics.accepted_updates == 1
    np.testing.assert_allclose(result.path[1], [-0.14, -0.2], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("lower", [1e20, -1e20, float(2**56), -float(2**56)])
def test_continuous_limits_reject_origins_that_collapse_a_revolution(factory, lower):
    upper = lower + 1e6
    goal = np.nextafter(lower, upper)
    planner = factory([lower], [upper])
    with pytest.raises(ValueError, match="represent"):
        planner.set_continuous_joints([0])
    # A failed topology assignment leaves the ordinary bounded joint usable.
    if factory is hm.PathOptimizer:
        result = planner.optimize([[lower], [goal]])
    else:
        result = planner.plan([lower], [goal])
    assert result.success
    assert result.statistics.final_path_length > 0.0
    np.testing.assert_array_equal(result.path[-1], [goal])


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("lower", [0.0, -math.pi, 1e10])
def test_continuous_limits_keep_origins_that_resolve_a_revolution(factory, lower):
    planner = factory([lower], [lower + 2 * math.pi + 1e-6])
    planner.set_continuous_joints([0])
    if factory is hm.PathOptimizer:
        result = planner.optimize([[lower], [lower + 0.5]])
    else:
        result = planner.plan([lower], [lower + 0.5])
    assert result.success
    np.testing.assert_array_equal(result.path[-1], [lower + 0.5])


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
def test_rejected_continuous_origin_preserves_the_previous_topology(factory):
    lower = 1e20
    planner = factory([-math.pi, lower], [math.pi, lower + 1e6])
    planner.set_continuous_joints([0])
    with pytest.raises(ValueError, match="represent"):
        planner.set_continuous_joints([1])
    path = [[math.pi - 0.1, lower], [-math.pi + 0.1, lower]]
    if factory is hm.PathOptimizer:
        result = planner.optimize(path)
    else:
        result = planner.plan(*path)
    assert result.success
    assert result.statistics.final_path_length == pytest.approx(0.2 / (2 * math.pi))


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("upper", [1e20, 1e100, 1e150])
def test_continuous_limits_reject_an_unresolved_upper_endpoint(factory, upper):
    planner = factory([0.0], [upper])
    with pytest.raises(ValueError, match="represent"):
        planner.set_continuous_joints([0])
    if factory is hm.PathOptimizer:
        result = planner.optimize([[0.0], [1.0]])
    else:
        result = planner.plan([0.0], [1.0])
    assert result.success
    np.testing.assert_array_equal(result.path[-1], [1.0])


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
def test_continuous_limits_allow_a_wide_range_with_resolved_endpoints(factory):
    planner = factory([0.0], [1e10])
    planner.set_continuous_joints([0])
    if factory is hm.PathOptimizer:
        result = planner.optimize([[0.0], [0.5]])
    else:
        result = planner.plan([0.0], [0.5])
    assert result.success
    assert result.statistics.final_path_length > 0.0
    np.testing.assert_array_equal(result.path[-1], [0.5])


@pytest.mark.parametrize(
    "scale,weight,length_weight",
    [
        (1e-100, 1e-300, 1.0),
        (1e-150, 1e-200, 1.0),
        (1e-20, 1e-310, 1.0),
        (1.0, 1e-200, 1e-200),
    ],
)
def test_length_descent_recovers_underflowed_weighted_gradient(
    scale, weight, length_weight
):
    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0])
    optimizer.set_joint_weights([weight, weight])
    options = hm.PathOptimizationOptions()
    options.length_weight = length_weight
    options.smoothness_weight = 0.0
    options.minimum_improvement = 0.0
    options.step_size = 0.35 * scale
    options.max_iterations = 10
    path = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]) * scale
    result = optimizer.optimize(path, options)
    assert result.success
    assert result.statistics.accepted_updates > 0
    assert result.statistics.final_objective < 0.6 * result.statistics.initial_objective
    np.testing.assert_array_equal(result.path[0], path[0])
    np.testing.assert_array_equal(result.path[-1], path[-1])
