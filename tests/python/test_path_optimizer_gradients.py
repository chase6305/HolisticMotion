"""State-cost differences must follow the actual samples after projection."""

import holistic_motion as hm
import numpy as np
import pytest


def cost_options(step, update):
    options = hm.PathOptimizationOptions()
    options.max_iterations = 1
    options.timeout_seconds = 2.0
    options.step_size = update
    options.line_search_steps = 1
    options.length_weight = 0.0
    options.smoothness_weight = 0.0
    options.state_cost_weight = 1.0
    options.state_cost_step_size = 1.0
    options.finite_difference_step = step
    options.minimum_improvement = 0.0
    return options


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("step", [1e-4, 0.1])
def test_asymmetric_samples_descend_near_joint_limit(direction, step):
    position = direction * (1.0 - step * 0.25)
    minimum = position - direction * step * 0.1
    samples = []

    def cost(q):
        samples.append(q.copy())
        assert -1.0 <= q[0] <= 1.0
        return ((q[0] - minimum) / step) ** 2

    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(cost)
    result = optimizer.optimize(
        [[-0.5], [position], [0.5]], cost_options(step, step * 0.1)
    )

    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    np.testing.assert_allclose(result.path[1], [minimum], atol=1e-14, rtol=0.0)
    assert result.statistics.final_objective < result.statistics.initial_objective
    assert result.statistics.state_cost_evaluations == len(samples) == 4


def test_large_weight_does_not_overflow_representable_geometry_gradient():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_joint_weights([1e308])
    options = cost_options(0.001, 0.1)
    options.state_cost_weight = 0.0
    options.smoothness_weight = 1.0
    result = optimizer.optimize([[-0.5], [0.1], [0.5]], options)
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    np.testing.assert_allclose(result.path[1], [0.0], atol=1e-15, rtol=0.0)
    assert result.statistics.final_objective < result.statistics.initial_objective


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_almost_collapsed_sample_uses_the_longer_side(direction):
    position = direction * np.nextafter(1.0, 0.0)
    samples = []

    def cost(q):
        samples.append(q.copy())
        return (q[0] - direction * 0.9) ** 2 + 1.0

    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(cost)
    result = optimizer.optimize([[-0.5], [position], [0.5]], cost_options(0.01, 0.05))
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    assert abs(result.path[1][0]) < abs(position)
    # No cost query at the nearly identical outward sample.
    assert result.statistics.state_cost_evaluations == len(samples) == 3


@pytest.mark.parametrize("step", [np.pi, 2.0 * np.pi, 10.0, 1e308])
@pytest.mark.parametrize("position", [0.8, -0.8])
def test_large_circular_steps_preserve_descent_direction(step, position):
    optimizer = hm.PathOptimizer([-np.pi], [np.pi])
    optimizer.set_continuous_joints([0])
    optimizer.set_state_cost(lambda q: 1.0 - np.cos(q[0]))
    result = optimizer.optimize([[-0.5], [position], [0.5]], cost_options(step, 0.1))
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    assert abs(result.path[1][0]) == pytest.approx(abs(position) - 0.1)


def test_circular_samples_cross_the_seam_without_reversing_gradient():
    minimum = -np.pi + 0.1
    optimizer = hm.PathOptimizer([-np.pi], [np.pi])
    optimizer.set_continuous_joints([0])
    optimizer.set_state_cost(lambda q: 1.0 - np.cos(q[0] - minimum))
    result = optimizer.optimize([[3.0], [np.pi - 0.05], [-3.0]], cost_options(0.2, 0.1))
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    assert result.path[1][0] == pytest.approx(-np.pi + 0.05)


def test_unrepresentable_numerical_gradient_has_clear_error():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_state_cost(lambda q: 1e300 if q[0] > 0.0 else 0.0)
    with pytest.raises(ValueError, match="finite-difference state cost gradient"):
        optimizer.optimize([[-0.5], [0.0], [0.5]], cost_options(1e-20, 0.1))


@pytest.mark.parametrize("weight", [np.nextafter(0.0, 1.0), 1e-320, 1e-300, 1.0])
@pytest.mark.parametrize("cost_scale", [1.0, 1e200])
@pytest.mark.parametrize("analytic", [False, True])
def test_descent_preserves_weight_ratios_when_preconditioning_overflows(
    weight, cost_scale, analytic
):
    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    optimizer.set_joint_weights([weight, 4.0 * weight])
    optimizer.set_state_cost(lambda q: cost_scale * np.dot(q, q))
    if analytic:
        optimizer.set_state_cost_gradient(lambda q: 2.0 * cost_scale * np.asarray(q))
    result = optimizer.optimize(
        [[-0.5, 0.0], [0.5, 0.25], [0.5, 0.0]], cost_options(0.001, 0.1)
    )
    assert result.status == hm.PathOptimizationStatus.OPTIMIZED
    np.testing.assert_allclose(result.path[1], [0.4, 0.2375], atol=1e-13, rtol=0.0)
    assert result.statistics.final_objective < result.statistics.initial_objective


def test_zero_gradient_remains_stationary_with_subnormal_weights():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_joint_weights([np.nextafter(0.0, 1.0)])
    optimizer.set_state_cost(lambda q: q[0] ** 2)
    optimizer.set_state_cost_gradient(lambda q: [2.0 * q[0]])
    result = optimizer.optimize([[-0.5], [0.0], [0.5]], cost_options(0.001, 0.1))
    assert result.status == hm.PathOptimizationStatus.UNCHANGED
    np.testing.assert_array_equal(result.path[1], [0.0])


def test_overflowing_combined_gradient_is_rejected_before_trial_callbacks():
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    samples = []

    def cost(q):
        samples.append(q.copy())
        return 1.0

    optimizer.set_state_cost(cost)
    optimizer.set_state_cost_gradient(lambda _q: [1e308])
    options = cost_options(0.001, 0.1)
    options.state_cost_weight = 2.0
    with pytest.raises(ValueError, match="combined path objective gradient"):
        optimizer.optimize([[-0.5], [0.0], [0.5]], options)
    assert len(samples) == 1
    np.testing.assert_array_equal(samples[0], [0.0])


@pytest.mark.parametrize("length_weight", [0.0, 1.0])
def test_disabled_smoothness_does_not_overflow_the_active_objective(length_weight):
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    optimizer.set_joint_weights([1e308])
    options = cost_options(0.001, 0.1)
    options.length_weight = length_weight
    options.state_cost_weight = 1.0 - length_weight
    if options.state_cost_weight:
        optimizer.set_state_cost(lambda q: 1e308 * q[0] ** 2)
        optimizer.set_state_cost_gradient(lambda q: [1e308 * (2.0 * q[0])])
    result = optimizer.optimize([[-0.5], [0.5], [-0.5]], options)
    assert result.success, result.message
    assert np.isfinite(result.statistics.final_objective)
    assert result.statistics.initial_objective == pytest.approx(
        2e154 if length_weight else 2.5e307
    )
    np.testing.assert_array_equal(np.asarray(result.path)[[0, -1]], [[-0.5], [-0.5]])
    if options.state_cost_weight:
        assert result.statistics.final_objective < result.statistics.initial_objective
        assert result.path[1][0] == pytest.approx(0.4)
