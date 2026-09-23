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
