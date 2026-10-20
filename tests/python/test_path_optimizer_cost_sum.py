"""Accumulate weighted state costs without overflowing an unused raw sum."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("cost_scale", [7e307, 1e308])
@pytest.mark.parametrize("gradient", ["constant", "analytic", "difference"])
@pytest.mark.parametrize("count", [3, 7])
def test_large_state_cost_sum_retains_a_finite_weighted_objective(
    cost_scale, gradient, count
):
    optimizer = hm.PathOptimizer([-1.0], [1.0])
    if gradient == "constant":
        optimizer.set_state_cost(lambda _q: cost_scale)
        optimizer.set_state_cost_gradient(lambda _q: [0.0])
    else:
        optimizer.set_state_cost(lambda q: cost_scale * (0.5 + q[0] ** 2))
        if gradient == "analytic":
            optimizer.set_state_cost_gradient(lambda q: [cost_scale * (2 * q[0])])
    options = hm.PathOptimizationOptions()
    options.length_weight = 0.0
    options.smoothness_weight = 0.0
    options.state_cost_weight = 1.0 / cost_scale
    options.max_iterations = 1
    options.step_size = 0.1
    options.minimum_improvement = 0.0
    options.timeout_seconds = 10.0
    path = np.array([[-0.5]] + [[0.6]] * count + [[0.5]])
    result = optimizer.optimize(path, options)
    assert result.feasible, result.message
    assert (
        result.statistics.state_cost_evaluations
        == count * {"constant": 1, "analytic": 2, "difference": 4}[gradient]
    )
    assert np.isfinite(result.statistics.initial_objective)
    assert np.isfinite(result.statistics.final_objective)
    if gradient == "constant":
        assert result.status == hm.PathOptimizationStatus.UNCHANGED
        expected_initial = expected_final = float(count)
    else:
        assert result.status == hm.PathOptimizationStatus.OPTIMIZED
        expected_initial, expected_final = count * 0.86, count * 0.75
        np.testing.assert_allclose(np.asarray(result.path)[1:-1], 0.5, atol=1e-14)
    assert result.statistics.initial_objective == pytest.approx(expected_initial)
    assert result.statistics.final_objective == pytest.approx(expected_final)
