"""Reentrant configuration edits apply to subsequent planning requests."""

import numpy as np
import pytest


@pytest.mark.parametrize("algorithm", ["RRT_CONNECT", "RRT_STAR", "INFORMED_RRT_STAR"])
@pytest.mark.parametrize("edit", ["options", "validator"])
@pytest.mark.parametrize("release_gil", [False, True])
def test_sampling_keeps_its_validated_request_during_callbacks(
    algorithm, edit, release_gil
):
    import holistic_motion as hm

    options = hm.PlanningOptions()
    options.algorithm = getattr(hm.SamplingAlgorithm, algorithm)
    options.max_iterations = 64
    options.edge_resolution = 0.01
    options.simplify_path = False
    calls = []

    def valid(q):
        calls.append(float(q[0]))
        if edit == "options":
            options.edge_resolution = np.inf
        else:
            planner.set_state_validator(lambda q: True)
        return abs(q[0]) >= 0.1

    planner = hm.SamplingPlanner([-1.0], [1.0], valid)
    result = planner.plan([-0.8], [0.8], options, release_gil=release_gil)
    # This one-dimensional obstacle disconnects the endpoints. A callback
    # must not replace the validated edge sampler halfway through the query.
    assert not result.success
    assert result.status == hm.PlanningStatus.NO_SOLUTION
    assert result.path == []
    assert len(calls) > 3
    next_result = planner.plan([-0.8], [0.8], options, release_gil=release_gil)
    if edit == "options":
        assert next_result.status == hm.PlanningStatus.INVALID_PROBLEM
    else:
        assert next_result.success


@pytest.mark.parametrize("edit", ["options", "validator"])
@pytest.mark.parametrize("release_gil", [False, True])
def test_optimizer_keeps_original_edge_validation_during_callbacks(edit, release_gil):
    import holistic_motion as hm

    options = hm.PathOptimizationOptions()
    options.edge_resolution = 0.01

    def valid(q):
        if edit == "options":
            options.edge_resolution = np.inf
        else:
            optimizer.set_state_validator(lambda q: True)
        return abs(q[0]) >= 0.1

    optimizer = hm.PathOptimizer([-1.0], [1.0], valid)
    result = optimizer.optimize([[-0.8], [0.8]], options, release_gil=release_gil)
    assert not result.success and not result.feasible
    assert result.status == hm.PathOptimizationStatus.INVALID_PATH
    if edit == "validator":
        assert optimizer.optimize(
            [[-0.8], [0.8]], options, release_gil=release_gil
        ).success


@pytest.mark.parametrize("edit", ["options", "cost", "gradient"])
@pytest.mark.parametrize("release_gil", [False, True])
def test_optimizer_keeps_one_objective_during_a_request(edit, release_gil):
    import holistic_motion as hm

    path = [[-0.8, 0.3], [0.0, 0.5], [0.8, 0.3]]

    def options():
        result = hm.PathOptimizationOptions()
        result.max_iterations = 8
        result.length_weight = 0.0
        result.smoothness_weight = 0.0
        result.state_cost_weight = 1.0
        result.state_cost_step_size = 0.1
        return result

    def cost(q):
        return q[1] ** 2

    def gradient(q):
        return np.array([0.0, 2.0 * q[1]])

    reference = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    reference.set_state_cost(cost)
    reference.set_state_cost_gradient(gradient)
    expected = reference.optimize(path, options(), release_gil=release_gil)
    configured = options()
    edited = []

    def changing_cost(q):
        if not edited:
            edited.append(True)
            if edit == "options":
                configured.state_cost_weight = 0.0
                configured.max_iterations = 0
            elif edit == "cost":
                optimizer.set_state_cost(lambda q: 0.0)
            else:
                optimizer.set_state_cost_gradient(lambda q: np.zeros(2))
        return cost(q)

    optimizer = hm.PathOptimizer([-1.0, -1.0], [1.0, 1.0])
    optimizer.set_state_cost(changing_cost)
    optimizer.set_state_cost_gradient(gradient)
    actual = optimizer.optimize(path, configured, release_gil=release_gil)
    assert actual.success and edited
    assert actual.statistics.accepted_updates == expected.statistics.accepted_updates
    assert actual.statistics.final_objective == pytest.approx(
        expected.statistics.final_objective, abs=1e-15
    )
    np.testing.assert_array_equal(actual.path, expected.path)


@pytest.mark.parametrize("release_gil", [False, True])
def test_periodic_request_retains_its_space_when_callback_reconfigures_planner(
    release_gil,
):
    import holistic_motion as hm

    space = hm.JointSpace([-np.pi], [np.pi], ["turn"]).with_continuous_joints([0])

    def valid(q):
        planner.set_continuous_joints([])
        planner.set_joint_weights([9.0])
        return abs(q[0]) > 2.5

    planner = hm.SamplingPlanner(space, valid)
    result = planner.plan([3.0], [-3.0], release_gil=release_gil)
    assert result.success
    assert result.joint_space.continuous_joint_indices == [0]
    np.testing.assert_array_equal(result.joint_space.joint_weights, space.joint_weights)
    assert planner.joint_space.continuous_joint_indices == []
    np.testing.assert_array_equal(planner.joint_space.joint_weights, [9.0])
