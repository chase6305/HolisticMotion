"""Collision finite differences must use the actual projected sample offsets."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


def make_solver(tmp_path, solver_type, cost, step, limit=1.0):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "bounded_joint.urdf"
    urdf.write_text(f"""<robot name="bounded_joint">
        <link name="base"/><link name="tool"/>
        <joint name="joint" type="revolute">
          <parent link="base"/><child link="tool"/><axis xyz="0 0 1"/>
          <limit lower="{-limit}" upper="{limit}" velocity="2" effort="1"/>
        </joint></robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = solver_type(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": ["joint"]},
        frame_tasks={"left_hand": FrameTask(position_cost=0.0, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        collision_cost=cost,
        collision_finite_difference_step=step,
        collision_tolerance=1e-12,
        damping=2.0,
        step_size=1.0,
        integration_dt=0.1,
        **options,
    )
    solver.prepare("left_arm")
    return solver


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("step", [1e-4, 0.1])
def test_asymmetric_limit_samples_match_quadratic_derivative_and_reuse_cache(
    tmp_path,
    solver_type,
    direction,
    step,
):
    position = direction * (1.0 - step * 0.25)
    minimum = position - direction * step * 0.1
    calls = []

    def cost(q):
        calls.append(q.copy())
        assert -1.0 <= q[0] <= 1.0
        return float((q[0] - minimum) ** 2)

    solver = make_solver(tmp_path, solver_type, cost, step)
    q = np.array([position])
    counts = [0, 0]
    cost_cache, gradient_cache = {}, {}
    current_cost = solver._collision_cost_value(q, counts, cost_cache)
    gradient = solver._collision_gradient_value(
        q,
        current_cost,
        np.array([0]),
        counts,
        cost_cache,
        gradient_cache,
    )
    np.testing.assert_allclose(gradient, [2.0 * (position - minimum)], atol=1e-12)
    assert direction * gradient[0] > 0.0
    assert counts == [3, 1]
    cached = solver._collision_gradient_value(
        q,
        current_cost,
        np.array([0]),
        counts,
        cost_cache,
        gradient_cache,
    )
    np.testing.assert_array_equal(cached, gradient)
    assert len(calls) == 3
    assert counts == [3, 1]


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_collision_only_solve_descends_near_joint_limit(
    tmp_path, solver_type, direction
):
    minimum = direction * 0.94
    solver = make_solver(
        tmp_path,
        solver_type,
        lambda q: float((q[0] - minimum) ** 2),
        0.1,
    )
    result = solver.solve({"left_hand": np.eye(4)}, seed=[direction * 0.95])
    assert result.success
    np.testing.assert_allclose(result.configuration, [minimum], atol=1e-6)
    assert result.collision_cost <= solver.collision_tolerance


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("limit", [1e-16, 1e-18])
@pytest.mark.parametrize("fraction", [0.0, 0.75])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_representable_small_offsets_preserve_the_quadratic_gradient(
    tmp_path, solver_type, limit, fraction, direction
):
    position = direction * fraction * limit
    minimum = position - direction * 0.1 * limit
    solver = make_solver(
        tmp_path,
        solver_type,
        lambda q: float(((q[0] - minimum) / limit) ** 2),
        limit,
        limit=limit,
    )
    q = np.array([position])
    counts, costs, gradients = [0, 0], {}, {}
    current = solver._collision_cost_value(q, counts, costs)
    actual = solver._collision_gradient_value(
        q, current, np.array([0]), counts, costs, gradients
    )
    # Scale the derivative back to dimensionless units for comparison.
    np.testing.assert_allclose(actual * limit, [direction * 0.2], atol=1e-14)
    assert counts == [3, 1]
    solver._collision_gradient_value(
        q, current, np.array([0]), counts, costs, gradients
    )
    assert counts == [3, 1]


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("limit", [1e-16, 1e-18])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_small_joint_range_does_not_make_collision_solve_stagnate(
    tmp_path, solver_type, limit, direction
):
    minimum = direction * 0.25 * limit
    solver = make_solver(
        tmp_path,
        solver_type,
        lambda q: float(((q[0] - minimum) / limit) ** 2),
        1e-4,
        limit=limit,
    )
    result = solver.solve({"left_hand": np.eye(4)}, seed=[0.0])
    assert result.success
    np.testing.assert_allclose(
        result.configuration / limit, [direction * 0.25], atol=1e-6
    )


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_unrepresentable_offsets_still_reuse_the_current_cost(tmp_path, solver_type):
    solver = make_solver(
        tmp_path, solver_type, lambda q: float((q[0] - 0.25) ** 2), 1e-20
    )
    q = np.array([0.5])
    counts, costs = [0, 0], {}
    current = solver._collision_cost_value(q, counts, costs)
    actual = solver._collision_gradient_value(
        q, current, np.array([0]), counts, costs, {}
    )
    np.testing.assert_array_equal(actual, [0.0])
    assert counts == [1, 1]


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_tiny_side_near_bound_does_not_amplify_cost_roundoff(
    tmp_path, solver_type, direction
):
    solver = make_solver(
        tmp_path,
        solver_type,
        lambda q: float(1e5 + (q[0] - direction * 0.5) ** 2),
        1e-4,
    )
    q = np.array([np.nextafter(direction, 0.0)])
    counts, costs = [0, 0], {}
    current = solver._collision_cost_value(q, counts, costs)
    actual = solver._collision_gradient_value(
        q, current, np.array([0]), counts, costs, {}
    )
    # The outward sample is just one ULP away and its cost rounds to the
    # current value. Use the well-resolved inward side instead.
    np.testing.assert_allclose(actual, [direction], atol=2e-4, rtol=0.0)
    assert counts == [2, 1]
