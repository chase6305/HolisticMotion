"""Uniform objective scaling must preserve the bounded joint update."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("scale", [1e-280, 1e-100, 1e-20, 1e-12, 1.0, 1e100, 1e280])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("fixed_joint", [False, True])
def test_box_qp_retains_the_analytic_minimum_under_uniform_cost_scaling(
    solver_type, scale, direction, fixed_joint
):
    basis = np.array([[180.0, -180.0, -150.0], [2.0, -2.0, -1.0], [-3.0, 4.0, 4.0]])
    hessian = basis.T @ basis + np.eye(3)
    gradient = direction * np.array([16.0, 0.0, -5.0])
    lower, upper = -np.ones(3), np.ones(3)
    # Fixing the first two coordinates at their upper bounds leaves the
    # stationary third coordinate (-5 - (-27014 + 27018)) / 22518.
    # Its gradient vanishes, and both upper-bound multipliers have the correct
    # sign, so strict convexity makes this the unique global minimum.
    expected = direction * np.array([1.0, 1.0, -9.0 / 22518.0])
    if fixed_joint:
        coupling = np.array([2.0, -1.0, 3.0])
        expanded = np.zeros((4, 4))
        expanded[:3, :3] = hessian
        expanded[:3, 3] = expanded[3, :3] = coupling
        expanded[3, 3] = 100.0
        hessian = expanded
        gradient = np.r_[gradient + 0.25 * coupling, -100.0]
        lower, upper = np.r_[lower, 0.25], np.r_[upper, 0.25]
        expected = np.r_[expected, 0.25]

    actual = solver_type._solve_box_qp(hessian * scale, gradient * scale, lower, upper)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert np.all(actual >= lower)
    assert np.all(actual <= upper)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_box_qp_zero_gradient_keeps_fixed_coupled_coordinates(scale):
    hessian = np.array([[2.0, -1.0], [-1.0, 2.0]]) * scale
    actual = PinkRetargetingSolver._solve_box_qp(
        hessian, np.zeros(2), np.array([1.0, -2.0]), np.array([1.0, 2.0])
    )
    np.testing.assert_allclose(actual, [1.0, 0.5], rtol=1e-15, atol=0.0)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("weight", [1e-8, 1e-6, 1.0, 1e6])
def test_weak_frame_tasks_keep_the_same_bounded_retargeting_step(
    tmp_path, solver_type, weight
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    axis_y = np.sqrt(0.19)
    path = tmp_path / "coupled_slides.urdf"
    path.write_text(f"""<robot name="coupled_slides">
      <link name="base"/><link name="first"/><link name="tool"/>
      <joint name="first_joint" type="prismatic">
        <parent link="base"/><child link="first"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/>
      </joint>
      <joint name="second_joint" type="prismatic">
        <parent link="first"/><child link="tool"/><axis xyz="0.9 {axis_y} 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/>
      </joint></robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = solver_type(
        path,
        {"torso": "tool"},
        {"torso": ["first_joint", "second_joint"]},
        frame_tasks={"torso": FrameTask(position_cost=weight, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        damping=0.001 * weight**2,
        integration_dt=1.0,
        **options,
    )
    solver.prepare("torso")
    target = np.eye(4)
    target[:2, 3] = [0.05, 0.455 / axis_y]
    result = solver.step({"torso": target}, seed=[0.0, 0.0])
    np.testing.assert_allclose(
        result.configuration, [-0.85 / 1.001, 1.0], rtol=1e-12, atol=1e-12
    )
    assert result.accepted_steps == 1
