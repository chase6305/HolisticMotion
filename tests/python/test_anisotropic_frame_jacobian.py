"""Anisotropic frame costs must account for the moving error coordinates."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    RetargetingTarget,
)


def make_solver(tmp_path, solver_type, task, chain=True):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "rotating_axes.urdf"
    extra = (
        """<link name="shoulder"/><link name="tool"/>
      <joint name="pitch" type="revolute">
        <parent link="waist"/><child link="shoulder"/>
        <origin xyz="0.3 0.1 0.2"/><axis xyz="0 1 0"/>
        <limit lower="-3" upper="3" velocity="2" effort="1"/>
      </joint>
      <joint name="roll" type="revolute">
        <parent link="shoulder"/><child link="tool"/>
        <origin xyz="0.4 0.2 0.1"/><axis xyz="1 0 0"/>
        <limit lower="-3" upper="3" velocity="2" effort="1"/>
      </joint>"""
        if chain
        else ""
    )
    urdf.write_text(f"""<robot name="rotating_axes">
      <link name="base"/><link name="waist"/>
      <joint name="yaw" type="revolute">
        <parent link="base"/><child link="waist"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" velocity="2" effort="1"/>
      </joint>{extra}</robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = solver_type(
        urdf,
        {"left_hand": "tool" if chain else "waist"},
        {"left_arm": ["yaw", "pitch", "roll"] if chain else ["yaw"]},
        frame_tasks={"left_hand": task},
        posture_task=PostureTask(cost=0.0),
        step_size=1.0,
        integration_dt=0.1,
        **options,
    )
    solver.prepare("left_arm")
    return solver


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize(
    "position,orientation",
    [
        ([1.0, 0.2, 0.0], 0.0),
        (0.0, [0.3, 1.0, 0.0]),
        ([1.0, 0.2, 0.0], [0.3, 1.0, 0.0]),
    ],
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_anisotropic_objective_gradient_matches_finite_differences(
    tmp_path, solver_type, position, orientation, direction
):
    solver = make_solver(
        tmp_path, solver_type, FrameTask(position, orientation, gain=0.4)
    )
    q = direction * np.array([0.3, -0.4, 0.2])
    pose = np.eye(4)
    pose[:3, :3] = solver.pin.exp3(np.array([0.8, -0.6, 1.1]))
    pose[:3, 3] = [-0.2, 0.8, 0.7]
    poses, weights = solver._prepare_targets(
        {"left_hand": RetargetingTarget(pose, weight=2.3)}
    )
    active, _ = solver._mode_limits()

    def state(configuration, build):
        return solver._task_state(
            configuration,
            weights,
            poses,
            active,
            solver.damping,
            build,
            True,
            [0, 0],
            {},
            {},
        )

    analytic = state(q, True)["gradient"].copy()
    expected = np.zeros(solver.model.nv)
    for axis in range(solver.model.nv):
        delta = np.zeros(solver.model.nv)
        delta[axis] = 1e-6
        positive = solver.pin.integrate(solver.model, q, delta)
        negative = solver.pin.integrate(solver.model, q, -delta)
        expected[axis] = (
            -(state(positive, False)["objective"] - state(negative, False)["objective"])
            / 2e-6
        )
    np.testing.assert_allclose(analytic, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_partial_position_task_can_reduce_error_by_rotating_its_axes(
    tmp_path, solver_type, direction
):
    solver = make_solver(
        tmp_path, solver_type, FrameTask([1.0, 0.0, 0.0], 0.0), chain=False
    )
    pose = np.eye(4)
    pose[:3, 3] = [1.0, direction, 0.0]
    result = solver.solve({"left_hand": pose}, seed=[0.0])
    assert result.success
    assert result.accepted_steps > 0
    np.testing.assert_allclose(
        result.configuration, [-direction * np.pi / 4], atol=1e-4
    )
    # Only local X is tracked. Turning the axes does not reduce full distance.
    assert result.position_residual == pytest.approx(np.sqrt(2.0))
