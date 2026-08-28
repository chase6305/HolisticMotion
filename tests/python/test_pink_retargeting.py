import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


def test_frame_task_supports_scalar_and_anisotropic_costs():
    task = FrameTask(position_cost=[1.0, 2.0, 3.0], orientation_cost=0.5)
    np.testing.assert_allclose(task.cost, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        FrameTask(position_cost=-1.0)
    with pytest.raises(ValueError, match="gain"):
        FrameTask(gain=1.1)
    with pytest.raises(ValueError, match="read-only"):
        task.position_cost[0] = 4.0


def test_posture_task_validation():
    assert PostureTask().cost > 0.0
    with pytest.raises(ValueError, match="non-negative"):
        PostureTask(cost=-0.1)


def test_solver_rejects_non_finite_numeric_options(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "empty.urdf"
    urdf.write_text('<robot name="empty"><link name="base"/></robot>')

    with pytest.raises(ValueError, match="finite and positive"):
        PinkRetargetingSolver(
            urdf,
            {"left_hand": "base", "right_hand": "base", "head": "base"},
            {"left_arm": [], "right_arm": []},
            damping=float("nan"),
        )
    with pytest.raises(ValueError, match="task tolerances"):
        PinkRetargetingSolver(
            urdf,
            {"left_hand": "base", "right_hand": "base", "head": "base"},
            {"left_arm": [], "right_arm": []},
            position_tolerance=0.0,
        )


def test_box_qp_respects_bounds():
    solution = PinkRetargetingSolver._solve_box_qp(
        np.diag([2.0, 4.0]),
        np.array([4.0, -8.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(solution, [1.0, -1.0])


def test_pink_solver_handles_dual_arm_mode(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "pink.urdf"
    urdf.write_text(
        """<robot name="pink">
        <link name="base"/><link name="left_hand"/>
        <link name="right_hand"/><link name="head"/>
        <joint name="left_joint" type="revolute">
          <parent link="base"/><child link="left_hand"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        <joint name="right_joint" type="revolute">
          <parent link="base"/><child link="right_hand"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        <joint name="head_joint" type="revolute">
          <parent link="base"/><child link="head"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    solver = PinkRetargetingSolver(
        urdf,
        {"left_hand": "left_hand", "right_hand": "right_hand", "head": "head"},
        {"left_arm": ["left_joint"], "right_arm": ["right_joint"]},
        integration_dt=0.1,
        acceleration_limits={
            "left_joint": 0.5,
            "right_joint": 0.5,
            "head_joint": 0.5,
        },
    )
    q = np.asarray(solver.pin.neutral(solver.model))
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    targets = {
        name: np.asarray(solver.data.oMf[frame].homogeneous)
        for name, frame in solver._frame_ids.items()
    }
    solver.set_mode("dual_arm")
    result = solver.solve(targets)
    assert result.success
    assert result.iterations == 1
    assert result.termination_reason == "converged"
    assert result.position_residual == pytest.approx(0.0)
    assert set(result.target_residuals) == {"left_hand", "right_hand"}

    angle = 0.08
    moved = {name: pose.copy() for name, pose in targets.items()}
    moved["left_hand"][:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    step_result = solver.step(moved, seed=q)
    assert step_result.termination_reason == "maximum_iterations"
    left_velocity = solver.model.joints[solver.model.getJointId("left_joint")].idx_v
    assert abs(solver._last_velocity[left_velocity]) <= 0.05 + 1e-12

    solver.reset(q)
    result = solver.solve(moved)
    assert result.success
    assert result.accepted_steps > 0
    left_joint = solver.model.joints[solver.model.getJointId("left_joint")]
    np.testing.assert_allclose(result.configuration[left_joint.idx_q], angle, atol=2e-4)
