"""Position tasks must not inherit errors from an ignored target rotation."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_solver(
    tmp_path, solver_type, yaw, task, active=True, movable_orientation=False
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "cartesian_tool.urdf"
    tool_type = "revolute" if movable_orientation else "fixed"
    tool_limits = (
        '<axis xyz="0 0 1"/><limit lower="-3.1" upper="3.1" velocity="2" effort="1"/>'
        if movable_orientation
        else ""
    )
    urdf.write_text(f"""<robot name="cartesian_tool">
        <link name="base"/><link name="x"/><link name="y"/><link name="tool"/>
        <joint name="x_joint" type="prismatic">
          <parent link="base"/><child link="x"/><axis xyz="1 0 0"/>
          <limit lower="-2" upper="2" velocity="2" effort="1"/>
        </joint>
        <joint name="y_joint" type="prismatic">
          <parent link="x"/><child link="y"/><axis xyz="0 1 0"/>
          <limit lower="-2" upper="2" velocity="2" effort="1"/>
        </joint>
        <joint name="tool_joint" type="{tool_type}">
          <parent link="y"/><child link="tool"/><origin rpy="0 0 {yaw}"/>
          {tool_limits}
        </joint></robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    joints = ["x_joint", "y_joint"] + (["tool_joint"] if movable_orientation else [])
    solver = solver_type(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": joints if active else []},
        frame_tasks={"left_hand": task},
        posture_task=PostureTask(cost=0.0),
        integration_dt=1.0,
        damping=1e-9,
        step_size=1.0,
        **options,
    )
    solver.prepare("left_arm")
    return solver


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("yaw", [0.0, 0.7])
@pytest.mark.parametrize("angle", [-2.8, 2.8])
@pytest.mark.parametrize("position_cost", [1.0, [1.0, 0.0, 0.0]])
def test_position_only_update_is_independent_of_target_rotation(
    tmp_path, solver_type, yaw, angle, position_cost
):
    solver = make_solver(
        tmp_path, solver_type, yaw, FrameTask(position_cost, orientation_cost=0.0)
    )
    aligned = np.eye(4)
    aligned[:3, :3] = rotation_z(yaw)
    aligned[:3, 3] = rotation_z(yaw) @ np.array([0.4, -0.3, 0.0])
    rotated = aligned.copy()
    rotated[:3, :3] = rotation_z(yaw + angle)
    reference = solver.solve({"left_hand": aligned}, seed=[0.0, 0.0], max_iterations=1)
    result = solver.solve({"left_hand": rotated}, seed=[0.0, 0.0], max_iterations=1)

    assert reference.success
    assert result.success
    np.testing.assert_allclose(result.configuration, reference.configuration, atol=1e-9)
    assert result.objective == pytest.approx(reference.objective, abs=1e-12)
    assert result.orientation_residual == pytest.approx(abs(angle))
    actual_distance = np.linalg.norm(rotated[:2, 3] - result.configuration)
    assert result.position_residual == pytest.approx(actual_distance, abs=1e-12)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("cost", [0.0, 1.0])
def test_frame_telemetry_reports_cartesian_distance_with_rotation_error(
    tmp_path, solver_type, cost
):
    solver = make_solver(
        tmp_path, solver_type, 0.7, FrameTask(cost, cost), active=False
    )
    target = np.eye(4)
    target[:3, :3] = rotation_z(0.7 + 2.8)
    target[:3, 3] = [0.4, -0.3, 0.0]
    result = solver.solve({"left_hand": target}, seed=[0.0, 0.0])

    assert result.success == (cost == 0.0)
    assert result.position_residual == pytest.approx(0.5)
    assert result.orientation_residual == pytest.approx(2.8)
    np.testing.assert_allclose(result.target_residuals["left_hand"], [0.5, 2.8])
    assert result.objective == pytest.approx(0.5 * cost**2 * (0.5**2 + 2.8**2))


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("angle", [-1.2, 1.2])
def test_full_pose_update_tracks_translation_and_rotation_together(
    tmp_path, solver_type, angle
):
    solver = make_solver(
        tmp_path, solver_type, 0.7, FrameTask(), movable_orientation=True
    )
    target = np.eye(4)
    target[:3, :3] = rotation_z(0.7 + angle)
    target[:3, 3] = [0.4, -0.3, 0.0]
    result = solver.solve({"left_hand": target}, seed=[0.0, 0.0, 0.0], max_iterations=1)

    assert result.success
    np.testing.assert_allclose(result.configuration, [0.4, -0.3, angle], atol=2e-9)
    assert result.position_residual <= solver.position_tolerance
    assert result.orientation_residual <= solver.orientation_tolerance
