import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    PinocchioRetargetingSolver,
    RetargetingMode,
    RetargetingModeManager,
    RetargetingModeSpec,
    RetargetingTarget,
)


def test_builtin_arm_and_whole_body_modes():
    manager = RetargetingModeManager(initial_mode="left_arm")
    assert manager.mode is RetargetingMode.LEFT_ARM
    assert manager.spec.targets == ("left_hand",)

    spec = manager.set_mode("dual_arm")
    assert spec.targets == ("left_hand", "right_hand")
    manager.validate_targets({"left_hand", "right_hand"})

    spec = manager.set_mode(RetargetingMode.WHOLE_BODY)
    assert spec.active_joint_groups == ("whole_body",)
    with pytest.raises(ValueError, match="head"):
        manager.validate_targets({"left_hand", "right_hand"})

    spec = manager.set_mode("dual_leg")
    assert spec.targets == ("left_foot", "right_foot")
    assert spec.active_joint_groups == ("left_leg", "right_leg")

    spec = manager.set_mode("full_body")
    assert {
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
        "head",
        "pelvis",
    } == set(spec.targets)


def test_custom_mode_cycle_and_validation():
    manager = RetargetingModeManager(
        {
            RetargetingMode.LEFT_ARM: RetargetingModeSpec(("tool",), ("manipulator",)),
            RetargetingMode.WHOLE_BODY: RetargetingModeSpec(
                ("tool", "base"), ("whole_body",)
            ),
        },
        initial_mode="left_arm",
    )
    assert manager.cycle().targets == ("tool", "base")
    assert manager.mode is RetargetingMode.WHOLE_BODY
    assert manager.cycle().targets == ("tool",)


def test_mode_spec_normalizes_and_owns_sequences():
    targets = ["left_hand"]
    groups = ["left_arm"]
    spec = RetargetingModeSpec(targets, groups)
    targets.append("right_hand")
    groups.append("right_arm")

    assert spec.targets == ("left_hand",)
    assert spec.active_joint_groups == ("left_arm",)
    with pytest.raises(ValueError, match="unique"):
        RetargetingModeSpec(("hand", "hand"), ("arm",))


def test_target_owns_validated_pose():
    pose = np.eye(4)
    target = RetargetingTarget(pose, weight=2.0)
    pose[0, 0] = 4.0
    assert target.pose[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        target.pose[0, 0] = 2.0

    with pytest.raises(ValueError, match="4x4"):
        RetargetingTarget(np.eye(3))
    invalid_row = np.eye(4)
    invalid_row[3, 0] = 0.1
    with pytest.raises(ValueError, match="homogeneous"):
        RetargetingTarget(invalid_row)
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="proper orthonormal"):
        RetargetingTarget(reflection)
    scaled = np.eye(4)
    scaled[0, 0] = 1.1
    with pytest.raises(ValueError, match="proper orthonormal"):
        RetargetingTarget(scaled)
    with pytest.raises(ValueError, match="positive"):
        RetargetingTarget(np.eye(4), weight=0.0)


def test_retargeting_result_owns_immutable_outputs():
    from holistic_motion.kit.retargeting import RetargetingResult

    configuration = np.zeros(2)
    residuals = {"left_hand": (0.1, 0.2)}
    result = RetargetingResult(
        configuration=configuration,
        success=True,
        iterations=1,
        residual=0.2,
        solve_ms=1.0,
        mode=RetargetingMode.LEFT_ARM,
        target_residuals=residuals,
    )
    configuration[0] = 1.0
    residuals["left_hand"] = (9.0, 9.0)

    assert result.configuration[0] == 0.0
    assert result.target_residuals["left_hand"] == (0.1, 0.2)
    with pytest.raises(ValueError, match="read-only"):
        result.configuration[0] = 2.0
    with pytest.raises(TypeError):
        result.target_residuals["head"] = (0.0, 0.0)


def test_pinocchio_linear_solve_falls_back_to_least_squares(monkeypatch):
    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", fail)
    result = PinocchioRetargetingSolver._linear_solve(
        np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([2.0, 0.0])
    )

    np.testing.assert_allclose(result, [2.0, 0.0])


def test_pinocchio_solver_accepts_arm_and_whole_body_targets(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "retargeting.urdf"
    urdf.write_text(
        """<robot name="retargeting">
        <link name="base"/><link name="left_hand"/>
        <link name="right_hand"/><link name="head"/>
        <joint name="left_joint" type="revolute">
          <parent link="base"/><child link="left_hand"/>
          <origin xyz="0 0.2 0"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        <joint name="right_joint" type="revolute">
          <parent link="base"/><child link="right_hand"/>
          <origin xyz="0 -0.2 0"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        <joint name="head_joint" type="revolute">
          <parent link="base"/><child link="head"/>
          <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )
    solver = PinocchioRetargetingSolver(
        urdf,
        {"left_hand": "left_hand", "right_hand": "right_hand", "head": "head"},
        {"left_arm": ["left_joint"], "right_arm": ["right_joint"]},
    )
    # Placements are populated on the first kinematics update.
    solver.pin.forwardKinematics(solver.model, solver.data, np.zeros(solver.nq))
    solver.pin.updateFramePlacements(solver.model, solver.data)
    poses = {
        name: np.array(solver.data.oMf[frame_id].homogeneous)
        for name, frame_id in solver._frame_ids.items()
    }

    solver.set_mode("left_arm")
    result = solver.solve({"left_hand": poses["left_hand"]})
    assert result.success
    assert result.mode is RetargetingMode.LEFT_ARM

    angle = 0.1
    moved_left = poses["left_hand"].copy()
    moved_left[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = solver.solve({"left_hand": moved_left})
    assert result.success
    left_idx = solver.model.joints[solver.model.getJointId("left_joint")].idx_q
    np.testing.assert_allclose(result.configuration[left_idx], angle, atol=2e-4)

    solver.set_mode("whole_body")
    result = solver.solve(poses)
    assert result.success
    assert result.mode is RetargetingMode.WHOLE_BODY

    moved_all = {name: pose.copy() for name, pose in poses.items()}
    for pose, target_angle in zip(moved_all.values(), (0.05, -0.06, 0.04)):
        pose[:3, :3] = np.array(
            [
                [np.cos(target_angle), -np.sin(target_angle), 0.0],
                [np.sin(target_angle), np.cos(target_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    solve_shapes = []
    original_solve = np.linalg.solve

    def counted_solve(matrix, vector):
        solve_shapes.append(matrix.shape)
        return original_solve(matrix, vector)

    monkeypatch.setattr(np.linalg, "solve", counted_solve)
    result = solver.solve(moved_all, seed=np.zeros(solver.nq))

    assert result.success
    assert solve_shapes
    assert set(solve_shapes) == {(solver.model.nv, solver.model.nv)}


def test_pinocchio_solver_reports_unreachable_fixed_model_without_crashing(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    solver = PinocchioRetargetingSolver(urdf, {"left_hand": "base"}, {"left_arm": []})
    solver.set_mode("left_arm")
    target = np.eye(4)
    target[0, 3] = 1.0

    result = solver.solve({"left_hand": target})

    assert not result.success
    assert result.iterations == 1
    assert result.residual == pytest.approx(1.0)
