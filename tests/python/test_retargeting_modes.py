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


def test_custom_mode_cycle_and_validation():
    manager = RetargetingModeManager(
        {
            RetargetingMode.LEFT_ARM: RetargetingModeSpec(
                ("tool",), ("manipulator",)
            ),
            RetargetingMode.WHOLE_BODY: RetargetingModeSpec(
                ("tool", "base"), ("whole_body",)
            ),
        },
        initial_mode="left_arm",
    )
    assert manager.cycle().targets == ("tool", "base")
    assert manager.mode is RetargetingMode.WHOLE_BODY
    assert manager.cycle().targets == ("tool",)


def test_target_owns_validated_pose():
    pose = np.eye(4)
    target = RetargetingTarget(pose, weight=2.0)
    pose[0, 0] = 4.0
    assert target.pose[0, 0] == 1.0

    with pytest.raises(ValueError, match="4x4"):
        RetargetingTarget(np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        RetargetingTarget(np.eye(4), weight=0.0)


def test_pinocchio_solver_accepts_arm_and_whole_body_targets(tmp_path):
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
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    result = solver.solve({"left_hand": moved_left})
    assert result.success
    left_idx = solver.model.joints[solver.model.getJointId("left_joint")].idx_q
    np.testing.assert_allclose(result.configuration[left_idx], angle, atol=2e-4)

    solver.set_mode("whole_body")
    result = solver.solve(poses)
    assert result.success
    assert result.mode is RetargetingMode.WHOLE_BODY
