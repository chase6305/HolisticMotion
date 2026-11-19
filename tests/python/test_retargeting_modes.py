from collections.abc import Mapping

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    PinkRetargetingSolver,
    PinocchioRetargetingSolver,
    RetargetingMode,
    RetargetingModeManager,
    RetargetingModeSpec,
    RetargetingResult,
    RetargetingTarget,
)
from optional_dependencies import python_pinocchio_or_skip


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
    modes = {
        RetargetingMode.LEFT_ARM: RetargetingModeSpec(("tool",), ("manipulator",)),
        RetargetingMode.WHOLE_BODY: RetargetingModeSpec(
            ("tool", "base"), ("whole_body",)
        ),
    }
    manager = RetargetingModeManager(modes, initial_mode="left_arm")
    modes.clear()
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
    with pytest.raises(TypeError, match="sequences"):
        RetargetingModeSpec("hand", ("arm",))
    with pytest.raises(TypeError, match="sequences"):
        RetargetingModeSpec(("hand",), "arm")


def test_mode_manager_exposes_immutable_validated_snapshot():
    manager = RetargetingModeManager()
    snapshot = manager.mode_specs
    with pytest.raises(TypeError):
        snapshot[RetargetingMode.LEFT_ARM] = snapshot[RetargetingMode.RIGHT_ARM]
    with pytest.raises(TypeError, match="RetargetingModeSpec"):
        RetargetingModeManager({RetargetingMode.LEFT_ARM: object()}, "left_arm")
    with pytest.raises(TypeError, match="mapping"):
        RetargetingModeManager([])
    from collections.abc import Mapping

    class DuplicateModes(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 2

        def items(self):
            spec = RetargetingModeSpec(("hand",), ("arm",))
            return (("left_arm", spec), (RetargetingMode.LEFT_ARM, spec))

    with pytest.raises(ValueError, match="duplicate"):
        RetargetingModeManager(DuplicateModes(), "left_arm")


def test_mode_manager_rejects_ambiguous_target_collections():
    manager = RetargetingModeManager(initial_mode="left_arm")
    with pytest.raises(TypeError, match="not a string"):
        manager.validate_targets("left_hand")
    with pytest.raises(ValueError, match="non-empty strings"):
        manager.validate_targets(["left_hand", ""])


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
    with pytest.raises(TypeError, match="weight must be numeric"):
        RetargetingTarget(np.eye(4), weight=object())


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_target_rejects_complex_pose_without_discarding_components(dtype, imaginary):
    pose = np.eye(4, dtype=dtype)
    pose[0, 3] = np.complex128(0.5 + imaginary * 1j)
    with pytest.raises(TypeError, match="real"):
        RetargetingTarget(pose)


def test_target_accepts_real_object_pose_and_owns_it():
    pose = np.eye(4, dtype=object)
    pose[0, 3] = 0.5
    target = RetargetingTarget(pose)
    pose[0, 3] = 2.0
    assert target.pose.dtype == np.float64
    assert target.pose[0, 3] == 0.5
    assert not target.pose.flags.writeable


@pytest.mark.parametrize("value", [np.complex64(0.5), np.complex128(0.5 + 1j)])
def test_target_rejects_complex_weight(value):
    with pytest.raises(TypeError, match="real"):
        RetargetingTarget(np.eye(4), weight=value)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
def test_retargeting_result_rejects_complex_configuration(dtype):
    from holistic_motion.kit.retargeting import RetargetingResult

    values = np.array([np.complex128(0.5 + 1j)], dtype=dtype)
    with pytest.raises(TypeError, match="real"):
        RetargetingResult(values, True, 1, 0.0, 0.0, RetargetingMode.LEFT_ARM)


@pytest.mark.parametrize(
    "field",
    [
        "residual",
        "solve_ms",
        "collision_cost",
        "objective",
        "position_residual",
        "orientation_residual",
        "support_polygon_violation",
        "center_of_mass_residual",
        "zmp_residual",
    ],
)
def test_retargeting_result_rejects_complex_diagnostics(field):
    from holistic_motion.kit.retargeting import RetargetingResult

    kwargs = {
        "configuration": [0.0],
        "success": True,
        "iterations": 1,
        "residual": 0.0,
        "solve_ms": 0.0,
        "mode": RetargetingMode.LEFT_ARM,
    }
    kwargs[field] = np.complex128(0.5 + 1j)
    with pytest.raises(TypeError, match="real"):
        RetargetingResult(**kwargs)


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


def test_retargeting_result_validates_public_snapshot():
    from holistic_motion.kit.retargeting import RetargetingResult

    values = {
        "configuration": np.zeros(1),
        "success": False,
        "iterations": 1,
        "residual": 0.1,
        "solve_ms": 1.0,
        "mode": RetargetingMode.LEFT_ARM,
    }
    with pytest.raises(ValueError, match="residual"):
        RetargetingResult(**{**values, "residual": -1.0})
    with pytest.raises(ValueError, match="two values"):
        RetargetingResult(**values, target_residuals={"hand": (0.1,)})
    with pytest.raises(ValueError, match="finite and non-negative"):
        RetargetingResult(**values, target_residuals={"hand": (float("nan"), 0.1)})
    with pytest.raises(TypeError, match="integer"):
        RetargetingResult(**{**values, "iterations": True})
    with pytest.raises(ValueError, match="termination_reason"):
        RetargetingResult(**values, termination_reason="")


def test_pinocchio_linear_solve_falls_back_to_least_squares(monkeypatch):
    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", fail)
    result = PinocchioRetargetingSolver._linear_solve(
        np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([2.0, 0.0])
    )

    np.testing.assert_allclose(result, [2.0, 0.0])


def test_pinocchio_solver_accepts_arm_and_whole_body_targets(tmp_path, monkeypatch):
    python_pinocchio_or_skip()
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
    frames = {
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "head": "head",
    }
    groups = {"left_arm": ["left_joint"], "right_arm": ["right_joint"]}
    with pytest.raises(TypeError, match="max_iterations"):
        PinocchioRetargetingSolver(urdf, frames, groups, max_iterations=1.5)
    with pytest.raises(TypeError, match="frames"):
        PinocchioRetargetingSolver(urdf, [], groups)
    with pytest.raises(TypeError, match="sequences"):
        PinocchioRetargetingSolver(urdf, frames, {"left_arm": "left_joint"})
    with pytest.raises(ValueError, match="duplicate"):
        PinocchioRetargetingSolver(
            urdf, frames, {"left_arm": ["left_joint", "left_joint"]}
        )
    solver = PinocchioRetargetingSolver(
        urdf,
        frames,
        groups,
    )
    with pytest.raises(TypeError, match="targets must be a mapping"):
        solver.solve([])
    with pytest.raises(TypeError):
        solver.frames["left_hand"] = "base"
    with pytest.raises(TypeError):
        solver.joint_groups["left_arm"] = ("right_joint",)
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
    mode_plan = solver._mode_plan()
    assert solver._mode_plan() is mode_plan
    with pytest.raises(ValueError, match="read-only"):
        mode_plan.active_velocity_indices[0] = 0

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

    # When the iteration budget is exhausted, metrics must describe the
    # integrated configuration rather than the state before the final step.
    budget_target = poses["left_hand"].copy()
    budget_angle = 0.5
    budget_target[:3, :3] = np.array(
        [
            [np.cos(budget_angle), -np.sin(budget_angle), 0.0],
            [np.sin(budget_angle), np.cos(budget_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    solver.max_iterations = 1
    result = solver.solve({"left_hand": budget_target}, seed=np.zeros(solver.nq))
    solver.pin.forwardKinematics(solver.model, solver.data, result.configuration)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    actual_error = solver.pin.log6(
        solver.data.oMf[solver._frame_ids["left_hand"]].inverse()
        * solver.pin.SE3(budget_target[:3, :3], budget_target[:3, 3])
    ).vector
    assert result.residual == pytest.approx(np.linalg.norm(actual_error))
    assert result.residual < budget_angle
    solver.max_iterations = 50

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
    python_pinocchio_or_skip()
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    solver = PinocchioRetargetingSolver(urdf, {"left_hand": "base"}, {"left_arm": []})
    solver.prepare("left_arm")
    assert RetargetingMode.LEFT_ARM in solver._mode_plans
    with pytest.raises(ValueError, match="no frame mappings"):
        solver.prepare("whole_body")
    assert solver.mode is RetargetingMode.LEFT_ARM
    target = np.eye(4)
    target[0, 3] = 1.0

    result = solver.solve({"left_hand": target})

    assert not result.success
    assert result.iterations == 1
    assert result.residual == pytest.approx(1.0)


@pytest.mark.parametrize(
    "solver_type",
    [PinocchioRetargetingSolver, PinkRetargetingSolver, CuroboRetargetingSolver],
)
@pytest.mark.parametrize("action", ["clear", "replace"])
def test_frame_map_is_captured_before_joint_group_conversion(
    tmp_path, solver_type, action
):
    python_pinocchio_or_skip()
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    frames = {"left_hand": "base"}
    calls = []

    class JointNames:
        def __iter__(self):
            calls.append(True)
            if action == "clear":
                frames.clear()
            else:
                frames["left_hand"] = "missing"
            return iter(())

    solver = solver_type(urdf, frames, {"left_arm": JointNames()})

    assert calls == [True]
    assert solver.frames == {"left_hand": "base"}
    solver.prepare("left_arm")
    if action == "clear":
        assert not frames
    else:
        assert frames == {"left_hand": "missing"}


@pytest.mark.parametrize(
    "solver_type",
    [PinocchioRetargetingSolver, PinkRetargetingSolver, CuroboRetargetingSolver],
)
def test_frame_map_is_read_once_before_validation(tmp_path, solver_type):
    python_pinocchio_or_skip()
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    calls = []

    class Frames(Mapping):
        def __iter__(self):
            return iter(("left_hand",))

        def __len__(self):
            return 1

        def __getitem__(self, name):
            if name != "left_hand":
                raise KeyError(name)
            calls.append(name)
            return "base" if len(calls) == 1 else "missing"

    solver = solver_type(urdf, Frames(), {"left_arm": []})

    assert calls == ["left_hand"]
    assert solver.frames == {"left_hand": "base"}
    solver.prepare("left_arm")


@pytest.mark.parametrize(
    "solver_type",
    [PinocchioRetargetingSolver, PinkRetargetingSolver, CuroboRetargetingSolver],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_solver_uses_the_target_value_it_validates(tmp_path, solver_type, wrapped):
    python_pinocchio_or_skip()
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    solver = solver_type(urdf, {"left_hand": "base"}, {"left_arm": []})
    solver.prepare("left_arm")
    pose = np.eye(4)
    first = RetargetingTarget(pose) if wrapped else pose
    replacement = pose.copy()
    replacement[0, 3] = 1.0

    class Targets(Mapping):
        reads = 0

        def __iter__(self):
            return iter(("left_hand",))

        def __len__(self):
            return 1

        def __getitem__(self, name):
            if name != "left_hand":
                raise KeyError(name)
            self.reads += 1
            return first if self.reads == 1 else replacement

    result = solver.solve(Targets())

    assert result.success
    assert result.residual == 0.0
    assert result.configuration.shape == (0,)


@pytest.mark.parametrize(
    "solver_type",
    [PinocchioRetargetingSolver, PinkRetargetingSolver, CuroboRetargetingSolver],
)
@pytest.mark.parametrize("action", ["clear", "replace"])
def test_solver_captures_selected_targets_before_pose_conversion(
    tmp_path, solver_type, action
):
    python_pinocchio_or_skip()
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    solver = solver_type(
        urdf,
        {"left_hand": "base", "right_hand": "base"},
        {"left_arm": [], "right_arm": []},
    )
    solver.prepare("dual_arm")
    source = {}
    calls = []

    class FirstPose:
        def __array__(self, dtype=None, copy=None):
            calls.append(True)
            if action == "clear":
                source.clear()
            else:
                replacement = np.eye(4)
                replacement[0, 3] = 1.0
                source["right_hand"] = replacement
            return np.eye(4, dtype=dtype)

    source.update(left_hand=FirstPose(), right_hand=np.eye(4))
    result = solver.solve(source)

    assert calls == [True]
    assert result.success
    assert result.residual == 0.0
    if action == "clear":
        assert source == {}
    else:
        assert source["right_hand"][0, 3] == 1.0


@pytest.mark.parametrize(
    "solver_type",
    [PinocchioRetargetingSolver, PinkRetargetingSolver, CuroboRetargetingSolver],
)
def test_solver_consumes_reused_target_pose_before_next_mapping_lookup(
    tmp_path, solver_type
):
    python_pinocchio_or_skip()
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    solver = solver_type(
        urdf,
        {"left_hand": "base", "right_hand": "base"},
        {"left_arm": [], "right_arm": []},
    )
    solver.prepare("dual_arm")

    class Targets(Mapping):
        pose = np.eye(4)

        def __iter__(self):
            return iter(("left_hand", "right_hand"))

        def __len__(self):
            return 2

        def __getitem__(self, name):
            if name not in ("left_hand", "right_hand"):
                raise KeyError(name)
            self.pose[0, 3] = 0.0 if name == "left_hand" else 1.0
            return self.pose

    result = solver.solve(Targets())

    assert not result.success
    assert result.residual == 1.0


@pytest.mark.parametrize("action", ["clear", "replace"])
def test_result_captures_residual_entries_before_scalar_conversion(action):
    source = {}
    calls = []

    class Residual:
        def __float__(self):
            calls.append(True)
            if action == "clear":
                source.clear()
            else:
                source["head"] = (-1.0, -1.0)
            return 1.0

    source.update(left_hand=(Residual(), 2.0), head=(3.0, 4.0))
    result = RetargetingResult(
        [0.0],
        True,
        1,
        0.0,
        0.0,
        RetargetingMode.LEFT_ARM,
        target_residuals=source,
    )

    assert calls == [True]
    assert list(result.target_residuals) == ["left_hand", "head"]
    assert result.target_residuals == {"left_hand": (1.0, 2.0), "head": (3.0, 4.0)}
    with pytest.raises(TypeError):
        result.target_residuals["left_hand"] = (5.0, 6.0)


def test_result_reads_custom_residual_mapping_before_advancing():
    class Residuals(Mapping):
        def __init__(self):
            self.pair = [0.0, 0.0]

        def __iter__(self):
            return iter(("left_hand", "head"))

        def __len__(self):
            return 2

        def __getitem__(self, name):
            self.pair[:] = {"left_hand": (1.0, 2.0), "head": (3.0, 4.0)}[name]
            return self.pair

    result = RetargetingResult(
        [0.0],
        True,
        1,
        0.0,
        0.0,
        RetargetingMode.LEFT_ARM,
        target_residuals=Residuals(),
    )

    assert result.target_residuals == {"left_hand": (1.0, 2.0), "head": (3.0, 4.0)}
