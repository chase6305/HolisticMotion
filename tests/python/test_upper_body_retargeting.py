"""Shared-torso IK and staged retargeting on caller-created robot models."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PinocchioRetargetingSolver,
    PostureTask,
    RetargetingMode,
    RetargetingModeManager,
    RetargetingResult,
    TorsoFirstResult,
    solve_torso_first,
)


def make_solver(
    tmp_path,
    solver_type=PinkRetargetingSolver,
    *,
    redundant=False,
    continuous=False,
    waist_velocity=4.0,
    **kwargs,
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    waist_type = "continuous" if continuous else "revolute"
    left_origin = "0 0 0" if redundant else "0 0.25 0"
    right_origin = "0 0 0" if redundant else "0 -0.25 0"
    urdf = tmp_path / "upper_body.urdf"
    urdf.write_text(
        f'''<robot name="upper_body">
        <link name="base"/><link name="torso_link"/>
        <link name="left_link"/><link name="right_link"/>
        <link name="left_tool"/><link name="right_tool"/><link name="leg"/>
        <joint name="waist" type="{waist_type}">
          <parent link="base"/><child link="torso_link"/><axis xyz="0 0 1"/>
          <limit lower="-1.2" upper="1.2" velocity="{waist_velocity}" effort="10"/>
        </joint>
        <joint name="left" type="revolute">
          <parent link="torso_link"/><child link="left_link"/>
          <origin xyz="{left_origin}"/><axis xyz="0 0 1"/>
          <limit lower="-1.2" upper="1.2" velocity="4" effort="10"/>
        </joint>
        <joint name="right" type="revolute">
          <parent link="torso_link"/><child link="right_link"/>
          <origin xyz="{right_origin}"/><axis xyz="0 0 1"/>
          <limit lower="-1.2" upper="1.2" velocity="4" effort="10"/>
        </joint>
        <joint name="left_tip" type="fixed">
          <parent link="left_link"/><child link="left_tool"/>
          <origin xyz="1 0 0"/>
        </joint>
        <joint name="right_tip" type="fixed">
          <parent link="right_link"/><child link="right_tool"/>
          <origin xyz="1 0 0"/>
        </joint>
        <joint name="aaa_leg" type="revolute">
          <parent link="base"/><child link="leg"/><axis xyz="0 1 0"/>
          <limit lower="-1" upper="1" velocity="2" effort="10"/>
        </joint></robot>'''
    )
    options = {"tolerance": 1e-6, "max_iterations": 150, "step_size": 1.0}
    if issubclass(solver_type, PinkRetargetingSolver):
        options["posture_task"] = PostureTask(cost=0.0)
    if issubclass(solver_type, CuroboRetargetingSolver):
        options["num_seeds"] = 3
    options.update(kwargs)
    return solver_type(
        urdf,
        {"torso": "torso_link", "left_hand": "left_tool", "right_hand": "right_tool"},
        {"torso": ["waist"], "left_arm": ["left"], "right_arm": ["right"]},
        **options,
    )


def configuration(solver, **angles):
    velocity = np.zeros(solver.model.nv)
    for name, angle in angles.items():
        joint = solver.model.joints[solver.model.getJointId(name)]
        velocity[joint.idx_v] = angle
    return np.asarray(solver.pin.integrate(solver.model, solver._neutral_q, velocity))


def poses(solver, q):
    data = solver.model.createData()
    solver.pin.forwardKinematics(solver.model, data, q)
    solver.pin.updateFramePlacements(solver.model, data)
    return {
        name: np.asarray(data.oMf[solver.model.getFrameId(frame)].homogeneous).copy()
        for name, frame in solver.frames.items()
    }


def joint_values(solver, q, name):
    joint = solver.model.joints[solver.model.getJointId(name)]
    return q[joint.idx_q : joint.idx_q + joint.nq]


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("method", ["step", "solve"])
def test_zero_velocity_torso_stays_fixed_during_arm_retargeting(
    tmp_path,
    solver_type,
    continuous,
    method,
):
    solver = make_solver(
        tmp_path, solver_type, continuous=continuous, redundant=True, waist_velocity=0.0
    )
    solver.prepare("torso_left_arm")
    seed = configuration(solver, waist=0.3, left=0.1)
    target = poses(solver, configuration(solver, waist=0.3, left=0.5))["left_hand"]
    result = getattr(solver, method)({"left_hand": target}, seed=seed)
    np.testing.assert_allclose(
        joint_values(solver, result.configuration, "waist"),
        joint_values(solver, seed, "waist"),
        atol=1e-15,
        rtol=0.0,
    )
    assert joint_values(solver, result.configuration, "left")[0] > 0.1
    if method == "solve":
        assert result.success


@pytest.mark.parametrize("continuous", [False, True])
def test_zero_velocity_mode_does_not_generate_alternative_seeds(tmp_path, continuous):
    solver = make_solver(
        tmp_path, CuroboRetargetingSolver, continuous=continuous, waist_velocity=0.0
    )
    solver.prepare("torso")
    seed = configuration(solver, waist=0.3)
    target = poses(solver, configuration(solver, waist=0.8))["torso"]
    result = solver.solve({"torso": target}, seed=seed)
    assert not result.success
    assert solver.last_num_seeds_evaluated == 1
    np.testing.assert_allclose(result.configuration, seed, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("limit", [-1.0, float("nan")])
def test_invalid_velocity_limit_rejected_on_mode_preparation(
    tmp_path, solver_type, limit
):
    solver = make_solver(tmp_path, solver_type)
    # Pinocchio rejects these URDF values; also validate runtime model data.
    waist = solver.model.joints[solver.model.getJointId("waist")]
    solver._model_velocity_limits[waist.idx_v] = limit
    previous_mode = solver.mode
    for _ in range(2):
        with pytest.raises(ValueError, match="velocity"):
            solver.prepare("torso_left_arm")
        assert solver.mode is previous_mode
    # A failed preparation must not corrupt caches for other valid modes.
    solver.prepare("left_arm")


@pytest.mark.parametrize(
    "solver_type",
    [
        PinocchioRetargetingSolver,
        PinkRetargetingSolver,
        CuroboRetargetingSolver,
    ],
)
def test_scaled_continuous_seed_preserves_heading_and_scalar_limits(
    tmp_path, solver_type
):
    solver = make_solver(tmp_path, solver_type, continuous=True)
    seed = solver._neutral_q.copy()
    waist = solver.model.joints[solver.model.getJointId("waist")]
    left = solver.model.joints[solver.model.getJointId("left")]
    seed[waist.idx_q : waist.idx_q + 2] = [2.0, 1.0]
    seed[left.idx_q] = 2.0
    solver.reset(seed)
    expected = configuration(solver, waist=np.arctan2(1.0, 2.0), left=1.2)
    np.testing.assert_allclose(solver._last_q, expected, atol=1e-14)
    np.testing.assert_array_equal(seed[waist.idx_q : waist.idx_q + 2], [2.0, 1.0])


@pytest.mark.parametrize(
    "solver_type",
    [
        PinocchioRetargetingSolver,
        PinkRetargetingSolver,
        CuroboRetargetingSolver,
    ],
)
@pytest.mark.parametrize("operation", ["reset", "solve"])
def test_degenerate_continuous_seed_rejected_without_history_changes(
    tmp_path,
    solver_type,
    operation,
):
    solver = make_solver(tmp_path, solver_type, continuous=True)
    solver.prepare("torso")
    initial = configuration(solver, waist=0.3, left=0.2)
    solver.reset(initial)
    if isinstance(solver, PinkRetargetingSolver):
        solver._last_velocity[:] = 0.02
    bad_seed = initial.copy()
    waist = solver.model.joints[solver.model.getJointId("waist")]
    bad_seed[waist.idx_q : waist.idx_q + 2] = 0.0
    with pytest.raises(ValueError, match="configuration"):
        if operation == "reset":
            solver.reset(bad_seed)
        else:
            solver.solve({"torso": poses(solver, initial)["torso"]}, seed=bad_seed)
    np.testing.assert_array_equal(solver._last_q, initial)
    if isinstance(solver, PinkRetargetingSolver):
        np.testing.assert_array_equal(
            solver._last_velocity, np.full(solver.model.nv, 0.02)
        )


def test_degenerate_posture_target_preserves_preference(tmp_path):
    solver = make_solver(tmp_path, continuous=True)
    initial = configuration(solver, waist=0.3)
    solver.set_posture_target(initial)
    bad = initial.copy()
    waist = solver.model.joints[solver.model.getJointId("waist")]
    bad[waist.idx_q : waist.idx_q + 2] = 0.0
    with pytest.raises(ValueError, match="configuration"):
        solver.set_posture_target(bad)
    np.testing.assert_array_equal(solver._posture_q, initial)


@pytest.mark.parametrize("magnitude", [1e308, 1e-308])
def test_unrepresentable_normalization_rejected(tmp_path, magnitude):
    solver = make_solver(tmp_path, continuous=True)
    bad = solver._neutral_q.copy()
    waist = solver.model.joints[solver.model.getJointId("waist")]
    bad[waist.idx_q : waist.idx_q + 2] = magnitude
    with pytest.raises(ValueError, match="configuration"):
        solver.reset(bad)
    np.testing.assert_array_equal(solver._last_q, solver._neutral_q)


@pytest.mark.parametrize(
    "solver_type",
    [
        PinocchioRetargetingSolver,
        PinkRetargetingSolver,
        CuroboRetargetingSolver,
    ],
)
def test_floating_configuration_normalizes_rotation_without_clipping(
    tmp_path, solver_type
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "floating.urdf"
    urdf.write_text("""<robot name="floating">
        <link name="base"/><link name="body"/>
        <joint name="floating" type="floating">
          <parent link="base"/><child link="body"/>
        </joint></robot>""")
    solver = solver_type(urdf, {"left_hand": "body"}, {"left_arm": ["floating"]})
    seed = np.array([0.2, -0.3, 0.4, 2.0, 1.0, 0.0, 0.0])
    solver.reset(seed)
    expected = seed.copy()
    expected[3:] /= np.sqrt(5.0)
    np.testing.assert_allclose(solver._last_q, expected, atol=1e-14)
    bad = expected.copy()
    bad[3:] = 0.0
    with pytest.raises(ValueError, match="configuration"):
        solver.reset(bad)
    np.testing.assert_allclose(solver._last_q, expected, atol=1e-14)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_torso_takes_over_when_arm_reaches_position_limit(
    tmp_path,
    continuous,
    direction,
    solver_type,
):
    solver = make_solver(
        tmp_path,
        solver_type,
        continuous=continuous,
        redundant=True,
        frame_tasks={"left_hand": FrameTask(position_cost=0.0)},
    )
    solver.prepare("torso_left_arm")
    seed = configuration(solver, left=direction * 1.19)
    target = poses(
        solver, configuration(solver, waist=direction * 0.2, left=direction * 1.19)
    )["left_hand"]
    result = solver.step({"left_hand": target}, seed=seed)
    assert result.success
    assert abs(joint_values(solver, result.configuration, "left")[0]) <= 1.2
    delta = np.asarray(solver.pin.difference(solver.model, seed, result.configuration))
    waist = solver.model.joints[solver.model.getJointId("waist")]
    assert direction * delta[waist.idx_v] > 0.18
    assert np.max(np.abs(delta)) <= 4 * solver.integration_dt + 1e-12


@pytest.mark.parametrize(
    "solver_type",
    [
        PinocchioRetargetingSolver,
        PinkRetargetingSolver,
        CuroboRetargetingSolver,
    ],
)
def test_joint_groups_reject_universe(tmp_path, solver_type):
    solver = make_solver(tmp_path, solver_type)
    with pytest.raises(ValueError, match="movable joint"):
        solver_type(solver.urdf_path, solver.frames, {"torso": ["universe"]})


def test_acceleration_mapping_rejects_universe(tmp_path):
    with pytest.raises(ValueError, match="movable joint"):
        make_solver(tmp_path, acceleration_limits={"universe": 1.0})


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_step_brakes_at_reached_target_and_restarts_from_actual_velocity(
    tmp_path,
    solver_type,
    continuous,
    direction,
):
    solver = make_solver(
        tmp_path, solver_type, continuous=continuous, acceleration_limits={"waist": 0.5}
    )
    solver.prepare("torso")
    q = solver._neutral_q.copy()
    target = poses(solver, configuration(solver, waist=direction * 0.4))["torso"]
    waist = solver.model.joints[solver.model.getJointId("waist")]
    previous_velocity = 0.0
    # Build nonzero velocity, then request the current pose for two cycles.
    for cycle in range(5):
        desired = poses(solver, q)["torso"] if cycle in (2, 3) else target
        result = solver.step({"torso": desired}, seed=q)
        velocity = (
            solver.pin.difference(solver.model, q, result.configuration)[waist.idx_v]
            / solver.integration_dt
        )
        assert abs(velocity - previous_velocity) <= 0.5 * solver.integration_dt + 1e-12
        assert solver._last_velocity[waist.idx_v] == pytest.approx(velocity, abs=1e-12)
        if cycle == 2:
            assert direction * velocity > 0.0  # Reaching a pose cannot bypass braking.
        if cycle == 3:
            assert velocity == pytest.approx(0.0, abs=1e-12)
        previous_velocity = velocity
        q = result.configuration


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_step_with_reached_target_records_zero_command_velocity(tmp_path, solver_type):
    solver = make_solver(tmp_path, solver_type)
    solver.prepare("left_arm")
    target = poses(solver, configuration(solver, left=0.15))["left_hand"]
    moved = solver.step({"left_hand": target}, seed=solver._neutral_q)
    assert np.linalg.norm(solver._last_velocity) > 0.0
    # A hold is feasible without acceleration limits, but must update history.
    held = solver.step(
        {"left_hand": poses(solver, moved.configuration)["left_hand"]},
        seed=moved.configuration,
    )
    np.testing.assert_allclose(held.configuration, moved.configuration, atol=1e-15)
    np.testing.assert_array_equal(solver._last_velocity, np.zeros(solver.model.nv))


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("acceleration", [None, 0.5])
@pytest.mark.parametrize("budget", [None, 5])
def test_acceleration_enforced_solve_returns_one_cycle_command(
    tmp_path,
    solver_type,
    continuous,
    acceleration,
    budget,
):
    options = (
        {} if acceleration is None else {"acceleration_limits": {"waist": acceleration}}
    )
    solver = make_solver(tmp_path, solver_type, continuous=continuous, **options)
    solver.prepare("torso")
    seed = solver._neutral_q.copy()
    target = poses(solver, configuration(solver, waist=1.0))["torso"]
    result = solver.solve(
        {"torso": target}, seed=seed, max_iterations=budget, enforce_acceleration=True
    )
    actual_velocity = (
        np.asarray(solver.pin.difference(solver.model, seed, result.configuration))
        / solver.integration_dt
    )
    waist = solver.model.joints[solver.model.getJointId("waist")]
    assert abs(actual_velocity[waist.idx_v]) <= 4.0 + 1e-12
    if acceleration is not None:
        assert (
            abs(actual_velocity[waist.idx_v])
            <= acceleration * solver.integration_dt + 1e-12
        )
    np.testing.assert_allclose(solver._last_velocity, actual_velocity, atol=1e-12)
    assert result.iterations == 1
    assert result.accepted_steps == 1
    if isinstance(solver, CuroboRetargetingSolver):
        assert solver.last_num_seeds_evaluated == 1


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_offline_solve_retains_iterative_convergence(tmp_path, solver_type):
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = make_solver(tmp_path, solver_type, **options)
    solver.prepare("torso")
    target = poses(solver, configuration(solver, waist=1.0))["torso"]
    result = solver.solve({"torso": target}, seed=solver._neutral_q, max_iterations=10)
    assert result.success
    assert result.accepted_steps > 1


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("budget,error", [(0, ValueError), (1.5, TypeError)])
def test_single_cycle_rejects_invalid_budget_without_changing_history(
    tmp_path,
    solver_type,
    budget,
    error,
):
    solver = make_solver(tmp_path, solver_type)
    solver.prepare("torso")
    seed = configuration(solver, waist=0.15)
    solver.reset(seed)
    velocity = solver._last_velocity.copy()
    target = poses(solver, configuration(solver, waist=1.0))["torso"]
    with pytest.raises(error, match="max_iterations"):
        solver.solve(
            {"torso": target}, max_iterations=budget, enforce_acceleration=True
        )
    np.testing.assert_array_equal(solver._last_q, seed)
    np.testing.assert_array_equal(solver._last_velocity, velocity)


@pytest.mark.parametrize(
    "mode,hands,groups",
    [
        ("torso", ("torso",), ("torso",)),
        ("torso_left_arm", ("left_hand",), ("torso", "left_arm")),
        ("torso_right_arm", ("right_hand",), ("torso", "right_arm")),
        (
            "torso_dual_arm",
            ("left_hand", "right_hand"),
            ("torso", "left_arm", "right_arm"),
        ),
    ],
)
def test_upper_body_mode_targets(mode, hands, groups):
    manager = RetargetingModeManager(initial_mode=mode)
    assert manager.spec.targets == hands
    assert manager.spec.active_joint_groups == groups
    manager.validate_targets(hands)


@pytest.mark.parametrize(
    "solver_type",
    [
        PinocchioRetargetingSolver,
        PinkRetargetingSolver,
        CuroboRetargetingSolver,
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "torso_left_arm",
        "torso_right_arm",
        "torso_dual_arm",
    ],
)
def test_shared_torso_reaches_targets_and_preserves_inactive_joints(
    tmp_path,
    solver_type,
    mode,
):
    solver = make_solver(tmp_path, solver_type)
    seed = configuration(solver, aaa_leg=0.15, left=0.05, right=-0.05)
    desired = configuration(solver, waist=0.4, left=0.2, right=-0.1, aaa_leg=0.15)
    solver.prepare(mode)
    target_poses = poses(solver, desired)
    targets = {name: target_poses[name] for name in solver.mode_manager.spec.targets}
    result = solver.solve(targets, seed=seed)
    assert result.success
    actual = poses(solver, result.configuration)
    for name, target in targets.items():
        np.testing.assert_allclose(actual[name], target, atol=3e-6)
    frozen = ["aaa_leg"]
    if mode == "torso_left_arm":
        frozen.append("right")
    elif mode == "torso_right_arm":
        frozen.append("left")
    for name in frozen:
        np.testing.assert_array_equal(
            joint_values(solver, result.configuration, name),
            joint_values(solver, seed, name),
        )
    np.testing.assert_allclose(
        joint_values(solver, result.configuration, "waist"), [0.4], atol=1e-5
    )
    solver.set_mode(mode.removeprefix("torso_"))
    assert not solver.solve(targets, seed=seed).success


@pytest.mark.parametrize("continuous", [False, True])
def test_joint_motion_cost_reduces_torso_motion_in_redundant_ik(tmp_path, continuous):
    ordinary = make_solver(tmp_path, redundant=True, continuous=continuous)
    costs = {"waist": 2.0}
    penalized = make_solver(
        tmp_path, redundant=True, continuous=continuous, joint_motion_costs=costs
    )
    costs["waist"] = 0.0
    target = poses(ordinary, configuration(ordinary, left=0.15))["left_hand"]
    displacements = []
    for solver in (ordinary, penalized):
        solver.prepare("torso_left_arm")
        result = solver.step({"left_hand": target}, seed=solver._neutral_q)
        delta = solver.pin.difference(
            solver.model, solver._neutral_q, result.configuration
        )
        waist = solver.model.joints[solver.model.getJointId("waist")]
        displacements.append(abs(delta[waist.idx_v]))
        assert np.linalg.norm(
            poses(solver, result.configuration)["left_hand"] - target
        ) < np.linalg.norm(poses(solver, solver._neutral_q)["left_hand"] - target)
    assert displacements[1] < displacements[0] * 0.1
    if continuous:
        assert penalized.model.nq != penalized.model.nv
        assert np.linalg.norm(
            joint_values(penalized, result.configuration, "waist")
        ) == pytest.approx(1.0)


def test_named_posture_preference_works_with_zero_default_and_task_replacement(
    tmp_path,
):
    solver = make_solver(tmp_path, redundant=True)
    costs = {"waist": 0.5}
    task = PostureTask(cost=0.0, joint_costs=costs)
    costs["waist"] = 0.0
    assert task.joint_costs["waist"] == 0.5
    with pytest.raises(TypeError):
        task.joint_costs["waist"] = 0.0
    solver.prepare("torso_left_arm")
    solver.set_posture_target(configuration(solver, waist=0.35))
    target = poses(solver, configuration(solver, left=0.5))["left_hand"]
    original = solver.solve({"left_hand": target}, seed=solver._neutral_q)
    solver.posture_task = task
    preferred = solver.solve({"left_hand": target}, seed=solver._neutral_q)
    assert original.success and preferred.success
    np.testing.assert_allclose(
        joint_values(solver, preferred.configuration, "waist"), [0.35], atol=1e-5
    )
    assert abs(joint_values(solver, original.configuration, "waist")[0] - 0.35) > 0.05


@pytest.mark.parametrize(
    "costs,error",
    [
        ([], TypeError),
        ({"": 1.0}, ValueError),
        ({"waist": -1.0}, ValueError),
        ({"waist": float("nan")}, ValueError),
        ({"waist": float("inf")}, ValueError),
        ({"waist": object()}, TypeError),
    ],
)
def test_joint_cost_input_validation(costs, error):
    with pytest.raises(error):
        PostureTask(joint_costs=costs)


@pytest.mark.parametrize(
    "costs", [{"missing": 1.0}, {"universe": 1.0}, {"waist": 1e300}]
)
def test_solver_rejects_unresolvable_or_overflowing_costs(tmp_path, costs):
    with pytest.raises(ValueError):
        make_solver(tmp_path, joint_motion_costs=costs)
    with pytest.raises(ValueError):
        make_solver(tmp_path, posture_task=PostureTask(joint_costs=costs))


@pytest.mark.parametrize("arm_mode", ["left_arm", "right_arm", "dual_arm"])
@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_torso_first_freezes_other_stage_and_restores_mode(
    tmp_path, arm_mode, solver_type
):
    solver = make_solver(tmp_path, solver_type)
    seed = configuration(solver, aaa_leg=0.15, left=0.05, right=-0.05)
    target = poses(solver, configuration(solver, waist=0.4, left=0.2, right=-0.1))
    hands = {
        name: target[name]
        for name in RetargetingModeManager(initial_mode=arm_mode).spec.targets
    }
    solver.set_mode("torso_dual_arm")
    result = solve_torso_first(
        solver, target["torso"], hands, seed=seed, arm_mode=arm_mode
    )
    assert result.success
    assert solver.mode is RetargetingMode.TORSO_DUAL_ARM
    assert result.torso.mode is RetargetingMode.TORSO
    assert result.arms.mode is RetargetingMode(arm_mode)
    for name in ("left", "right", "aaa_leg"):
        np.testing.assert_array_equal(
            joint_values(solver, result.torso.configuration, name),
            joint_values(solver, seed, name),
        )
    np.testing.assert_array_equal(
        joint_values(solver, result.arms.configuration, "waist"),
        joint_values(solver, result.torso.configuration, "waist"),
    )
    for name, pose in hands.items():
        np.testing.assert_allclose(
            poses(solver, result.arms.configuration)[name], pose, atol=3e-6
        )


def test_torso_failure_does_not_attempt_arm_stage(tmp_path):
    solver = make_solver(tmp_path)
    target = poses(solver, solver._neutral_q)
    target["torso"][2, 3] = 1.0  # A yaw joint cannot change torso height.
    result = solve_torso_first(solver, target["torso"], target, seed=solver._neutral_q)
    assert not result.success
    assert not result.torso.success
    assert result.arms is None
    np.testing.assert_array_equal(solver._last_q, result.torso.configuration)
    assert solver.mode is RetargetingMode.DUAL_ARM


def test_multiseed_candidates_preserve_inactive_manifold_joint(tmp_path):
    solver = make_solver(tmp_path, CuroboRetargetingSolver, continuous=True)
    solver.prepare("left_arm")
    seed = configuration(solver, waist=0.4, left=0.2, right=-0.1, aaa_leg=0.15)
    candidates = solver._seed_bank(seed)
    assert len(candidates) == 3
    for candidate in candidates:
        for name in ("waist", "right", "aaa_leg"):
            np.testing.assert_allclose(
                joint_values(solver, candidate, name),
                joint_values(solver, seed, name),
                atol=1e-15,
                rtol=0.0,
            )


def test_joint_preference_validation_does_not_corrupt_solver_history(tmp_path):
    solver = make_solver(tmp_path)
    solver.prepare("torso_left_arm")
    seed = configuration(solver, waist=0.2)
    solver.reset(seed)
    target = poses(solver, configuration(solver, waist=0.3))
    solver.posture_task = PostureTask(joint_costs={"missing": 0.1})
    with pytest.raises(ValueError, match="unknown joint"):
        solver.prepare("torso_dual_arm")
    assert solver.mode is RetargetingMode.TORSO_LEFT_ARM
    with pytest.raises(ValueError, match="unknown joint"):
        solver.solve(target)
    np.testing.assert_array_equal(solver._last_q, seed)


def test_arm_failure_preserves_successful_torso_stage(tmp_path):
    solver = make_solver(tmp_path)
    target = poses(solver, configuration(solver, waist=0.3))
    target["left_hand"][2, 3] = 1.0
    result = solve_torso_first(solver, target["torso"], target, seed=solver._neutral_q)
    assert not result.success
    assert result.torso.success and not result.arms.success
    np.testing.assert_array_equal(
        joint_values(solver, result.arms.configuration, "waist"),
        joint_values(solver, result.torso.configuration, "waist"),
    )


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_stage_prevalidation_and_exception_restore_history(
    tmp_path, monkeypatch, error_type
):
    solver = make_solver(tmp_path)
    seed = configuration(solver, waist=0.1)
    solver.reset(seed)
    solver._last_velocity[:] = 0.02
    target = poses(solver, configuration(solver, waist=0.3))
    with pytest.raises(ValueError, match="right_hand"):
        solve_torso_first(
            solver, target["torso"], {"left_hand": target["left_hand"]}, seed=seed
        )
    np.testing.assert_array_equal(solver._last_q, seed)
    assert solver.mode is RetargetingMode.DUAL_ARM

    original = solver.solve

    def fail_on_arm(*args, **kwargs):
        if solver.mode is RetargetingMode.DUAL_ARM:
            raise error_type("arm backend failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "solve", fail_on_arm)
    with pytest.raises(error_type, match="arm backend"):
        solve_torso_first(solver, target["torso"], target, seed=seed)
    np.testing.assert_array_equal(solver._last_q, seed)
    np.testing.assert_array_equal(solver._last_velocity, np.full(solver.model.nv, 0.02))
    assert solver.mode is RetargetingMode.DUAL_ARM


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_stage_failure_restores_multiseed_diagnostics(
    tmp_path, monkeypatch, error_type
):
    solver = make_solver(tmp_path, CuroboRetargetingSolver)
    solver.prepare("torso")
    initial = configuration(solver, waist=0.1)
    target = poses(solver, configuration(solver, waist=0.4))
    previous = solver.step({"torso": target["torso"]}, seed=initial)
    history = (
        solver._last_q.copy(),
        solver._last_velocity.copy(),
        solver.last_seed_index,
        solver.last_num_seeds_evaluated,
    )
    assert history[3] == 1
    original = solver.solve

    def fail_on_arm(*args, **kwargs):
        if solver.mode is RetargetingMode.DUAL_ARM:
            assert solver.last_num_seeds_evaluated == 3
            raise error_type("arm backend failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "solve", fail_on_arm)
    with pytest.raises(error_type, match="arm backend"):
        solve_torso_first(solver, target["torso"], target, seed=previous.configuration)
    np.testing.assert_array_equal(solver._last_q, history[0])
    np.testing.assert_array_equal(solver._last_velocity, history[1])
    assert (solver.last_seed_index, solver.last_num_seeds_evaluated) == history[2:]
    assert solver.mode is RetargetingMode.TORSO


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_interrupted_step_restores_velocity_and_can_retry(
    tmp_path, monkeypatch, solver_type, error_type
):
    solver = make_solver(tmp_path, solver_type)
    solver.prepare("torso")
    seed = configuration(solver, waist=0.1)
    solver.reset(seed)
    target = {"torso": poses(solver, configuration(solver, waist=0.4))["torso"]}
    original = solver._task_state
    interruption = error_type("interrupted after accepted update")

    def interrupt_after_update(*args, **kwargs):
        if np.any(solver._last_velocity):
            raise interruption
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(solver, "_task_state", interrupt_after_update)
        with pytest.raises(error_type) as caught:
            solver.step(target)
    assert caught.value is interruption
    np.testing.assert_array_equal(solver._last_q, seed)
    np.testing.assert_array_equal(solver._last_velocity, np.zeros(solver.model.nv))
    if isinstance(solver, CuroboRetargetingSolver):
        assert solver.last_num_seeds_evaluated == 0

    retried = solver.step(target)
    retried_velocity = solver._last_velocity.copy()
    solver.reset(seed)
    expected = solver.step(target)
    assert retried.accepted_steps == expected.accepted_steps == 1
    np.testing.assert_array_equal(retried.configuration, expected.configuration)
    np.testing.assert_array_equal(retried_velocity, solver._last_velocity)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "solver_type",
    [PinocchioRetargetingSolver, PinkRetargetingSolver, CuroboRetargetingSolver],
)
def test_interrupted_prepare_restores_mode(
    tmp_path, monkeypatch, solver_type, error_type
):
    solver = make_solver(tmp_path, solver_type)
    original = solver._solve_workspace

    def interrupt_workspace(plan):
        original(plan)
        raise error_type("workspace interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(solver, "_solve_workspace", interrupt_workspace)
        with pytest.raises(error_type, match="workspace interrupted"):
            solver.prepare("torso")
    assert solver.mode is RetargetingMode.DUAL_ARM
    solver.prepare("torso")
    assert solver.mode is RetargetingMode.TORSO


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_pink_limit_preparation_restores_mode(
    tmp_path, monkeypatch, error_type
):
    solver = make_solver(tmp_path)
    original = solver._mode_limits

    def interrupt_limits():
        original()
        raise error_type("limit preparation interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(solver, "_mode_limits", interrupt_limits)
        with pytest.raises(error_type, match="limit preparation interrupted"):
            solver.prepare("torso")
    assert solver.mode is RetargetingMode.DUAL_ARM
    solver.prepare("torso")
    assert solver.mode is RetargetingMode.TORSO


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
def test_torso_first_requires_explicit_seed(tmp_path, solver_type):
    solver = make_solver(tmp_path, solver_type)
    initial = configuration(solver, waist=0.3, aaa_leg=0.2)
    solver.reset(initial)
    target = poses(solver, initial)
    with pytest.raises(ValueError, match="seed"):
        solve_torso_first(solver, target["torso"], target, seed=None)
    np.testing.assert_array_equal(solver._last_q, initial)


@pytest.mark.parametrize(
    "torso_mode,arm_mode,arm_size",
    [
        (RetargetingMode.LEFT_ARM, RetargetingMode.DUAL_ARM, 2),
        (RetargetingMode.TORSO, RetargetingMode.TORSO_DUAL_ARM, 2),
        (RetargetingMode.TORSO, RetargetingMode.TORSO, 2),
        (RetargetingMode.TORSO, RetargetingMode.DUAL_ARM, 3),
    ],
)
def test_stage_result_rejects_inconsistent_modes_and_dimensions(
    torso_mode, arm_mode, arm_size
):
    torso = RetargetingResult(
        configuration=np.zeros(2),
        success=True,
        iterations=1,
        residual=0.0,
        solve_ms=0.0,
        mode=torso_mode,
    )
    arms = RetargetingResult(
        configuration=np.zeros(arm_size),
        success=True,
        iterations=1,
        residual=0.0,
        solve_ms=0.0,
        mode=arm_mode,
    )
    with pytest.raises(ValueError):
        TorsoFirstResult(torso, arms)


def test_stages_reject_shared_joint_between_torso_and_arm_modes(tmp_path):
    solver = make_solver(tmp_path, joint_motion_costs={"waist": 0.1})
    from holistic_motion.kit.retargeting import RetargetingModeSpec

    specs = dict(solver.mode_manager.mode_specs)
    specs[RetargetingMode.DUAL_ARM] = RetargetingModeSpec(
        ("left_hand", "right_hand"), ("torso", "left_arm", "right_arm")
    )
    solver.mode_manager = RetargetingModeManager(specs)
    target = poses(solver, solver._neutral_q)
    with pytest.raises(ValueError, match="disjoint"):
        solve_torso_first(solver, target["torso"], target, seed=solver._neutral_q)
    assert solver.mode is RetargetingMode.DUAL_ARM
