import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    CuroboRetargetingSolver,
    FrameTask,
    PostureTask,
    RetargetingResult,
    SupportPolygonTask,
    ZmpTask,
)


def _solver(tmp_path, **kwargs):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "curobo_style.urdf"
    urdf.write_text(
        """<robot name="curobo_style">
        <link name="base"/><link name="left_hand"/>
        <joint name="left_joint" type="revolute">
          <parent link="base"/><child link="left_hand"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    options = {
        "num_seeds": 4,
        "seed_spread": 0.4,
        "sampler_seed": 7,
        "integration_dt": 0.1,
    }
    options.update(kwargs)
    solver = CuroboRetargetingSolver(
        urdf,
        {"left_hand": "left_hand", "right_hand": "base", "head": "base"},
        {"left_arm": ["left_joint"], "right_arm": []},
        **options,
    )
    solver.set_mode("left_arm")
    return solver


def _target(solver, angle):
    q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    pose = np.asarray(solver.data.oMf[solver._frame_ids["left_hand"]].homogeneous)
    pose[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return {"left_hand": pose}


def test_curobo_style_solver_refines_deterministic_seed_bank(tmp_path):
    first = _solver(tmp_path)
    target = _target(first, 0.25)
    first_result = first.solve(target, seed=[-0.8])

    second = _solver(tmp_path)
    second_result = second.solve(target, seed=[-0.8])

    assert first_result.success
    assert first.last_num_seeds_evaluated == 4
    assert 0 <= first.last_seed_index < 4
    np.testing.assert_allclose(first_result.configuration, [0.25], atol=2e-4)
    np.testing.assert_allclose(
        first_result.configuration, second_result.configuration, atol=1e-12
    )
    assert first.last_seed_index == second.last_seed_index


def test_retargeting_configuration_projects_joint_limits(tmp_path):
    solver = _solver(tmp_path, num_seeds=1)

    np.testing.assert_allclose(solver._configuration([5.0]), [1.0])
    np.testing.assert_allclose(solver._configuration([-5.0]), [-1.0])


def test_curobo_style_solver_prepares_target_poses_once_per_solve(
    tmp_path, monkeypatch
):
    solver = _solver(tmp_path)
    target = _target(solver, 0.25)
    original = solver.pin.SE3
    calls = 0

    def counted_se3(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(solver.pin, "SE3", counted_se3)
    result = solver.solve(target, seed=[-0.8])

    assert result.success
    assert solver.last_num_seeds_evaluated == 4
    assert calls == 1


def test_curobo_style_solver_does_not_initialize_rng_during_solve(
    tmp_path, monkeypatch
):
    solver = _solver(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("solve initialized the seed RNG")

    monkeypatch.setattr(np.random, "default_rng", fail)
    result = solver.solve(_target(solver, 0.25), seed=[-0.8])

    assert result.success


def test_curobo_style_solver_can_stop_after_first_success(tmp_path):
    solver = _solver(tmp_path, stop_on_success=True)
    result = solver.solve(_target(solver, 0.0), seed=[0.0])

    assert result.success
    assert solver.last_num_seeds_evaluated == 1
    assert solver.last_seed_index == 0


def test_curobo_style_solver_skips_duplicate_zero_spread_seeds(tmp_path):
    solver = _solver(tmp_path, seed_spread=0.0)
    result = solver.solve(_target(solver, 0.0), seed=[0.0])

    assert result.success
    assert solver.last_num_seeds_evaluated == 1


def test_curobo_style_step_uses_only_physical_primary_seed(tmp_path):
    solver = _solver(tmp_path, acceleration_limits={"left_joint": 0.5})
    result = solver.step(_target(solver, 0.08), seed=[0.0])

    joint = solver.model.joints[solver.model.getJointId("left_joint")]
    assert result.accepted_steps == 1
    assert solver.last_num_seeds_evaluated == 1
    assert abs(solver._last_velocity[joint.idx_v]) <= 0.05 + 1e-12


def test_curobo_style_solver_uses_collision_cost_gradient(tmp_path):
    calls = {"cost": 0, "gradient": 0}

    def collision_cost(q):
        calls["cost"] += 1
        return float((q[0] - 0.5) ** 2)

    def collision_gradient(q):
        calls["gradient"] += 1
        return [2.0 * (q[0] - 0.5)]

    solver = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=collision_cost,
        collision_gradient=collision_gradient,
        collision_cost_weight=10.0,
        collision_tolerance=1e-4,
        orientation_tolerance=1.0,
        posture_task=PostureTask(cost=0.0),
    )
    result = solver.solve(_target(solver, 0.0), seed=[0.0])

    assert result.configuration[0] > 0.4
    assert result.collision_cost < 1e-4
    assert result.collision_evaluations == calls["cost"]
    assert result.collision_gradient_evaluations == calls["gradient"]
    assert calls["gradient"] > 0


def test_curobo_style_solver_uses_combined_collision_query_once(tmp_path):
    calls = {"cost": 0, "combined": 0}

    def collision_cost(_q):
        calls["cost"] += 1
        return 1.0

    def collision_cost_gradient(_q):
        calls["combined"] += 1
        return 1.0, [0.0]

    solver = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=collision_cost,
        collision_cost_gradient=collision_cost_gradient,
        collision_tolerance=1.0,
    )
    result = solver.solve(_target(solver, 0.0), seed=[0.0])

    assert result.success
    assert calls == {"cost": 0, "combined": 1}
    assert result.collision_evaluations == 1
    assert result.collision_gradient_evaluations == 1


def test_curobo_style_solver_finite_differences_collision_cost(tmp_path):
    solver = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda q: float((q[0] - 0.4) ** 2),
        collision_cost_weight=10.0,
        collision_tolerance=1e-4,
        orientation_tolerance=1.0,
        posture_task=PostureTask(cost=0.0),
    )
    result = solver.solve(_target(solver, 0.0), seed=[0.0])

    assert result.configuration[0] > 0.3
    assert result.collision_cost < 1e-4
    assert result.collision_evaluations > result.iterations
    assert result.collision_gradient_evaluations > 0


def test_collision_finite_difference_reuses_cost_at_joint_limit(tmp_path):
    solver = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda q: float((q[0] + 0.5) ** 2),
    )
    q = np.array([-1.0])
    evaluations = [0, 0]
    current_cost = solver._collision_cost_value(q, evaluations)
    gradient = solver._collision_gradient_value(
        q, current_cost, np.array([0]), evaluations
    )

    np.testing.assert_allclose(gradient, [-1.0], atol=2e-4)
    assert evaluations == [2, 1]


def test_collision_finite_difference_uses_continuous_joint_tangent(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "continuous.urdf"
    urdf.write_text(
        """<robot name="continuous">
        <link name="base"/><link name="left_hand"/>
        <joint name="left_joint" type="continuous">
          <parent link="base"/><child link="left_hand"/><axis xyz="0 0 1"/>
          <limit velocity="2" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )

    def collision_cost(q):
        angle = np.arctan2(q[1], q[0])
        return float((angle - 0.4) ** 2)

    solver = CuroboRetargetingSolver(
        urdf,
        {"left_hand": "left_hand", "right_hand": "base", "head": "base"},
        {"left_arm": ["left_joint"], "right_arm": []},
        num_seeds=1,
        collision_cost=collision_cost,
        collision_cost_weight=10.0,
        collision_tolerance=1e-4,
        orientation_tolerance=1.0,
        posture_task=PostureTask(cost=0.0),
    )
    solver.set_mode("left_arm")
    neutral = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    result = solver.solve(_target(solver, 0.0), seed=neutral)

    angle = np.arctan2(result.configuration[1], result.configuration[0])
    assert angle > 0.3
    assert result.collision_cost < 1e-4


def test_curobo_style_solver_validates_collision_callbacks(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        _solver(
            tmp_path,
            collision_cost=lambda _q: 1.0,
            collision_gradient=lambda _q: [0.0],
            collision_cost_gradient=lambda _q: (1.0, [0.0]),
        )
    invalid_cost = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda _q: -1.0,
    )
    with pytest.raises(ValueError, match="collision_cost"):
        invalid_cost.solve(_target(invalid_cost, 0.0), seed=[0.0])

    invalid_gradient = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda _q: 1.0,
        collision_gradient=lambda _q: [0.0, 1.0],
    )
    with pytest.raises(ValueError, match="collision_gradient"):
        invalid_gradient.solve(_target(invalid_gradient, 0.0), seed=[0.0])

    invalid_combined_cost = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda _q: 1.0,
        collision_cost_gradient=lambda _q: (-1.0, [0.0]),
    )
    with pytest.raises(ValueError, match="finite non-negative cost"):
        invalid_combined_cost.solve(_target(invalid_combined_cost, 0.0), seed=[0.0])

    invalid_combined_gradient = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda _q: 1.0,
        collision_cost_gradient=lambda _q: (1.0, [0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="gradient with"):
        invalid_combined_gradient.solve(
            _target(invalid_combined_gradient, 0.0), seed=[0.0]
        )


def test_curobo_style_solver_rejects_weighted_collision_overflow(tmp_path):
    invalid_cost = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda _q: np.finfo(float).max,
        collision_cost_weight=np.finfo(float).max,
    )
    with pytest.raises(ValueError, match="weighted collision cost"):
        invalid_cost.solve(_target(invalid_cost, 0.0), seed=[0.0])

    invalid_gradient = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=lambda _q: 1.0,
        collision_gradient=lambda _q: [np.finfo(float).max],
        collision_cost_weight=np.finfo(float).max,
    )
    with pytest.raises(ValueError, match="weighted collision gradient"):
        invalid_gradient.solve(_target(invalid_gradient, 0.0), seed=[0.0])


def test_collision_cost_cache_is_scoped_to_one_solve(tmp_path):
    calls = 0

    def collision_cost(_q):
        nonlocal calls
        calls += 1
        return 0.0

    solver = _solver(
        tmp_path,
        num_seeds=1,
        collision_cost=collision_cost,
    )
    target = _target(solver, 0.0)
    first = solver.solve(target, seed=[0.0])
    second = solver.solve(target, seed=[0.0])

    assert first.collision_evaluations == 1
    assert second.collision_evaluations == 1
    assert calls == 2


def test_curobo_style_seeds_share_exact_collision_cost_cache(tmp_path, monkeypatch):
    calls = {"cost": 0, "gradient": 0}

    def collision_cost(_q):
        calls["cost"] += 1
        return 1.0

    def collision_gradient(_q):
        calls["gradient"] += 1
        return [0.0]

    solver = _solver(
        tmp_path,
        num_seeds=2,
        collision_cost=collision_cost,
        collision_gradient=collision_gradient,
        collision_tolerance=1.0,
    )
    seed = np.array([0.0])
    monkeypatch.setattr(
        solver, "_seed_bank", lambda _primary: [seed.copy(), seed.copy()]
    )

    result = solver.solve(_target(solver, 0.0), seed=seed)

    assert solver.last_num_seeds_evaluated == 2
    assert result.collision_evaluations == 1
    assert result.collision_gradient_evaluations == 1
    assert calls == {"cost": 1, "gradient": 1}


def test_curobo_style_solver_restores_history_after_seed_failure(tmp_path):
    calls = 0

    def collision_cost(_q):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("later seed failed")
        return 0.0

    solver = _solver(tmp_path, num_seeds=2, collision_cost=collision_cost)
    solver.reset([0.3])
    solver._last_velocity[:] = 0.2

    with pytest.raises(RuntimeError, match="later seed failed"):
        solver.solve(_target(solver, 0.0), seed=[0.0])

    np.testing.assert_allclose(solver._last_q, [0.3])
    np.testing.assert_allclose(solver._last_velocity, [0.2])


def test_curobo_style_seed_ranking_uses_weighted_objective(tmp_path):
    solver = _solver(tmp_path, num_seeds=1)
    mode = solver.mode
    lower_objective = solver._rank(
        RetargetingResult(
            configuration=[0.0],
            success=False,
            iterations=1,
            residual=1.0,
            solve_ms=0.0,
            mode=mode,
            objective=0.1,
            collision_cost=0.09,
        )
    )
    lower_raw_collision = solver._rank(
        RetargetingResult(
            configuration=[0.0],
            success=False,
            iterations=1,
            residual=0.1,
            solve_ms=0.0,
            mode=mode,
            objective=1.0,
            collision_cost=0.01,
        )
    )

    assert lower_objective < lower_raw_collision


def test_curobo_style_solver_supports_dual_leg_only_configuration(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "dual_leg.urdf"
    urdf.write_text(
        """<robot name="dual_leg">
        <link name="base"/><link name="left_foot"/><link name="right_foot"/>
        <joint name="left_leg_joint" type="revolute">
          <parent link="base"/><child link="left_foot"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint>
        <joint name="right_leg_joint" type="revolute">
          <parent link="base"/><child link="right_foot"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    solver = CuroboRetargetingSolver(
        urdf,
        {"left_foot": "left_foot", "right_foot": "right_foot"},
        {
            "left_leg": ["left_leg_joint"],
            "right_leg": ["right_leg_joint"],
        },
        num_seeds=2,
    )
    solver.set_mode("dual_leg")
    q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    targets = {}
    for name, angle in (("left_foot", 0.15), ("right_foot", -0.2)):
        pose = np.asarray(solver.data.oMf[solver._frame_ids[name]].homogeneous)
        pose[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        targets[name] = pose

    result = solver.solve(targets, seed=q)

    assert result.success
    np.testing.assert_allclose(result.configuration, [0.15, -0.2], atol=2e-4)


def test_mode_requirements_are_checked_lazily(tmp_path):
    solver = _solver(tmp_path, num_seeds=1)
    solver.set_mode("dual_leg")
    with pytest.raises(ValueError, match="frame mappings"):
        solver.solve({"left_foot": np.eye(4), "right_foot": np.eye(4)}, seed=[0.0])


def test_curobo_style_solver_tracks_center_of_mass_without_mink(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "center_of_mass.urdf"
    urdf.write_text(
        """<robot name="center_of_mass">
        <link name="base">
          <inertial><mass value="1"/><origin xyz="0 0 0"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
          </inertial>
        </link>
        <link name="left_foot">
          <inertial><mass value="1"/><origin xyz="0 0 0"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
          </inertial>
        </link>
        <joint name="left_leg_joint" type="prismatic">
          <parent link="base"/><child link="left_foot"/><axis xyz="1 0 0"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    solver = CuroboRetargetingSolver(
        urdf,
        {"left_foot": "left_foot"},
        {"left_leg": ["left_leg_joint"]},
        num_seeds=1,
        frame_tasks={"left_foot": FrameTask(position_cost=0.0, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        center_of_mass_task=CenterOfMassTask(cost=[1.0, 0.0, 0.0]),
        center_of_mass_tolerance=1e-5,
        position_tolerance=1.0,
        integration_dt=0.2,
    )
    solver.set_mode("left_leg")
    q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    foot_target = np.asarray(
        solver.data.oMf[solver._frame_ids["left_foot"]].homogeneous
    )
    with pytest.raises(ValueError, match="requires a center-of-mass target"):
        solver.solve({"left_foot": foot_target}, seed=q)
    with pytest.raises(ValueError, match="three finite"):
        solver.set_center_of_mass_target([0.0, 0.0])
    solver.set_center_of_mass_target([0.2, 0.0, 0.0])

    result = solver.solve({"left_foot": foot_target}, seed=q)

    assert result.success
    np.testing.assert_allclose(result.configuration, [0.2], atol=2e-4)
    assert result.center_of_mass_residual <= 1e-5


def test_curobo_style_solver_moves_com_inside_support_polygon(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "support_polygon.urdf"
    urdf.write_text(
        """<robot name="support_polygon">
        <link name="base"/>
        <link name="left_foot">
          <inertial><mass value="1"/><origin xyz="0 0 0"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
          </inertial>
        </link>
        <joint name="left_leg_joint" type="prismatic">
          <parent link="base"/><child link="left_foot"/><axis xyz="1 0 0"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    solver = CuroboRetargetingSolver(
        urdf,
        {"left_foot": "left_foot"},
        {"left_leg": ["left_leg_joint"]},
        num_seeds=1,
        frame_tasks={"left_foot": FrameTask(position_cost=0.0, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        support_polygon_task=SupportPolygonTask(
            [[0.15, -0.1], [0.25, -0.1], [0.25, 0.1], [0.15, 0.1]],
            cost=10.0,
        ),
        support_polygon_tolerance=1e-5,
        position_tolerance=1.0,
        integration_dt=0.2,
    )
    solver.set_mode("left_leg")
    q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    foot_target = np.asarray(
        solver.data.oMf[solver._frame_ids["left_foot"]].homogeneous
    )

    result = solver.solve({"left_foot": foot_target}, seed=q)

    assert result.success
    assert 0.149 <= result.configuration[0] <= 0.251
    assert result.support_polygon_violation <= 1e-5

    solver.support_polygon_task = SupportPolygonTask(
        [[0.15, -0.1], [0.25, -0.1], [0.25, 0.1], [0.15, 0.1]],
        cost=0.0,
    )
    solver.reset(q)
    disabled_result = solver.solve({"left_foot": foot_target}, seed=q)
    assert disabled_result.success
    assert disabled_result.iterations == 1


def test_curobo_style_solver_tracks_kinematic_zmp(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "zmp.urdf"
    urdf.write_text(
        """<robot name="zmp">
        <link name="base"/>
        <link name="left_foot">
          <inertial><mass value="1"/><origin xyz="0 0 0"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
          </inertial>
        </link>
        <joint name="left_leg_joint" type="prismatic">
          <parent link="base"/><child link="left_foot"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    solver = CuroboRetargetingSolver(
        urdf,
        {"left_foot": "left_foot"},
        {"left_leg": ["left_leg_joint"]},
        num_seeds=1,
        frame_tasks={"left_foot": FrameTask(position_cost=0.0, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        zmp_task=ZmpTask(cost=[1.0, 0.0], plane_height=0.05),
        zmp_tolerance=1e-5,
        position_tolerance=1.0,
        integration_dt=0.2,
    )
    solver.set_mode("left_leg")
    q = np.asarray(solver.pin.neutral(solver.model), dtype=float)
    solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    foot_target = np.asarray(
        solver.data.oMf[solver._frame_ids["left_foot"]].homogeneous
    )
    with pytest.raises(ValueError, match="requires a ZMP target"):
        solver.solve({"left_foot": foot_target}, seed=q)
    with pytest.raises(ValueError, match="two finite"):
        solver.set_zmp_target([0.0])
    with pytest.raises(ValueError, match="three finite"):
        solver.set_center_of_mass_acceleration([1.0, 0.0])
    solver.set_center_of_mass_acceleration([9.81, 0.0, 0.0])
    # The zero-cost Y axis is telemetry-only and must not block convergence.
    solver.set_zmp_target([-0.1, 5.0])

    result = solver.solve({"left_foot": foot_target}, seed=q)

    assert result.success
    np.testing.assert_allclose(result.configuration, [0.15], atol=2e-4)
    assert result.zmp_residual > 4.0

    support_solver = CuroboRetargetingSolver(
        urdf,
        {"left_foot": "left_foot"},
        {"left_leg": ["left_leg_joint"]},
        num_seeds=1,
        frame_tasks={"left_foot": FrameTask(position_cost=0.0, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        zmp_task=ZmpTask(cost=0.0, plane_height=0.05),
        support_polygon_task=SupportPolygonTask(
            [[-0.12, -0.1], [-0.08, -0.1], [-0.08, 0.1], [-0.12, 0.1]],
            cost=10.0,
            reference="zmp",
        ),
        support_polygon_tolerance=1e-5,
        position_tolerance=1.0,
        integration_dt=0.2,
    )
    support_solver.set_mode("left_leg")
    support_solver.set_center_of_mass_acceleration([9.81, 0.0, 0.0])
    support_result = support_solver.solve({"left_foot": foot_target}, seed=q)

    assert support_result.success
    assert 0.129 <= support_result.configuration[0] <= 0.171
    assert support_result.support_polygon_violation <= 1e-5


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"num_seeds": 0}, ValueError),
        ({"num_seeds": True}, TypeError),
        ({"seed_spread": -1.0}, ValueError),
        ({"sampler_seed": False}, TypeError),
        ({"sampler_seed": -1}, ValueError),
        ({"stop_on_success": "yes"}, TypeError),
    ],
)
def test_curobo_style_solver_validates_seed_options(tmp_path, kwargs, exception):
    with pytest.raises(exception):
        _solver(tmp_path, **kwargs)


def test_curobo_style_solver_rejects_fractional_iteration_budget(tmp_path):
    solver = _solver(tmp_path, num_seeds=1)
    with pytest.raises(TypeError, match="max_iterations"):
        solver.solve(_target(solver, 0.0), max_iterations=1.5)
