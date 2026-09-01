from itertools import product

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    RetargetingMode,
    RetargetingModeManager,
    RetargetingModeSpec,
    SupportPolygonTask,
    ZmpTask,
)


def test_frame_task_supports_scalar_and_anisotropic_costs():
    task = FrameTask(position_cost=[1.0, 2.0, 3.0], orientation_cost=0.5)
    np.testing.assert_allclose(task.cost, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        FrameTask(position_cost=-1.0)
    with pytest.raises(ValueError, match="gain"):
        FrameTask(gain=1.1)
    with pytest.raises(TypeError, match="gain must be numeric"):
        FrameTask(gain=object())
    with pytest.raises(ValueError, match="read-only"):
        task.position_cost[0] = 4.0
    with pytest.raises(ValueError, match="read-only"):
        task.cost[0] = 4.0
    assert task.cost is task.cost


def test_posture_task_validation():
    assert PostureTask().cost > 0.0
    with pytest.raises(ValueError, match="non-negative"):
        PostureTask(cost=-0.1)
    with pytest.raises(TypeError, match="posture cost must be numeric"):
        PostureTask(cost="invalid")


def test_center_of_mass_task_supports_anisotropic_costs():
    task = CenterOfMassTask(cost=[1.0, 2.0, 0.0], gain=0.5)
    np.testing.assert_allclose(task.cost, [1.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        CenterOfMassTask(cost=-1.0)
    with pytest.raises(ValueError, match="gain"):
        CenterOfMassTask(gain=0.0)


def test_support_polygon_task_normalizes_clockwise_vertices():
    task = SupportPolygonTask(
        [[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1]],
        margin=0.01,
    )
    assert task.vertices.shape == (4, 2)
    assert task.normals.shape == (4, 2)
    assert task.offsets.shape == (4,)
    with pytest.raises(ValueError, match="read-only"):
        task.offsets[0] = 1.0
    with pytest.raises(ValueError, match="strictly convex"):
        SupportPolygonTask([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    pentagon = np.column_stack((np.cos(angles), np.sin(angles)))
    with pytest.raises(ValueError, match="strictly convex"):
        SupportPolygonTask(pentagon[[0, 2, 4, 1, 3]])
    with pytest.raises(ValueError, match="reference"):
        SupportPolygonTask(task.vertices, reference="foot")
    with pytest.raises(TypeError, match="margin must be numeric"):
        SupportPolygonTask(task.vertices, margin=object())


def test_zmp_task_supports_anisotropic_costs():
    task = ZmpTask(cost=[1.0, 0.0], gain=0.5, gravity=9.8, plane_height=0.2)
    np.testing.assert_allclose(task.cost, [1.0, 0.0])
    assert task.gravity == 9.8
    assert task.plane_height == 0.2
    with pytest.raises(ValueError, match="non-negative"):
        ZmpTask(cost=-1.0)
    with pytest.raises(ValueError, match="gravity"):
        ZmpTask(gravity=0.0)
    with pytest.raises(ValueError, match="plane_height"):
        ZmpTask(plane_height=float("nan"))


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
    common = (
        urdf,
        {"left_hand": "base", "right_hand": "base", "head": "base"},
        {"left_arm": [], "right_arm": []},
    )
    with pytest.raises(TypeError, match="frame_tasks"):
        PinkRetargetingSolver(*common, frame_tasks={"left_hand": object()})
    with pytest.raises(TypeError, match="mapping"):
        PinkRetargetingSolver(*common, frame_tasks=[])
    with pytest.raises(TypeError, match="posture_task"):
        PinkRetargetingSolver(*common, posture_task=object())
    with pytest.raises(TypeError, match="stagnation_iterations"):
        PinkRetargetingSolver(*common, stagnation_iterations=1.5)
    with pytest.raises(TypeError, match="max_backtracks"):
        PinkRetargetingSolver(*common, max_backtracks=True)
    with pytest.raises(TypeError, match="collision_cost"):
        PinkRetargetingSolver(*common, collision_cost=1.0)
    with pytest.raises(TypeError, match="center_of_mass_task"):
        PinkRetargetingSolver(*common, center_of_mass_task=object())


def test_zero_cost_frame_axes_do_not_block_convergence(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text('<robot name="fixed"><link name="base"/></robot>')
    solver = PinkRetargetingSolver(
        urdf,
        {"left_hand": "base"},
        {"left_arm": []},
        frame_tasks={"left_hand": FrameTask(position_cost=0.0, orientation_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
    )
    solver.prepare("left_arm")
    assert solver.mode in solver._system_workspace
    assert solver.mode in solver._solve_workspaces
    solve_workspace = solver._solve_workspaces[solver.mode]
    target = np.eye(4)
    target[0, 3] = 1.0

    result = solver.solve({"left_hand": target})

    assert solver._solve_workspaces[solver.mode] is solve_workspace
    assert result.success
    assert result.iterations == 1
    assert result.position_residual == pytest.approx(1.0)

    solver.frame_tasks["left_hand"] = FrameTask()
    result = solver.solve({"left_hand": target})
    assert not result.success
    assert result.iterations == 1
    assert result.termination_reason == "no_active_dofs"


def test_zero_cost_frame_error_does_not_change_lm_damping(tmp_path):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "single_joint.urdf"
    urdf.write_text(
        """<robot name="single_joint">
        <link name="base"/><link name="tool"/>
        <joint name="joint" type="revolute">
          <parent link="base"/><child link="tool"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>"""
    )
    manager = RetargetingModeManager(
        {
            RetargetingMode.LEFT_ARM: RetargetingModeSpec(
                ("tracked", "disabled"), ("arm",)
            )
        },
        initial_mode=RetargetingMode.LEFT_ARM,
    )
    solver = PinkRetargetingSolver(
        urdf,
        {"tracked": "tool", "disabled": "tool"},
        {"arm": ["joint"]},
        mode_manager=manager,
        frame_tasks={
            "tracked": FrameTask(position_cost=0.0, lm_damping=0.1),
            "disabled": FrameTask(
                position_cost=0.0, orientation_cost=0.0, lm_damping=1.0
            ),
        },
        posture_task=PostureTask(cost=0.0),
        damping=1e-9,
        step_size=1.0,
        max_iterations=1,
    )
    tracked = np.eye(4)
    angle = 0.4
    tracked[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    nearby_disabled = np.eye(4)
    distant_disabled = np.eye(4)
    distant_disabled[0, 3] = 100.0

    nearby = solver.solve(
        {"tracked": tracked, "disabled": nearby_disabled}, seed=np.zeros(solver.nq)
    )
    distant = solver.solve(
        {"tracked": tracked, "disabled": distant_disabled}, seed=np.zeros(solver.nq)
    )

    np.testing.assert_allclose(distant.configuration, nearby.configuration, atol=1e-12)


def test_box_qp_respects_bounds():
    solution = PinkRetargetingSolver._solve_box_qp(
        np.diag([2.0, 4.0]),
        np.array([4.0, -8.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(solution, [1.0, -1.0])


def test_box_qp_does_not_require_an_eigendecomposition(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("box QP should use the O(n^2) norm bound")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    solution = PinkRetargetingSolver._solve_box_qp(
        np.array([[4.0, 1.0], [1.0, 3.0]]),
        np.array([8.0, -6.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
    )

    assert np.all(solution >= -1.0)
    assert np.all(solution <= 1.0)


def test_box_qp_reuses_supplied_unconstrained_solution(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("box QP repeated the linear solve")

    monkeypatch.setattr(PinkRetargetingSolver, "_linear_solve", fail)
    solution = PinkRetargetingSolver._solve_box_qp(
        np.diag([2.0, 4.0]),
        np.array([4.0, -8.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        unconstrained=np.array([2.0, -2.0]),
    )

    np.testing.assert_allclose(solution, [1.0, -1.0])


def test_box_qp_matches_enumerated_active_sets_for_ill_conditioned_problem():
    generator = np.random.default_rng(42)
    basis, _ = np.linalg.qr(generator.normal(size=(4, 4)))
    hessian = basis @ np.diag([1.0, 100.0, 1e4, 1e7]) @ basis.T
    gradient = np.array([2e5, -4e5, 3e5, -1e5])
    lower = np.array([-0.8, -0.6, -0.4, -0.2])
    upper = np.array([0.3, 0.5, 0.7, 0.9])

    expected = None
    expected_objective = float("inf")
    for status in product((-1, 0, 1), repeat=4):
        status = np.asarray(status)
        free = status == 0
        active = ~free
        candidate = np.zeros(4)
        candidate[status == -1] = lower[status == -1]
        candidate[status == 1] = upper[status == 1]
        if np.any(free):
            rhs = gradient[free] - hessian[np.ix_(free, active)] @ candidate[active]
            candidate[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
        if np.any(candidate < lower - 1e-10) or np.any(candidate > upper + 1e-10):
            continue
        objective = 0.5 * candidate @ hessian @ candidate - gradient @ candidate
        if objective < expected_objective:
            expected = candidate.copy()
            expected_objective = objective

    solution = PinkRetargetingSolver._solve_box_qp(hessian, gradient, lower, upper)

    np.testing.assert_allclose(solution, expected, rtol=1e-9, atol=1e-9)


def test_pink_solver_handles_dual_arm_mode(tmp_path, monkeypatch):
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
    calls = {"joint": 0, "frame": 0}
    original_joint = solver.pin.computeJointJacobians
    original_frame = solver.pin.getFrameJacobian

    def counted_joint(*args):
        calls["joint"] += 1
        return original_joint(*args)

    def counted_frame(*args):
        calls["frame"] += 1
        return original_frame(*args)

    monkeypatch.setattr(solver.pin, "computeJointJacobians", counted_joint)
    monkeypatch.setattr(solver.pin, "getFrameJacobian", counted_frame)
    result = solver.solve(moved)
    assert result.success
    assert result.accepted_steps > 0
    assert calls["frame"] == 2 * calls["joint"]
    left_joint = solver.model.joints[solver.model.getJointId("left_joint")]
    np.testing.assert_allclose(result.configuration[left_joint.idx_q], angle, atol=2e-4)
