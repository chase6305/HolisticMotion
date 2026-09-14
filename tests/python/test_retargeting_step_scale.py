"""Offline line-search scaling must preserve the QP displacement bounds."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


def make_solver(tmp_path, solver_type, continuous, step_size, damping=1e-6):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "scaled_waist.urdf"
    kind = "continuous" if continuous else "revolute"
    limits = "" if continuous else 'lower="-2" upper="2"'
    urdf.write_text(f"""<robot name="scaled_waist">
      <link name="base"/><link name="torso"/>
      <joint name="waist" type="{kind}">
        <parent link="base"/><child link="torso"/><axis xyz="0 0 1"/>
        <limit {limits} velocity="1" effort="1"/>
      </joint></robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = solver_type(
        urdf,
        {"torso": "torso"},
        {"torso": ["waist"]},
        frame_tasks={"torso": FrameTask(position_cost=0.0)},
        posture_task=PostureTask(cost=0.0),
        step_size=step_size,
        damping=damping,
        integration_dt=0.1,
        **options,
    )
    solver.prepare("torso")
    return solver


def target(angle):
    pose = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return {"torso": pose}


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("step_size", [2.0, 1e4])
def test_scaled_offline_update_stays_within_velocity_bounds(
    tmp_path, solver_type, continuous, direction, step_size
):
    solver = make_solver(tmp_path, solver_type, continuous, step_size)
    seed = solver._neutral_q.copy()
    result = solver.solve(target(direction * 0.8), seed=seed, max_iterations=1)
    displacement = solver.pin.difference(solver.model, seed, result.configuration)

    np.testing.assert_allclose(displacement, [direction * 0.1], atol=1e-12)
    assert result.accepted_steps == 1
    assert result.orientation_residual == pytest.approx(0.7)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_limited_scale_preserves_the_multijoint_search_direction(
    tmp_path, solver_type, direction
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "two_slides.urdf"
    urdf.write_text("""<robot name="two_slides">
      <link name="base"/><link name="x"/><link name="tool"/>
      <joint name="x_joint" type="prismatic">
        <parent link="base"/><child link="x"/><axis xyz="1 0 0"/>
        <limit lower="-2" upper="2" velocity="1" effort="1"/>
      </joint>
      <joint name="y_joint" type="prismatic">
        <parent link="x"/><child link="tool"/><axis xyz="0 1 0"/>
        <limit lower="-2" upper="2" velocity="1" effort="1"/>
      </joint></robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = solver_type(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": ["x_joint", "y_joint"]},
        posture_task=PostureTask(cost=0.0),
        step_size=2.0,
        damping=9.0,
        integration_dt=0.1,
        **options,
    )
    solver.prepare("left_arm")
    pose = np.eye(4)
    pose[:2, 3] = direction * np.array([0.8, 0.2])
    result = solver.solve({"left_hand": pose}, seed=[0.0, 0.0], max_iterations=1)
    # QP direction [0.08, 0.02] permits a common scale of 1.25. Independent
    # clipping would instead return [0.1, 0.04] and change that direction.
    np.testing.assert_allclose(
        result.configuration, direction * np.array([0.1, 0.025]), atol=1e-12
    )


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_backtracking_starts_from_the_feasible_scale(tmp_path, solver_type, direction):
    solver = make_solver(tmp_path, solver_type, False, 1e4, damping=9.0)
    solver.collision_cost = lambda q: float(10.0 * q[0] ** 2)
    solver.max_backtracks = 1
    result = solver.solve(target(direction * 0.8), seed=[0.0], max_iterations=1)
    # The feasible 0.1 step raises the total cost; its half-step lowers it.
    # Backtracking the unbounded scale would exhaust this budget without moving.
    np.testing.assert_allclose(result.configuration, [direction * 0.05], atol=1e-12)
    assert result.accepted_steps == 1
    assert result.objective < 0.5 * 0.8**2
    assert result.collision_evaluations == 3
    assert result.orientation_residual == pytest.approx(0.75)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_scaling_above_one_is_retained_when_bounds_allow_it(
    tmp_path, solver_type, direction
):
    solver = make_solver(tmp_path, solver_type, False, 2.0, damping=9.0)
    result = solver.solve(target(direction * 0.2), seed=[0.0], max_iterations=1)
    # Unscaled QP displacement is 0.02; doubling it remains inside the 0.1 bound.
    np.testing.assert_allclose(result.configuration, [direction * 0.04], atol=1e-12)
    assert result.accepted_steps == 1
