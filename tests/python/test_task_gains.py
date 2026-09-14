"""Task gains must agree between the QP direction and line-search objective."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    SupportPolygonTask,
    ZmpTask,
)


def make_solver(tmp_path, solver_type, kind, gain, posture_gain):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "sliding_mass.urdf"
    urdf.write_text("""<robot name="sliding_mass">
        <link name="base"/><link name="tool"><inertial>
          <mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
        </inertial></link>
        <joint name="slide" type="prismatic">
          <parent link="base"/><child link="tool"/><axis xyz="1 0 0"/>
          <limit lower="-2" upper="2" velocity="2" effort="1"/>
        </joint></robot>""")
    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    options["frame_tasks"] = {
        "left_hand": FrameTask(
            position_cost=1.0 if kind == "frame" else 0.0,
            orientation_cost=0.0,
            gain=gain,
        )
    }
    if kind == "com":
        options["center_of_mass_task"] = CenterOfMassTask(gain=gain)
    if kind == "zmp":
        options["zmp_task"] = ZmpTask(gain=gain)
    if kind.startswith("support"):
        options["support_polygon_task"] = SupportPolygonTask(
            [[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0]],
            gain=gain,
            reference="zmp" if kind == "support_zmp" else "center_of_mass",
        )
        if kind == "support_zmp":
            options["zmp_task"] = ZmpTask(cost=0.0)
    solver = solver_type(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": ["slide"]},
        posture_task=PostureTask(cost=1.0, gain=posture_gain),
        damping=1e-9,
        step_size=1.0,
        integration_dt=1.0,
        stagnation_tolerance=1e-14,
        max_iterations=200,
        **options,
    )
    solver.prepare("left_arm")
    if kind == "com":
        solver.set_center_of_mass_target([1.0, 0.0, 0.0])
    if kind == "zmp":
        solver.set_zmp_target([1.0, 0.0])
    target = np.eye(4)
    target[0, 3] = 1.0
    return solver, {"left_hand": target}


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("kind", ["frame", "com", "zmp", "support_com", "support_zmp"])
@pytest.mark.parametrize("gain,posture_gain,seed", [(0.1, 1.0, 0.02), (1.0, 0.1, 0.55)])
def test_offline_solve_accepts_the_gain_weighted_descent_direction(
    tmp_path, solver_type, kind, gain, posture_gain, seed
):
    solver, targets = make_solver(tmp_path, solver_type, kind, gain, posture_gain)
    # Both tasks have unit cost and opposite goals. Gains scale their desired
    # corrections; the QP Hessian remains 2 + damping.
    expected_step = seed + (gain * (1.0 - seed) - posture_gain * seed) / (
        2.0 + solver.damping
    )
    step = solver.step(targets, seed=[seed])
    np.testing.assert_allclose(step.configuration, [expected_step], atol=1e-12)
    result = solver.solve(targets, seed=[seed])

    expected = gain / (gain + posture_gain)
    np.testing.assert_allclose(result.configuration, [expected], atol=1e-6)
    assert result.accepted_steps > 0
    # A compromise between conflicting tasks must not be labeled converged.
    assert not result.success
    q = float(result.configuration[0])
    def objective(x):
        return 0.5 * (gain * (1.0 - x) ** 2 + posture_gain * x**2)

    assert result.objective == pytest.approx(objective(q), abs=1e-12)
    assert result.objective < objective(seed)
    assert result.residual == pytest.approx(abs(1.0 - q))
