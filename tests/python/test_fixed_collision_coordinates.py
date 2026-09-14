"""Numerical collision derivatives need only sample movable coordinates."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("method", ["solve", "step"])
@pytest.mark.parametrize(
    "restriction", ["zero_velocity", "fixed_position", "continuous_zero_velocity"]
)
def test_fixed_waist_is_not_sampled_while_arm_collision_gradient_is_preserved(
    tmp_path, monkeypatch, solver_type, method, restriction
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    continuous = restriction == "continuous_zero_velocity"
    waist_type = "continuous" if continuous else "revolute"
    limits = (
        ""
        if continuous
        else (
            'lower="0" upper="0"'
            if restriction == "fixed_position"
            else 'lower="-1" upper="1"'
        )
    )
    velocity = 1 if restriction == "fixed_position" else 0
    urdf = tmp_path / "fixed_waist.urdf"
    urdf.write_text(f"""<robot name="fixed_waist">
        <link name="base"/><link name="waist"/><link name="tool"/>
        <joint name="waist_joint" type="{waist_type}">
          <parent link="base"/><child link="waist"/><axis xyz="0 0 1"/>
          <limit {limits} velocity="{velocity}" effort="1"/>
        </joint>
        <joint name="arm_joint" type="prismatic">
          <parent link="waist"/><child link="tool"/><axis xyz="1 0 0"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint></robot>""")
    samples = []

    def cost(q):
        samples.append(q.copy())
        waist = np.arctan2(q[1], q[0]) if continuous else q[0]
        return float((waist - 1.0) ** 2 + (q[-1] - 0.3) ** 2)

    def gradient(q):
        waist = np.arctan2(q[1], q[0]) if continuous else q[0]
        return [2.0 * (waist - 1.0), 2.0 * (q[-1] - 0.3)]

    options = {"num_seeds": 1} if solver_type is CuroboRetargetingSolver else {}
    solver = solver_type(
        urdf,
        {"left_hand": "tool"},
        {"torso": ["waist_joint"], "left_arm": ["arm_joint"]},
        frame_tasks={"left_hand": FrameTask(0.0, 0.0)},
        posture_task=PostureTask(cost=0.0),
        collision_cost=cost,
        damping=2.0,
        integration_dt=0.1,
        step_size=1.0,
        max_iterations=1,
        **options,
    )
    solver.prepare("torso_left_arm")
    seed = solver._neutral_q.copy()
    targets = {"left_hand": np.eye(4)}
    solver.collision_gradient = gradient
    reference = getattr(solver, method)(targets, seed=seed)
    solver.reset(seed)
    solver.collision_gradient = None
    samples.clear()
    integrations = []
    original = solver.pin.integrate

    def integrate(model, q, tangent):
        integrations.append(tangent.copy())
        return original(model, q, tangent)

    monkeypatch.setattr(solver.pin, "integrate", integrate)
    result = getattr(solver, method)(targets, seed=seed)

    np.testing.assert_allclose(
        result.configuration, reference.configuration, atol=1e-12
    )
    assert result.configuration[-1] > 0.09
    assert result.objective == pytest.approx(reference.objective)
    assert result.collision_evaluations == len(samples) == 4
    assert result.collision_gradient_evaluations == 1
    # Two arm difference samples and one actual update; none perturb the waist.
    assert len(integrations) == 3
    assert all(tangent[0] == 0.0 for tangent in integrations)
    for q in samples:
        np.testing.assert_allclose(q[:-1], seed[:-1], atol=1e-15)
