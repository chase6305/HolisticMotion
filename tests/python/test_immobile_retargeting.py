"""Immobile modes need diagnostics, but no Jacobians or collision gradients."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    PinkRetargetingSolver,
    PostureTask,
)


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("method", ["solve", "step"])
@pytest.mark.parametrize("restriction", ["velocity", "position", "empty_group"])
@pytest.mark.parametrize("reached", [False, True])
def test_immobile_mode_skips_derivatives_and_releases_arm_mode(
    tmp_path, monkeypatch, solver_type, method, restriction, reached
):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    velocity = 0 if restriction == "velocity" else 2
    lower, upper = (0, 0) if restriction == "position" else (-2, 2)
    urdf = tmp_path / "locked_torso.urdf"
    urdf.write_text(f"""<robot name="locked_torso">
        <link name="base"/><link name="torso"/><link name="tool"/>
        <joint name="waist" type="prismatic">
          <parent link="base"/><child link="torso"/><axis xyz="1 0 0"/>
          <limit lower="{lower}" upper="{upper}" velocity="{velocity}" effort="1"/>
        </joint>
        <joint name="arm" type="prismatic">
          <parent link="torso"/><child link="tool"/><axis xyz="1 0 0"/>
          <limit lower="-2" upper="2" velocity="2" effort="1"/>
        </joint></robot>""")
    calls = []

    def cost(q):
        calls.append(q.copy())
        return 0.0 if reached else 1.0

    def unexpected(*args, **kwargs):
        raise AssertionError("immobile mode must not evaluate derivatives")

    solver = solver_type(
        urdf,
        {"torso": "torso", "left_hand": "tool"},
        {
            "torso": [] if restriction == "empty_group" else ["waist"],
            "left_arm": ["arm"],
        },
        posture_task=PostureTask(cost=0.0),
        collision_cost=cost,
        collision_cost_gradient=unexpected,
        step_size=1.0,
        integration_dt=0.1,
    )
    solver.prepare("torso")
    # A previous inactive-arm command must be cleared by a hold step, while an
    # offline diagnostic solve leaves the command history unchanged.
    solver._last_velocity[-1] = 0.2
    target = np.eye(4)
    target[0, 3] = 0.0 if reached else 0.5
    original_fk = solver.pin.forwardKinematics
    fk_calls = []

    def forward(*args):
        fk_calls.append(1)
        return original_fk(*args)

    with monkeypatch.context() as patch:
        patch.setattr(solver.pin, "computeJointJacobians", unexpected)
        patch.setattr(solver.pin, "forwardKinematics", forward)
        patch.setattr(solver, "_solve_box_qp", unexpected)
        result = getattr(solver, method)({"torso": target}, seed=[0.0, 0.0])

    np.testing.assert_array_equal(result.configuration, [0.0, 0.0])
    assert result.success == reached
    reason = "no_active_dofs" if restriction == "empty_group" else "no_feasible_motion"
    assert result.termination_reason == ("converged" if reached else reason)
    assert result.iterations == 1
    assert result.accepted_steps == 0
    assert result.position_residual == pytest.approx(0.0 if reached else 0.5)
    assert result.collision_cost == (0.0 if reached else 1.0)
    assert result.collision_evaluations == 1
    assert result.collision_gradient_evaluations == 0
    assert len(calls) == len(fk_calls) == 1
    np.testing.assert_array_equal(
        solver._last_velocity, [0.0, 0.0 if method == "step" else 0.2]
    )

    # The immutable mode caches must not make subsequently enabled arm DOFs
    # look locked or suppress their Jacobians.
    solver.collision_cost_weight = 0.0
    solver.prepare("left_arm")
    target[0, 3] = 0.1
    arm_result = solver.step({"left_hand": target}, seed=result.configuration)
    assert arm_result.configuration[0] == 0.0
    assert arm_result.configuration[1] > 0.09
