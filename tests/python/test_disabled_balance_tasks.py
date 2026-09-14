"""Zero-cost balance tasks must not interfere with kinematic-only models."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    CuroboRetargetingSolver,
    PinkRetargetingSolver,
    PostureTask,
    SupportPolygonTask,
    ZmpTask,
)


def make_solver(tmp_path, solver_type, **options):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "kinematic_only.urdf"
    urdf.write_text("""<robot name="kinematic_only">
        <link name="base"/><link name="tool"/>
        <joint name="arm" type="revolute">
          <parent link="base"/><child link="tool"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="1"/>
        </joint></robot>""")
    solver = solver_type(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": ["arm"]},
        posture_task=PostureTask(cost=0.0),
        **options,
    )
    solver.prepare("left_arm")
    return solver


def polygon(cost=0.0, reference="center_of_mass"):
    return SupportPolygonTask(
        [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
        cost=cost,
        reference=reference,
    )


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize(
    "kind", ["com", "zmp", "support_com", "support_zmp", "all", "all_with_targets"]
)
def test_disabled_balance_tasks_do_not_evaluate_mass_or_block_tracking(
    tmp_path,
    monkeypatch,
    solver_type,
    kind,
):
    options = {}
    if kind in ("com", "all", "all_with_targets"):
        options["center_of_mass_task"] = CenterOfMassTask(cost=0.0, lm_damping=1.0)
    if kind in ("zmp", "all", "all_with_targets"):
        options["zmp_task"] = ZmpTask(cost=0.0, lm_damping=1.0)
    if kind in ("support_com", "support_zmp", "all", "all_with_targets"):
        options["support_polygon_task"] = polygon(
            reference="center_of_mass" if kind == "support_com" else "zmp"
        )
    solver = make_solver(tmp_path, solver_type, **options)
    if kind == "all_with_targets":
        solver.set_center_of_mass_target([1.0, 2.0, 3.0])
        solver.set_zmp_target([1.0, 2.0])

    def unexpected_mass_calculation(*args, **kwargs):
        raise AssertionError("disabled task evaluated center of mass")

    monkeypatch.setattr(solver.pin, "centerOfMass", unexpected_mass_calculation)
    monkeypatch.setattr(solver.pin, "jacobianCenterOfMass", unexpected_mass_calculation)
    target = np.eye(4)
    angle = 0.15
    target[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = solver.solve({"left_hand": target}, seed=[0.0])
    assert result.success
    np.testing.assert_allclose(result.configuration, [angle], atol=2e-4)
    assert np.isnan(result.center_of_mass_residual)
    assert np.isnan(result.zmp_residual)
    assert result.support_polygon_violation == 0.0


@pytest.mark.parametrize(
    "solver_type", [PinkRetargetingSolver, CuroboRetargetingSolver]
)
@pytest.mark.parametrize("kind", ["com", "zmp", "support_zmp"])
def test_reenabled_balance_tasks_still_validate_dependencies(
    tmp_path, solver_type, kind
):
    solver = make_solver(tmp_path, solver_type)
    solver.reset([0.2])
    if kind == "com":
        solver.center_of_mass_task = CenterOfMassTask(cost=1.0)
        message = "center-of-mass target"
    elif kind == "zmp":
        solver.zmp_task = ZmpTask(cost=1.0)
        message = "ZMP target"
    else:
        solver.support_polygon_task = polygon(cost=1.0, reference="zmp")
        message = "zmp_task"
    with pytest.raises(ValueError, match=message):
        solver.solve({"left_hand": np.eye(4)})
    np.testing.assert_array_equal(solver._last_q, [0.2])
