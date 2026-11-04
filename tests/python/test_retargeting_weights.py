"""Weighted targets keep their identity across solves and mode changes."""

import math

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    PinocchioRetargetingSolver,
    RetargetingMode,
    RetargetingModeManager,
    RetargetingModeSpec,
    RetargetingTarget,
)
from optional_dependencies import python_pinocchio_or_skip


def make_solver(tmp_path, names, *, movable=True, iterations=1):
    python_pinocchio_or_skip()
    urdf = tmp_path / "weighted_targets.urdf"
    urdf.write_text("""<robot name="weighted_targets">
      <link name="base"/><link name="tool"/>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="tool"/><axis xyz="1 0 0"/>
        <limit lower="-10" upper="10" velocity="2" effort="10"/>
      </joint></robot>""")
    modes = RetargetingModeManager(
        {
            RetargetingMode.WHOLE_BODY: RetargetingModeSpec(names, ("arm",)),
            RetargetingMode.LEFT_ARM: RetargetingModeSpec((names[-1],), ("arm",)),
        },
        initial_mode=RetargetingMode.WHOLE_BODY,
    )
    return PinocchioRetargetingSolver(
        urdf,
        dict.fromkeys(names, "tool"),
        {"arm": ["slide"] if movable else []},
        mode_manager=modes,
        damping=0.125,
        step_size=0.5,
        tolerance=1e-12,
        max_iterations=iterations,
    )


def targets(names, positions, weights):
    result = {}
    # Caller insertion order intentionally differs from the mode order.
    for name, position, weight in reversed(list(zip(names, positions, weights))):
        pose = np.eye(4)
        pose[0, 3] = position
        result[name] = RetargetingTarget(pose, weight)
    return result


@pytest.mark.parametrize("count", [1, 2, 3, 6])
@pytest.mark.parametrize("movable", [False, True])
@pytest.mark.parametrize("iterations", [1, 4])
def test_weighted_prismatic_solution_matches_scalar_recurrence(
    tmp_path, count, movable, iterations
):
    names = tuple(f"target{i}" for i in range(count))
    solver = make_solver(tmp_path, names, movable=movable, iterations=iterations)
    positions = np.linspace(-0.7, 0.9, count)
    seed = np.array([0.25])
    for weights in (np.geomspace(0.125, 16.0, count), np.geomspace(8.0, 0.25, count)):
        desired = targets(names, positions, weights)
        result = solver.solve(desired, seed=seed)
        expected = float(seed[0])
        if movable:
            for _ in range(iterations):
                correction = sum(
                    weight * (position - expected)
                    for position, weight in zip(positions, weights)
                ) / (sum(weights) + solver.damping)
                expected += solver.step_size * correction
        residual = math.hypot(
            *(
                math.sqrt(weight) * (position - expected)
                for position, weight in zip(positions, weights)
            )
        )
        assert result.configuration[0] == pytest.approx(expected, abs=1e-14)
        assert result.residual == pytest.approx(residual, abs=1e-14)
        assert result.objective == pytest.approx(0.5 * residual**2, abs=1e-14)
        assert not result.success
        assert result.iterations == (iterations if movable else 1)
        np.testing.assert_array_equal(seed, [0.25])
        np.testing.assert_array_equal(solver._last_q, result.configuration)
        for name, position, weight in zip(names, positions, weights):
            assert desired[name].pose[0, 3] == position
            assert desired[name].weight == weight
            assert not desired[name].pose.flags.writeable


def test_weights_are_refreshed_after_mode_changes_and_warm_starts(tmp_path):
    names = ("first", "second", "third")
    solver = make_solver(tmp_path, names)
    for mode in ("whole_body", "left_arm", "whole_body"):
        solver.set_mode(mode)
        active_names = names if mode == "whole_body" else names[-1:]
        weights = np.linspace(0.25, 4.0, len(active_names))
        positions = np.linspace(-0.5, 0.8, len(active_names))
        initial = solver._last_q[0]
        desired = targets(active_names, positions, weights)
        correction = sum(weights * (positions - initial)) / (
            sum(weights) + solver.damping
        )
        result = solver.solve(desired)
        assert result.configuration[0] == pytest.approx(
            initial + solver.step_size * correction, abs=1e-14
        )


def test_targets_keep_their_names_when_mapping_reads_switch_the_mode(tmp_path):
    names = ("first", "second", "third")
    solver = make_solver(tmp_path, names)

    class SwitchingTargets(dict):
        def __getitem__(self, name):
            solver.set_mode("left_arm")
            return super().__getitem__(name)

    desired = SwitchingTargets(targets(names, [-0.5, 0.1, 0.8], [0.25, 2.0, 4.0]))
    result = solver.solve(desired, seed=[0.0])
    expected = solver.step_size * 4.0 * 0.8 / (4.0 + solver.damping)
    assert result.mode is RetargetingMode.LEFT_ARM
    assert result.configuration[0] == pytest.approx(expected, abs=1e-14)
    assert result.residual == pytest.approx(2.0 * (0.8 - expected), abs=1e-14)
