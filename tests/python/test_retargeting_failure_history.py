"""Failed result publication must not change the next IK warm start."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import PinocchioRetargetingSolver


def make_solver(tmp_path, *, movable=True):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "history.urdf"
    urdf.write_text("""<robot name="history">
      <link name="base"/><link name="tool"/>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="tool"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/>
      </joint></robot>""")
    solver = PinocchioRetargetingSolver(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": ["slide"] if movable else []},
        max_iterations=1,
        step_size=0.5,
    )
    solver.prepare("left_arm")
    solver.reset([0.25])
    return solver


def target(position):
    pose = np.eye(4)
    pose[0, 3] = position
    return {"left_hand": pose}


@pytest.mark.parametrize("movable", [False, True])
def test_large_finite_residual_returns_diagnostics_and_updates_warm_start(
    tmp_path, movable
):
    solver = make_solver(tmp_path, movable=movable)
    # Squaring this finite translation overflows, while its Euclidean norm is
    # representable. Publish the residual; NaN marks an unavailable squared cost.
    result = solver.solve(target(1e200), seed=[0.5])
    assert not result.success
    assert result.residual == pytest.approx(1e200)
    assert np.isnan(result.objective)
    np.testing.assert_array_equal(solver._last_q, result.configuration)
    recovered = solver.solve(target(result.configuration[0]))
    assert recovered.success
    np.testing.assert_array_equal(recovered.configuration, result.configuration)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, MemoryError])
def test_aborted_result_construction_preserves_warm_start(
    tmp_path, monkeypatch, error_type
):
    from holistic_motion.kit.retargeting import pinocchio_solver

    solver = make_solver(tmp_path)
    previous = solver._last_q.copy()
    candidates = []

    def abort(**values):
        candidates.append(values["configuration"].copy())
        raise error_type("result publication aborted")

    with monkeypatch.context() as patch:
        patch.setattr(pinocchio_solver, "RetargetingResult", abort)
        with pytest.raises(error_type, match="result publication aborted"):
            solver.solve(target(0.8), seed=[0.5])
    assert len(candidates) == 1
    assert candidates[0][0] > 0.5  # A real IK update occurred before the failure.
    np.testing.assert_array_equal(solver._last_q, previous)
    recovered = solver.solve(target(previous[0]))
    assert recovered.success
    np.testing.assert_array_equal(recovered.configuration, previous)


def test_returned_budget_exhaustion_still_updates_warm_start(tmp_path):
    solver = make_solver(tmp_path)
    result = solver.solve(target(0.8), seed=[0.5])
    assert not result.success
    assert 0.5 < result.configuration[0] < 0.8
    np.testing.assert_array_equal(solver._last_q, result.configuration)
    resumed = solver.solve(target(result.configuration[0]))
    assert resumed.success
    np.testing.assert_array_equal(resumed.configuration, result.configuration)
