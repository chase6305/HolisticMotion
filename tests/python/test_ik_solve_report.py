"""The common report keeps algorithm, failure and output ownership explicit."""

import numpy as np
import pytest

from test_native_kinematics_inputs import solver_data  # noqa: F401


@pytest.mark.parametrize("selection", ["NEAREST", "ALL"])
def test_common_ik_report_round_trip_and_seed_ownership(solver_data, selection):
    import holistic_motion as hm

    solver, joints = solver_data
    target = solver.forward(joints)
    seed = joints.copy()
    report = solver.solve_ik(target, seed, getattr(hm.IKSelection, selection))
    assert report.success and report.status == hm.IKSolveStatus.SUCCESS
    assert report.algorithm == solver.capabilities.default_ik_algorithm
    assert report.selection == getattr(hm.IKSelection, selection)
    assert len(report.solutions) == len(report.distances) >= 1
    if selection == "NEAREST":
        assert len(report.solutions) == 1
    assert np.isfinite(report.distances).all()
    assert np.all(np.asarray(report.distances) >= 0)
    np.testing.assert_array_equal(seed, joints)
    for solution in report.solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-7)

    # Reports own their outputs, rather than retaining the next call's seed or
    # a solver workspace. Python exposes those outputs as read-only arrays.
    original = report.solutions[0].copy()
    seed.fill(np.nan)
    with pytest.raises(ValueError, match="read-only"):
        report.solutions[0][0] = np.nan
    solver.solve_ik(target, joints)
    np.testing.assert_array_equal(report.solutions[0], original)
    retained = report.solutions
    del report
    np.testing.assert_array_equal(retained[0], original)


@pytest.mark.parametrize(
    "defect", ["seed_size", "seed_nan", "last_row", "rotation", "target_nan"]
)
def test_common_ik_report_distinguishes_invalid_input(solver_data, defect):
    import holistic_motion as hm

    solver, joints = solver_data
    target = solver.forward(joints)
    if defect == "seed_size":
        joints = joints[:-1]
    elif defect == "seed_nan":
        joints[0] = np.nan
    elif defect == "last_row":
        target[3, 3] = 2
    elif defect == "rotation":
        target[:3, :3] *= 0.5
    else:
        target[0, 3] = np.nan
    report = solver.solve_ik(target, joints)
    assert not report.success
    assert report.status == hm.IKSolveStatus.INVALID_INPUT
    assert report.solutions == [] and report.distances == []
    assert report.algorithm == solver.capabilities.default_ik_algorithm


def test_common_ik_report_distinguishes_unreachable_target(solver_data):
    import holistic_motion as hm

    solver, joints = solver_data
    target = solver.forward(joints)
    target[0, 3] += 100
    report = solver.solve_ik(target, joints)
    assert not report.success
    assert report.status == hm.IKSolveStatus.NO_SOLUTION
    assert report.solutions == [] and report.distances == []


@pytest.mark.parametrize("argument", ["seed", "target"])
def test_common_ik_report_rejects_complex_transport_inputs(solver_data, argument):
    solver, joints = solver_data
    target = solver.forward(joints)
    if argument == "target":
        target = target.astype(complex)
    else:
        joints = joints.astype(complex)
    with pytest.raises(TypeError, match="real"):
        solver.solve_ik(target, joints)
