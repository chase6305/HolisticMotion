"""Inspect solver operations without assuming that a seven-axis facade is analytic."""

import numpy as np
import pytest

from test_srs_kinematics import _write_ideal_srs_urdf, _write_seven_revolute_urdf


@pytest.mark.parametrize("ideal", [False, True])
def test_srs_capabilities_distinguish_default_ik_and_geometry(tmp_path, ideal):
    import holistic_motion as hm

    urdf = tmp_path / "srs.urdf"
    (_write_ideal_srs_urdf if ideal else _write_seven_revolute_urdf)(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    info = solver.capabilities
    assert isinstance(solver, hm.KinematicsBase)
    assert hm.KinematicsBase is hm.AnalyticKinematics
    assert info.family == hm.KinematicsFamily.SRS
    assert info.default_ik_algorithm == hm.IKAlgorithm.NUMERICAL
    assert info.dof == 7
    assert info.supports_closed_form_ik is ideal
    assert info.supports_jacobian
    assert info.supports_redundancy
    assert info.supports_concurrent_queries
    assert not info.supports_batch_fk
    assert not info.cuda_compiled

    # Capability inspection and caller-specific geometry tolerances leave the
    # actual common numerical entry point unchanged.
    joints = np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15])
    target = solver.forward(joints)
    solver.analyze_geometry(1.0)
    solution = solver.inverse(target, joints)
    np.testing.assert_allclose(solver.forward(solution), target, atol=1e-8)
    assert solver.capabilities.supports_closed_form_ik is ideal
    with pytest.raises(AttributeError):
        info.dof = 1


def test_fep_capabilities_do_not_claim_analytic_ik(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "fep.urdf"
    _write_seven_revolute_urdf(urdf)
    solver = hm.Robot(str(urdf)).create_fep_kinematics("link0", "link7")
    info = solver.capabilities
    assert info.family == hm.KinematicsFamily.FEP
    assert info.default_ik_algorithm == hm.IKAlgorithm.NUMERICAL
    assert info.supports_jacobian and info.supports_redundancy
    assert info.supports_batch_fk and info.supports_concurrent_queries
    assert not info.supports_closed_form_ik
    assert info.cuda_compiled == hm.FEPKinematics.cuda_compiled
    joints = np.zeros((2, 7))
    poses = solver.forward_batch(joints, hm.FEPBackend.CPU)
    np.testing.assert_allclose(poses[0], solver.forward(joints[0]))


@pytest.mark.parametrize("family", ["OPW", "UR"])
def test_analytic_capabilities_are_available_through_the_common_base(family):
    import holistic_motion as hm

    parameters = getattr(hm, family + "Parameters")()
    parameters.rotation_directions = [1] * 6
    if family == "OPW":
        parameters.c1, parameters.c2 = 0.45, 0.6
        parameters.c3, parameters.c4 = 0.1, 0.1
    else:
        parameters.d1, parameters.a2 = 0.089159, -0.425
        parameters.a3, parameters.d4 = -0.39225, 0.10915
        parameters.d5, parameters.d6 = 0.09465, 0.0823
    solver = getattr(hm, family + "Kinematics")(
        parameters, np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
    )
    info = solver.capabilities
    assert info.family == getattr(hm.KinematicsFamily, family)
    assert info.default_ik_algorithm == hm.IKAlgorithm.ANALYTIC
    assert info.dof == 6 and info.supports_closed_form_ik
    assert info.supports_concurrent_queries
    assert not info.supports_jacobian
    assert not info.supports_redundancy
    assert not info.supports_batch_fk
    joints = np.array([0.25, -0.7, 0.85, -0.45, 0.65, -0.2])
    target = solver.forward(joints)
    solution = solver.inverse(target, joints)
    np.testing.assert_allclose(solver.forward(solution), target, atol=1e-7)


def test_numerical_capabilities_do_not_depend_on_robot_type(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "slide.urdf"
    urdf.write_text(
        "<robot name='slide'><link name='base'/><link name='tip'/>"
        "<joint name='slide' type='prismatic'>"
        "<parent link='base'/><child link='tip'/><axis xyz='1 0 0'/>"
        "<limit lower='-1' upper='1' velocity='1' effort='1'/>"
        "</joint></robot>"
    )
    solver = hm.Robot(str(urdf)).kinematics
    info = solver.capabilities
    assert info.family == hm.KinematicsFamily.NUMERICAL
    assert info.default_ik_algorithm == hm.IKAlgorithm.NUMERICAL
    assert info.dof == 1 and info.supports_jacobian
    assert info.supports_concurrent_queries
    assert not info.supports_redundancy
    assert not info.supports_closed_form_ik
    assert not info.supports_batch_fk
    np.testing.assert_allclose(solver.jacobian([0.3])[:, 0], [1, 0, 0, 0, 0, 0])
