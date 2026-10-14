import numpy as np
import pytest


@pytest.fixture(params=["numerical", "ur", "opw", "srs", "fep"])
def solver_data(request, tmp_path):
    import holistic_motion as hm

    if request.param in ("srs", "fep"):
        from test_srs_kinematics import _write_ideal_srs_urdf
        from test_fep_batch import _seven_axis_urdf

        urdf = tmp_path / "seven.urdf"
        if request.param == "srs":
            _write_ideal_srs_urdf(urdf)
            solver = hm.Robot(str(urdf)).kinematics
        else:
            urdf.write_text(_seven_axis_urdf())
            solver = hm.Robot(str(urdf)).create_fep_kinematics("link0", "link7")
        return solver, np.zeros(7)
    if request.param == "numerical":
        urdf = tmp_path / "single.urdf"
        urdf.write_text("""<robot name="single"><link name="base"/><link name="tip"/>
<joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>
<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="2"/>
</joint></robot>""")
        robot = hm.Robot(str(urdf))
        return robot.kinematics, np.array([0.25])
    if request.param == "ur":
        parameters = hm.URParameters()
        parameters.d1 = 0.089159
        parameters.a2 = -0.425
        parameters.a3 = -0.39225
        parameters.d4 = 0.10915
        parameters.d5 = 0.09465
        parameters.d6 = 0.0823
        solver_type = hm.URKinematics
    else:
        parameters = hm.OPWParameters()
        parameters.a1 = 0.15
        parameters.a2 = -0.10
        parameters.c1 = 0.45
        parameters.c2 = 0.60
        parameters.c3 = 0.10
        parameters.c4 = 0.10
        solver_type = hm.OPWKinematics
    parameters.rotation_directions = [1] * 6
    solver = solver_type(parameters, np.full(6, -2 * np.pi), np.full(6, 2 * np.pi))
    return solver, np.array([0.25, -0.70, 0.85, -0.45, 0.65, -0.20])


@pytest.mark.parametrize("method", ["inverse", "solve_all"])
@pytest.mark.parametrize("defect", ["last_row", "rotation", "reflection"])
def test_native_ik_rejects_nonrigid_matrix(solver_data, method, defect):
    solver, joints = solver_data
    target = solver.forward(joints)
    if defect == "last_row":
        target[3, 3] = 2.0
    elif defect == "rotation":
        target[:3, :3] *= 0.5
    else:
        target[:3, 0] *= -1
    with pytest.raises(ValueError, match="rigid"):
        getattr(solver, method)(target, joints)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
@pytest.mark.parametrize("argument", ["joints", "target", "seed"])
def test_native_kinematics_rejects_complex_arrays(
    solver_data, dtype, imaginary, argument
):
    solver, joints = solver_data
    target = solver.forward(joints)
    if argument == "joints":
        values = np.array(joints, dtype=dtype)
        values[0] = np.complex128(joints[0] + imaginary * 1j)
        call, args = solver.forward, (values,)
    elif argument == "target":
        values = np.array(target, dtype=dtype)
        values[0, 3] = np.complex128(target[0, 3] + imaginary * 1j)
        call, args = solver.inverse, (values, joints)
    else:
        values = np.array(joints, dtype=dtype)
        values[0] = np.complex128(joints[0] + imaginary * 1j)
        call, args = solver.inverse, (target, values)
    with pytest.raises(TypeError, match="real"):
        call(*args)


def test_native_kinematics_preserves_real_object_inputs(solver_data):
    solver, joints = solver_data
    target = solver.forward(joints)
    np.testing.assert_allclose(solver.forward(joints.astype(object)), target)
    recovered = solver.inverse(target.astype(object), joints.astype(object))
    np.testing.assert_allclose(solver.forward(recovered), target, atol=1e-7)


def test_native_kinematics_rejects_complex_lists(solver_data):
    solver, joints = solver_data
    target = solver.forward(joints)
    invalid = joints.astype(np.complex64).tolist()
    with pytest.raises(TypeError, match="real"):
        solver.forward(invalid)
    with pytest.raises(TypeError, match="real"):
        solver.inverse(target.astype(complex).tolist(), joints.tolist())
    np.testing.assert_allclose(solver.forward(joints.tolist()), target)


def test_native_kinematics_rejects_recursive_lists(solver_data):
    solver, _ = solver_data
    invalid = []
    invalid.append(invalid)
    with pytest.raises(TypeError):
        solver.forward(invalid)


@pytest.mark.parametrize(
    "method", ["forward_all", "forward_user", "jacobian", "configuration"]
)
def test_native_specialized_joints_reject_complex(solver_data, method):
    solver, joints = solver_data
    if not hasattr(solver, method):
        pytest.skip("method is not part of this solver family")
    with pytest.raises(TypeError, match="real"):
        getattr(solver, method)(joints.astype(complex) + 1j)


@pytest.mark.parametrize("argument", ["target", "seed"])
def test_native_specialized_solve_rejects_complex(solver_data, argument):
    solver, joints = solver_data
    if not hasattr(solver, "solve"):
        pytest.skip("specialized solver method")
    target = solver.forward(joints)
    if argument == "target":
        target = target.astype(complex) + 1j
    else:
        joints = joints.astype(complex) + 1j
    with pytest.raises(TypeError, match="real"):
        solver.solve(target, joints)


def test_native_batch_rejects_complex(solver_data):
    solver, joints = solver_data
    if not hasattr(solver, "forward_batch"):
        pytest.skip("batch FK is only part of FEP")
    with pytest.raises(TypeError, match="real"):
        solver.forward_batch((joints.astype(complex) + 1j)[None])


@pytest.mark.parametrize("setter", ["set_tcp", "set_user_frame", "home_joints"])
def test_native_complex_configuration_update_preserves_state(solver_data, setter):
    solver, joints = solver_data
    if not hasattr(solver, setter):
        pytest.skip("numerical configuration setter")
    name = {"set_tcp": "tcp", "set_user_frame": "user_frame"}.get(setter, setter)
    previous = np.asarray(getattr(solver, name)).copy()
    invalid = previous.astype(complex) + 1j
    with pytest.raises(TypeError, match="real"):
        if setter == "home_joints":
            solver.home_joints = invalid
        else:
            getattr(solver, setter)(invalid)
    np.testing.assert_array_equal(getattr(solver, name), previous)
