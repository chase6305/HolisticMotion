import numpy as np
import pytest


def parameters(family):
    import holistic_motion as hm

    if family == "ur":
        value = hm.URParameters()
        value.a2, value.a3 = -0.425, -0.39225
        solver = hm.URKinematics
    else:
        value = hm.OPWParameters()
        value.c2, value.c3 = 0.60, 0.10
        solver = hm.OPWKinematics
    value.rotation_directions = [1] * 6
    return value, solver


@pytest.mark.parametrize("family", ["ur", "opw"])
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_analytic_constructor_rejects_complex_limits(family, side, dtype, imaginary):
    value, solver = parameters(family)
    lower, upper = np.full(6, -6.0), np.full(6, 6.0)
    invalid = np.full(
        6, np.complex128((-6 if side == "lower" else 6) + imaginary * 1j), dtype=dtype
    )
    if side == "lower":
        lower = invalid
    else:
        upper = invalid
    with pytest.raises(TypeError, match="real"):
        solver(value, lower, upper)


@pytest.mark.parametrize(
    "family,name",
    [("ur", name) for name in ("d1", "a2", "a3", "d4", "d5", "d6")]
    + [("opw", name) for name in ("a1", "a2", "b", "c1", "c2", "c3", "c4")],
)
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_complex_geometry_assignment_preserves_parameter(family, name, imaginary):
    value, _ = parameters(family)
    previous = getattr(value, name)
    with pytest.raises(TypeError, match="real"):
        setattr(value, name, np.complex128(0.25 + imaginary * 1j))
    assert getattr(value, name) == previous


@pytest.mark.parametrize("family", ["ur", "opw"])
@pytest.mark.parametrize("name", ["offsets", "rotation_directions"])
def test_complex_parameter_array_preserves_state(family, name):
    value, _ = parameters(family)
    previous = list(getattr(value, name))
    with pytest.raises(TypeError, match="real"):
        setattr(value, name, np.full(6, np.complex128(1 + 1j), dtype=object))
    assert list(getattr(value, name)) == previous


@pytest.mark.parametrize("family", ["ur", "opw"])
@pytest.mark.parametrize("name", ["base_transform", "tool_transform"])
@pytest.mark.parametrize("defect", ["complex", "last_row", "rotation"])
def test_invalid_parameter_transform_preserves_state(family, name, defect):
    value, _ = parameters(family)
    previous = np.asarray(getattr(value, name)).copy()
    invalid = previous.copy()
    if defect == "complex":
        invalid = invalid.astype(complex) + 1j
        exception, message = TypeError, "real"
    else:
        if defect == "last_row":
            invalid[3, 3] = 2
        else:
            invalid[:3, :3] *= 0.5
        exception, message = ValueError, "rigid"
    with pytest.raises(exception, match=message):
        setattr(value, name, invalid)
    np.testing.assert_array_equal(getattr(value, name), previous)


@pytest.mark.parametrize("family", ["ur", "opw"])
def test_real_parameter_arrays_and_transforms_remain_supported(family):
    value, solver_type = parameters(family)
    offsets = np.arange(6, dtype=float) * 0.01
    value.offsets = offsets.astype(object)
    value.rotation_directions = np.array([1, -1, 1, -1, 1, -1], dtype=object)
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    value.base_transform = transform.astype(object)
    np.testing.assert_allclose(value.base_transform, transform)
    np.testing.assert_allclose(value.offsets, offsets)
    solver = solver_type(
        value, np.full(6, -6, dtype=object), np.full(6, 6, dtype=object)
    )
    assert np.isfinite(solver.forward(np.zeros(6))).all()
    offsets[:] = 1
    transform[:] = 0
    assert value.offsets[0] == 0
    assert value.base_transform[3, 3] == 1


@pytest.mark.parametrize("family", ["ur", "opw"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_native_constructor_rejects_nonfinite_parameters(family, invalid):
    value, solver = parameters(family)
    value.offsets = [invalid, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="offsets"):
        solver(value, np.full(6, -6.0), np.full(6, 6.0))


@pytest.mark.parametrize(
    "kind",
    ["ur_wrist", "ur_wrist_pi", "opw_shoulder", "opw_wrist", "opw_wrist_pi"],
)
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("offset", [-0.7, 0.0, 0.3])
@pytest.mark.parametrize("free_angle", [-2.0 * np.pi, -0.35, 0.0, 2.0 * np.pi])
def test_analytic_singular_branch_preserves_seed_coordinates(
    kind, direction, offset, free_angle
):
    import holistic_motion as hm

    if kind.startswith("ur"):
        params = hm.URParameters()
        params.d1, params.a2, params.a3, params.d4, params.d5, params.d6 = (
            0.089159,
            -0.425,
            -0.39225,
            0.10915,
            0.09465,
            0.0823,
        )
        solver_type = hm.URKinematics
        analytic_joints = np.array([0.2, -0.55, 0.75, 0.9, 0.0, free_angle])
        free_joint = 5
    else:
        params = hm.OPWParameters()
        # Binary-exact lengths keep the straight arm's wrist centre on the
        # shoulder axis, so the intended singular branch is reproducible.
        params.c1, params.c2, params.c3, params.c4 = 0.5, 0.5, 0.25, 0.125
        solver_type = hm.OPWKinematics
        analytic_joints = np.array([0.25, 0.0, 0.0, 0.4, 0.65, -0.2])
        free_joint = 0 if kind == "opw_shoulder" else 3
        analytic_joints[free_joint] = free_angle
        if free_joint == 3:
            analytic_joints[4] = 0.0
    if kind.endswith("_pi"):
        analytic_joints[4] = np.pi
    offsets = np.zeros(6)
    directions = np.ones(6, dtype=int)
    offsets[free_joint] = offset
    directions[free_joint] = direction
    params.offsets = offsets.tolist()
    params.rotation_directions = directions.tolist()
    solver = solver_type(params, np.full(6, -20.0), np.full(6, 20.0))
    seed = (analytic_joints + offsets) * directions
    target = solver.forward(seed)

    report = solver.solve_ik(target, seed, hm.IKSelection.ALL)
    assert report.success
    assert min(report.distances) < 1e-7
    assert any(
        np.allclose(solution, seed, atol=1e-7, rtol=0) for solution in report.solutions
    )
    for solution in report.solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-7, rtol=0)
    nearest = solver.solve_ik(target, seed, hm.IKSelection.NEAREST)
    assert nearest.success
    assert nearest.distances[0] < 1e-7
    np.testing.assert_allclose(nearest.solutions[0], seed, atol=1e-7, rtol=0)
