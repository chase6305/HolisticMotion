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
