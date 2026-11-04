"""Retargeting residuals remain nonnegative across floating-point ranges."""

import math

import numpy as np
import pytest
from holistic_motion.kit.retargeting import PinocchioRetargetingSolver
from holistic_motion.kit.retargeting.pinocchio_solver import _stable_norm
from optional_dependencies import python_pinocchio_or_skip

_TINY = np.nextafter(0.0, 1.0)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-0.0],
        [-3.0, 4.0],
        [-1e200],
        [-1e200, 1e200, 0.0],
        [-1e-200],
        [-1e-200, 1e-200, 0.0],
        [-_TINY],
        [-3 * _TINY, 4 * _TINY],
        [-np.finfo(float).max],
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "readonly"])
def test_stable_residual_norm_preserves_extreme_values(values, layout):
    supplied = np.array(values)
    if layout == "reversed":
        supplied = supplied[::-1]
    elif layout == "readonly":
        supplied.setflags(write=False)
    original = supplied.tobytes(), supplied.flags.writeable
    expected = math.hypot(*supplied)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        actual = _stable_norm(supplied)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, rel=2e-15, abs=0.0)
    assert not np.signbit(actual)
    assert original == (supplied.tobytes(), supplied.flags.writeable)


@pytest.mark.parametrize(
    "value",
    [
        -0.0,
        -1e200,
        1e200,
        -1e-200,
        1e-200,
        -_TINY,
        _TINY,
        -np.finfo(float).max,
        np.finfo(float).max,
    ],
)
def test_single_value_norm_honors_strict_floating_settings(value):
    # A one-element reduction must not introduce an extra floating operation
    # that raises underflow for an otherwise representable scalar residual.
    with np.errstate(all="raise"):
        actual = _stable_norm(value)
    assert actual == abs(value)
    assert not np.signbit(actual)


@pytest.mark.parametrize("scale", [1e200, 1e-200, 1e-320])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_solver_extreme_residual_is_nonnegative(tmp_path, scale, sign):
    python_pinocchio_or_skip()
    urdf = tmp_path / "residual.urdf"
    urdf.write_text("""<robot name="residual">
      <link name="base"/><link name="tool"/>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="tool"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/>
      </joint></robot>""")
    solver = PinocchioRetargetingSolver(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": []},
        max_iterations=1,
    )
    solver.prepare("left_arm")
    target = np.eye(4)
    target[0, 3] = sign * scale
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = solver.solve({"left_hand": target}, seed=[0.0])
    assert result.residual == pytest.approx(scale, rel=2e-15, abs=0.0)
    assert not np.signbit(result.residual)
    np.testing.assert_array_equal(result.configuration, [0.0])
