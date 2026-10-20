"""Real-valued Python APIs inspect nested object arrays before conversion."""

from decimal import Decimal

import numpy as np
import pytest
from holistic_motion._validation import real_array, require_real
from holistic_motion.geometry import SphereSpec
from holistic_motion.kit.retargeting.tasks import FrameTask, PostureTask
from holistic_motion.trajectory import ToppraTrajectory


def _nested(value, depth):
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = value
        value = wrapper
    return value


@pytest.mark.parametrize("depth", [1, 4, 32, 2000])
@pytest.mark.parametrize("value", [np.complex64(0.1 + 1j), np.complex128(0.1)])
def test_real_validation_rejects_complex_values_at_any_object_depth(depth, value):
    with pytest.raises(TypeError, match="real"):
        require_real(_nested(value, depth), name="nested input")


@pytest.mark.parametrize("entry", ["sphere", "frame", "posture", "toppra"])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_public_python_apis_reject_nested_complex_objects(entry, imaginary):
    nested = _nested(np.complex128(0.1 + imaginary * 1j), 4)
    values = np.zeros(3, dtype=object)
    values[0] = nested
    with pytest.raises(TypeError):
        if entry == "sphere":
            SphereSpec(values, 1.0)
        elif entry == "frame":
            FrameTask(position_cost=values)
        elif entry == "posture":
            PostureTask(cost=nested)
        else:
            path = np.array([[0.0], [0.5], [1.0]], dtype=object)
            path[1, 0] = nested
            ToppraTrajectory(path, [1.0], [1.0])


def test_real_validation_accepts_shared_acyclic_objects_and_decimal_values():
    nested = _nested(np.float64(0.25), 4)
    values = np.empty(3, dtype=object)
    values[0] = nested
    values[1] = nested
    values[2] = Decimal("0.5")
    np.testing.assert_array_equal(real_array(values, name="values"), [0.25, 0.25, 0.5])


@pytest.mark.parametrize("length", [1, 2, 16])
def test_real_validation_rejects_object_cycles_without_float_conversion(length):
    nodes = [np.empty((), dtype=object) for _ in range(length)]
    for index, node in enumerate(nodes):
        node[()] = nodes[(index + 1) % length]
    with pytest.raises(TypeError, match="cyclic"):
        require_real(nodes[0], name="cyclic input")


def test_real_validation_rejects_a_cycle_inside_an_acyclic_outer_array():
    cycle = np.empty((), dtype=object)
    cycle[()] = cycle
    outer = np.empty(1, dtype=object)
    outer[0] = cycle
    with pytest.raises(TypeError, match="cyclic"):
        require_real(outer, name="values")


def test_real_validation_traverses_shared_containers_once():
    value = np.array(0.5, dtype=object)
    for _ in range(32):
        outer = np.empty(2, dtype=object)
        outer[:] = [value, value]
        value = outer
    assert require_real(value, name="shared") is value
