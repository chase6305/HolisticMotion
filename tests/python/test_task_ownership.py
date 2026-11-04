"""Retargeting tasks keep immutable snapshots of caller-owned arrays."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    FrameTask,
    SupportPolygonTask,
    ZmpTask,
)


@pytest.mark.parametrize(
    "task_type,field,size",
    [
        (FrameTask, "position_cost", 3),
        (FrameTask, "orientation_cost", 3),
        (CenterOfMassTask, "cost", 3),
        (ZmpTask, "cost", 2),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        "scalar",
        "zero_dimensional",
        "list",
        "vector",
        "column",
        "strided",
        "reversed",
        "readonly",
    ],
)
def test_costs_are_readonly_snapshots(task_type, field, size, layout):
    storage = np.arange(1, 2 * size + 1, dtype=float)
    if layout == "scalar":
        supplied = 2.0
    elif layout == "zero_dimensional":
        supplied = storage[:1].reshape(())
    elif layout == "list":
        supplied = storage[:size].tolist()
    elif layout == "column":
        supplied = storage[:size, None]
    elif layout == "strided":
        supplied = storage[::2]
    elif layout == "reversed":
        supplied = storage[:size][::-1]
    else:
        supplied = storage[:size]
        if layout == "readonly":
            supplied.setflags(write=False)
    expected = (
        np.broadcast_to(supplied, (size,)).copy()
        if np.ndim(supplied) == 0
        else np.asarray(supplied).reshape(-1).copy()
    )
    original_writeable = (
        supplied.flags.writeable if isinstance(supplied, np.ndarray) else None
    )
    task = task_type(**{field: supplied})
    actual = getattr(task, field)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert not np.shares_memory(actual, storage)
    if isinstance(supplied, np.ndarray):
        assert supplied.flags.writeable == original_writeable
    storage[:] = 100.0
    if isinstance(supplied, list):
        supplied[:] = [100.0] * size
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="read-only"):
        actual[0] = 0.0
    if task_type is FrameTask:
        np.testing.assert_array_equal(
            task.cost[:3] if field == "position_cost" else task.cost[3:], expected
        )


@pytest.mark.parametrize("clockwise", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_support_geometry_is_independent_of_caller_vertices(clockwise, readonly):
    storage = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = storage[::-1] if clockwise else storage.view()
    if readonly:
        vertices.setflags(write=False)
    task = SupportPolygonTask(vertices)
    snapshots = [array.copy() for array in (task.vertices, task.normals, task.offsets)]
    assert vertices.flags.writeable == (not readonly)
    storage[:] = 99.0
    for actual, expected in zip((task.vertices, task.normals, task.offsets), snapshots):
        assert not actual.flags.writeable
        assert not np.shares_memory(actual, storage)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(task.normals @ np.zeros(2) - task.offsets, 1.0)


@pytest.mark.parametrize("clockwise", [False, True])
def test_geometry_snapshot_precedes_scalar_conversion(clockwise):
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    if clockwise:
        vertices = vertices[::-1].copy()
    expected = vertices[::-1].copy() if clockwise else vertices.copy()

    class MutatingCost(float):
        def __float__(self):
            vertices[:] = 99.0
            return 1.0

    task = SupportPolygonTask(vertices, cost=MutatingCost(1.0))
    np.testing.assert_array_equal(vertices, 99.0)
    np.testing.assert_array_equal(task.vertices, expected)
    np.testing.assert_array_equal(task.offsets, -1.0)
