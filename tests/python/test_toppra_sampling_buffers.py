"""Sampling returns independent buffers and leaves spline/timing data intact."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory


@pytest.mark.parametrize(
    "coordinate,clock",
    [
        (1.0, 1e-80),
        (1.0, 1.0),
        (1.0, 1e80),
        (1e200, 1e-80),
        (1e200, 1.0),
        (1e307, 1e-120),
    ],
)
@pytest.mark.parametrize("dof", [1, 7, 32])
@pytest.mark.parametrize("queries", ["empty", "single", "strided"])
def test_sampling_buffers_do_not_alias_model_or_previous_results(
    coordinate, clock, dof, queries
):
    parameter = np.linspace(0.0, 1.0, 5)[:, None]
    points = 0.4 * parameter + 0.001 * np.sin(4 * parameter + np.arange(dof))
    if dof > 1:
        points[:, -1] = 0.2
    trajectory = ToppraTrajectory(
        points * coordinate,
        np.full(dof, 0.8 * coordinate * clock),
        np.full(dof, ((2.0 * coordinate) * clock) * clock),
        grid_size=65,
        start_path_velocity=0.001 * clock,
        end_path_velocity=0.002 * clock,
    )
    model = trajectory._path_model
    coefficients = [model.a, model.b, model.c, model.d]
    if model._normalized_coefficients is not None:
        coefficients.extend(model._normalized_coefficients)
    original_coefficients = [array.copy() for array in coefficients]
    fields = ("gridpoints", "path_speeds", "path_accelerations", "times")
    original_result = {name: getattr(trajectory.result, name).copy() for name in fields}
    if queries == "empty":
        times = np.empty(0)
    elif queries == "single":
        times = np.array([0.371 * trajectory.duration])
    else:
        times = (np.linspace(-0.1, 1.1, 102) * trajectory.duration)[::-3]
        assert not times.flags.c_contiguous
    original_times = times.copy()
    initial = trajectory.sample(times)
    snapshots = [value.copy() for value in initial]
    subsequent = trajectory.sample(times)
    for old, new, snapshot in zip(initial, subsequent, snapshots):
        assert old.shape == (times.size, dof)
        np.testing.assert_array_equal(old, snapshot)
        np.testing.assert_array_equal(new, snapshot)
        assert not np.shares_memory(old, new)
        new[:] = 0.0
        np.testing.assert_array_equal(old, snapshot)
    repeated = trajectory.sample(times)
    for value, snapshot in zip(repeated, snapshots):
        np.testing.assert_array_equal(value, snapshot)
    np.testing.assert_array_equal(times, original_times)
    for value, original in zip(coefficients, original_coefficients):
        np.testing.assert_array_equal(value, original)
    for name, original in original_result.items():
        np.testing.assert_array_equal(getattr(trajectory.result, name), original)
