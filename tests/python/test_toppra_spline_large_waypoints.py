"""Large-coordinate spline fitting preserves long multi-waypoint paths."""

import warnings

import numpy as np
import pytest
from holistic_motion.trajectory.toppra import _NaturalCubicPath


@pytest.mark.parametrize("count", [3, 12, 100, 1000])
@pytest.mark.parametrize("dof", [1, 7])
@pytest.mark.parametrize("coordinate", [1e160, 1e200, 1e280])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_large_coordinate_spline_matches_dimensionless_reference(
    count, dof, coordinate, nonuniform
):
    generator = np.random.default_rng(27317)
    if nonuniform:
        grid = np.r_[0.0, np.cumsum(generator.uniform(0.5, 1.5, count - 1))]
        grid /= grid[-1]
    else:
        grid = np.linspace(0.0, 1.0, count)
    values = generator.uniform(-0.25, 0.25, (count, dof))
    reference = _NaturalCubicPath(grid, values)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = _NaturalCubicPath(grid, values * coordinate)
        query = np.unique(np.r_[grid, 0.5 * (grid[:-1] + grid[1:])])
        motion = actual.evaluate_all(query)
    expected = reference.evaluate_all(query)
    for value, target in zip(motion, expected):
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(target)))
        assert np.isfinite(value).all()
        np.testing.assert_allclose(
            value / coordinate, target, rtol=5e-11, atol=tolerance
        )
    np.testing.assert_allclose(actual.evaluate(grid) / coordinate, values, atol=2e-13)
    for name in ("b", "c", "d"):
        target = getattr(reference, name)
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(target)))
        np.testing.assert_allclose(
            getattr(actual, name) / coordinate, target, rtol=5e-11, atol=tolerance
        )
