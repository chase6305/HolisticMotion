"""Changing coordinate units must not overflow finite spline constraints."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory


@pytest.mark.parametrize("scale", [1e306, 1e307])
@pytest.mark.parametrize("points", [[[0.0], [1.0]], [[0.0], [1.0], [0.0]]])
@pytest.mark.parametrize("grid_size", [32, 200])
@pytest.mark.parametrize(
    "boundary", [(0.0, 0.0), (0.0, 0.05), (0.05, 0.0), (0.05, 0.1)]
)
def test_large_geometry_matches_ordinary_unit_control(
    scale, points, grid_size, boundary
):
    points = np.asarray(points)
    kwargs = {
        "grid_size": grid_size,
        "start_path_velocity": boundary[0],
        "end_path_velocity": boundary[1],
    }
    ordinary = ToppraTrajectory(points, [1.0], [1.0], **kwargs)
    scaled = ToppraTrajectory(points * scale, [scale], [scale], **kwargs)
    assert scaled.duration == pytest.approx(ordinary.duration, rel=2e-12)
    fractions = np.linspace(0.0, 1.0, 1001)
    expected = ordinary.sample(fractions * ordinary.duration)
    actual = scaled.sample(fractions * scaled.duration)
    for result, reference in zip(actual, expected):
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result / scale, reference, rtol=2e-10, atol=2e-11)
    assert np.max(np.abs(actual[1])) / scale <= 1.0 + 1e-10
    assert np.max(np.abs(actual[2])) / scale <= 1.0 + 1e-10


def test_large_spline_recovery_keeps_small_independent_axes():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    scales = np.array([1e307, 1e-100])
    ordinary = ToppraTrajectory(points, [1.0, 1.0], [1.0, 1.0])
    scaled = ToppraTrajectory(points * scales, scales, scales)
    assert scaled.duration == pytest.approx(ordinary.duration, rel=2e-12)
    fractions = np.linspace(0.0, 1.0, 1001)
    for actual, expected in zip(
        scaled.sample(fractions * scaled.duration),
        ordinary.sample(fractions * ordinary.duration),
    ):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual / scales, expected, rtol=2e-10, atol=2e-11)
