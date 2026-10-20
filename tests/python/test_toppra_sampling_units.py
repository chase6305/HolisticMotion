"""Large spline coefficients must not overflow before motion units cancel."""

import warnings

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory
from holistic_motion.trajectory.toppra import _NaturalCubicPath


def _curved_points():
    # A five-waypoint regression with finite physical spline coefficients.
    # Its former sampler overflowed before applying the time-unit factor.
    rng = np.random.default_rng(40216)
    for index in range(27):
        points = rng.uniform(-0.4, 0.4, (3 + index % 3, 1 + index % 2))
    return points


@pytest.mark.parametrize("coordinate", [1e200, 1e300, 1e307])
@pytest.mark.parametrize("clock", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("boundary", [(0.0, 0.0), (0.01, 0.02)])
@pytest.mark.parametrize("mixed_axes", [False, True])
def test_large_curved_sampling_restores_units_after_combining_motion(
    coordinate, clock, boundary, mixed_axes
):
    points = _curved_points()
    scales = np.array([coordinate])
    if mixed_axes:
        # Every moving axis is proportional, so chord-length knots are
        # unchanged by independent coordinate units; the last axis is static.
        points = np.c_[points, 0.3 * points, np.full_like(points, 0.125)]
        scales = np.array([coordinate, 1e-300, 1.0])
    velocity = np.arange(1, points.shape[1] + 1, dtype=float)
    acceleration = 2.0 * velocity
    options = {
        "grid_size": 21,
        "start_path_velocity": boundary[0],
        "end_path_velocity": boundary[1],
    }
    expected = ToppraTrajectory(points, velocity, acceleration, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        trajectory = ToppraTrajectory(
            points * scales,
            (velocity * scales) * clock,
            ((acceleration * scales) * clock) * clock,
            **{
                **options,
                "start_path_velocity": boundary[0] * clock,
                "end_path_velocity": boundary[1] * clock,
            },
        )
        assert trajectory.duration * clock == pytest.approx(
            expected.duration, rel=2e-12
        )
        fractions = np.linspace(0.0037, 0.9961, 1001)
        for order, (actual, reference) in enumerate(
            zip(
                trajectory.sample(fractions * trajectory.duration),
                expected.sample(fractions * expected.duration),
            )
        ):
            assert np.isfinite(actual).all()
            normalized = actual / scales
            for _ in range(order):
                normalized = normalized / clock
            np.testing.assert_allclose(normalized, reference, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("count", [2, 3, 5])
@pytest.mark.parametrize("coordinate", [1e200, 1e300, 1e307])
def test_large_spline_evaluation_keeps_physical_coefficients_and_all_orders(
    count, coordinate
):
    values = _curved_points()[:count]
    knots = np.r_[0.0, np.cumsum(np.abs(np.diff(values[:, 0])))]
    knots /= knots[-1]
    reference = _NaturalCubicPath(knots, values)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        path = _NaturalCubicPath(knots, values * coordinate)
        for name in ("a", "b", "c", "d"):
            np.testing.assert_allclose(
                getattr(path, name) / coordinate,
                getattr(reference, name),
                rtol=2e-13,
                atol=2e-13,
            )
        query = np.linspace(0.0, 1.0, 1001)
        actual = path.evaluate_all(query)
        expected = reference.evaluate_all(query)
        for order in range(3):
            assert np.isfinite(actual[order]).all()
            np.testing.assert_allclose(
                actual[order] / coordinate, expected[order], rtol=2e-12, atol=2e-12
            )
            np.testing.assert_allclose(
                path.evaluate(query, order=order), actual[order], rtol=0.0, atol=0.0
            )
        with pytest.raises(ValueError, match="order"):
            path.evaluate(query, order=3)


def test_unrepresentable_physical_derivatives_raise_validation_without_warning():
    rng = np.random.default_rng(40216)
    for index in range(159):
        points = rng.uniform(-0.4, 0.4, (3 + index % 3, 1 + index % 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="physical spline.*derivatives"):
            ToppraTrajectory(points * 1e307, [1e307], [2e307], grid_size=21)
