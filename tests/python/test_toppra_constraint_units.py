"""Overflowing normalized limits must retain their finite physical ratios."""

import warnings

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory


@pytest.mark.parametrize("coordinate", [1e306, 4e306, 1e307])
@pytest.mark.parametrize("clock", [1e-120, 1e-8, 0.1, 1.0, 2.0])
@pytest.mark.parametrize("at_rest", [False, True])
def test_large_normalized_limits_keep_every_joint_acceleration_constraint(
    coordinate, clock, at_rest
):
    points = np.array(
        [
            [-0.24580157500195576, 0.07696346259939685],
            [-0.17772220141070655, 0.11103528961462661],
            [0.17184162792972796, -0.15586459956318222],
        ]
    )
    velocity = np.array([1.173446497409783, 0.8577734524515942])
    acceleration = np.array([2.3731630871303775, 2.0305484347475384])
    start, end = (0.0, 0.0) if at_rest else (0.001161675716771791, 0.005274033871900206)
    options = {"grid_size": 65, "start_path_velocity": start, "end_path_velocity": end}
    reference = ToppraTrajectory(points, velocity, acceleration, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        trajectory = ToppraTrajectory(
            points * coordinate,
            (velocity * coordinate) * clock,
            ((acceleration * coordinate) * clock) * clock,
            **{
                **options,
                "start_path_velocity": start * clock,
                "end_path_velocity": end * clock,
            },
        )
        assert trajectory.duration * clock == pytest.approx(
            reference.duration, rel=2e-12
        )
        fractions = np.linspace(0.0037, 0.9961, 1001)
        expected = reference.sample(fractions * reference.duration)
        actual = trajectory.sample(fractions * trajectory.duration)
        normalized = []
        for order, (value, target) in enumerate(zip(actual, expected)):
            assert np.isfinite(value).all()
            value = value / coordinate
            for _ in range(order):
                value = value / clock
            normalized.append(value)
            np.testing.assert_allclose(value, target, rtol=2e-10, atol=2e-11)
        assert np.all(abs(normalized[1]) <= velocity * (1.0 + 2e-10))
        assert np.all(abs(normalized[2]) <= acceleration * (1.0 + 2e-10))


@pytest.mark.parametrize(
    "coordinate,grid_size", [(1e306, 32), (1e307, 32), (1e307, 200), (1e307, 2000)]
)
@pytest.mark.parametrize("clock", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("end_speed", [0.0, 0.2])
@pytest.mark.parametrize("inactive_axis", [False, True])
def test_finite_motion_survives_unrepresentable_unscaled_interval_coefficients(
    coordinate, grid_size, clock, end_speed, inactive_axis
):
    # Velocity limits fix the interior speeds. Ordinary physical acceleration
    # is effectively unbounded on this huge straight path, even though the
    # unscaled path-acceleration coefficients overflow on a fine grid.
    points = np.array([[0.0], [1.0]])
    velocity, acceleration = np.ones(1), np.array([1e20])
    if inactive_axis:
        points = np.c_[points, np.zeros(2)]
        velocity = np.r_[velocity, 1.0]
        acceleration = np.r_[acceleration, 1.0]
    reference = ToppraTrajectory(
        points, velocity, acceleration, grid_size=grid_size, end_path_velocity=end_speed
    )
    physical_velocity = np.full(points.shape[1], clock)
    physical_acceleration = np.full(points.shape[1], clock**2)
    if inactive_axis:
        physical_velocity[-1] = 1e-300 * clock
        physical_acceleration[-1] = 1e-300 * clock**2
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        trajectory = ToppraTrajectory(
            points * coordinate,
            physical_velocity,
            physical_acceleration,
            grid_size=grid_size,
            end_path_velocity=(end_speed / coordinate) * clock,
        )
        assert (trajectory.duration / coordinate) * clock == pytest.approx(
            reference.duration, rel=2e-12
        )
        midpoint_fractions = (
            reference.result.times[:-1] + 0.5 * np.diff(reference.result.times)
        ) / reference.duration
        fractions = np.r_[np.linspace(0.0037, 0.9961, 101), midpoint_fractions]
        actual = trajectory.sample(fractions * trajectory.duration)
        expected = reference.sample(fractions * reference.duration)
        normalized = [
            actual[0] / coordinate,
            actual[1] / clock,
            ((actual[2] / clock) / clock) * coordinate,
        ]
        for value, target in zip(normalized, expected):
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value, target, rtol=2e-9, atol=2e-10)
        assert np.all(abs(actual[1]) <= physical_velocity * (1 + 2e-10))
        assert np.all(abs(actual[2]) <= physical_acceleration * (1 + 2e-10))


@pytest.mark.parametrize("factor", [0.5, 0.9, 1.0])
@pytest.mark.parametrize(
    "grid", [[0.0, 0.25, 0.75, 1.0], [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]]
)
@pytest.mark.parametrize("boundary", [(0.0, 0.0), (0.01, 0.02)])
def test_bernstein_controls_may_exceed_range_while_the_curve_remains_finite(
    factor, grid, boundary
):
    # In the middle segment the first derivative has Bernstein controls
    # [0.88, 1.08, 0.88], but its true maximum is only 0.98. Chords are
    # proportional to [0.25, 0.5, 0.25], giving these exact spline knots.
    x = np.array([0.0, 0.20333333333333334, 0.6766666666666666, 0.88])
    lengths = 1.2 * np.array([0.25, 0.5, 0.25])
    y = np.r_[0.0, np.cumsum(np.sqrt(lengths**2 - np.diff(x) ** 2))]
    points = np.c_[x, y]
    options = {
        "gridpoints": grid,
        "start_path_velocity": boundary[0],
        "end_path_velocity": boundary[1],
    }
    reference = ToppraTrajectory(points, [0.2, 0.2], [0.2, 0.2], **options)
    scale = factor * np.finfo(float).max
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        trajectory = ToppraTrajectory(
            points * scale, [0.2 * scale] * 2, [0.2 * scale] * 2, **options
        )
        assert trajectory.duration == pytest.approx(reference.duration, rel=2e-12)
        fractions = np.linspace(0.0037, 0.9961, 1001)
        for actual, expected in zip(
            trajectory.sample(fractions * trajectory.duration),
            reference.sample(fractions * reference.duration),
        ):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual / scale, expected, rtol=2e-10, atol=2e-11)
