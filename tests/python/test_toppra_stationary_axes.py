"""Stationary joints retain their motion without restricting active joints."""

import warnings

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory


@pytest.mark.parametrize("waypoint_count", [2, 3, 12])
@pytest.mark.parametrize("dof,active_count", [(7, 1), (7, 3), (32, 7), (64, 7)])
@pytest.mark.parametrize("boundary", [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01)])
@pytest.mark.parametrize("inactive_limit", [1e-300, 1.0, 1e300])
def test_stationary_joints_do_not_change_active_motion(
    waypoint_count, dof, active_count, boundary, inactive_limit
):
    generator = np.random.default_rng(61283)
    active = np.sort(generator.choice(dof, active_count, replace=False))
    stationary = np.ones(dof, dtype=bool)
    stationary[active] = False
    points = generator.normal(size=(waypoint_count, active_count))
    velocity = generator.uniform(0.8, 1.5, active_count)
    acceleration = generator.uniform(2.0, 3.0, active_count)
    offsets = generator.uniform(-3.0, 3.0, dof)
    padded = np.broadcast_to(offsets, (waypoint_count, dof)).copy()
    padded[:, active] = points
    full_velocity = np.full(dof, inactive_limit)
    full_acceleration = np.full(dof, inactive_limit)
    full_velocity[active] = velocity
    full_acceleration[active] = acceleration
    options = {
        "start_path_velocity": boundary[0],
        "end_path_velocity": boundary[1],
        "gridpoints": np.linspace(0.0, 1.0, 65),
    }
    reference = ToppraTrajectory(points, velocity, acceleration, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = ToppraTrajectory(padded, full_velocity, full_acceleration, **options)
        assert actual.duration == pytest.approx(reference.duration, rel=2e-12)
        np.testing.assert_allclose(
            actual.result.path_speeds,
            reference.result.path_speeds,
            rtol=2e-12,
            atol=2e-13,
        )
        fractions = np.r_[0.0, np.linspace(0.0037, 0.9961, 101), 1.0]
        expected_motion = reference.sample(fractions * reference.duration)
        actual_motion = actual.sample(fractions * actual.duration)
    for order, (value, expected) in enumerate(zip(actual_motion, expected_motion)):
        np.testing.assert_allclose(value[:, active], expected, rtol=2e-10, atol=2e-11)
        expected_stationary = np.zeros_like(value[:, stationary])
        if order == 0:
            expected_stationary[:] = offsets[stationary]
        np.testing.assert_array_equal(value[:, stationary], expected_stationary)
    np.testing.assert_array_equal(actual.waypoints, padded)
    np.testing.assert_array_equal(actual.max_velocity, full_velocity)
    np.testing.assert_array_equal(actual.max_acceleration, full_acceleration)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("argument", ["max_velocity", "max_acceleration"])
def test_stationary_joint_limits_still_require_positive_finite_values(
    invalid, argument
):
    options = {"max_velocity": [1.0, 1.0], "max_acceleration": [2.0, 2.0]}
    options[argument][1] = invalid
    with pytest.raises(ValueError):
        ToppraTrajectory([[0.0, 0.4], [1.0, 0.4]], **options)
