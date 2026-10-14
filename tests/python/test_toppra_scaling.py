"""Retiming should preserve endpoint and dynamics contracts across time units."""

import warnings

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraResult, ToppraTrajectory


@pytest.mark.parametrize("time_scale", [1e-120, 1e-8, 1.0, 1e8, 1e120])
@pytest.mark.parametrize("end_velocity", [0.0, 0.01])
def test_roundoff_sized_next_speed_coefficient_does_not_force_a_stop(
    time_scale, end_velocity
):
    points = np.array([[0.125, 0.125], [0.25, 0.25], [0.375, 0.25], [0.5, 0.125]])
    velocity, acceleration = np.array([0.5, 0.25]), np.array([0.5, 0.5])
    trajectory = ToppraTrajectory(
        points,
        velocity * time_scale,
        (acceleration * time_scale) * time_scale,
        grid_size=20,
        end_path_velocity=end_velocity * time_scale,
    )
    assert np.isfinite(trajectory.duration) and trajectory.duration > 0.0
    assert np.all(trajectory.result.path_speeds[1:-1] > 0.0)
    _, _, dq, ddq = trajectory.sample_uniform(1001)
    assert np.all(np.abs(dq / time_scale) <= velocity + 1e-10)
    assert np.all(np.abs((ddq / time_scale) / time_scale) <= acceleration + 1e-10)


def test_coarse_grid_with_adjacent_stops_requests_refinement():
    rng = np.random.default_rng(63054)
    points = rng.normal(size=(16, 1))
    velocity = rng.uniform(0.3, 2.0, 1)
    acceleration = rng.uniform(0.4, 3.0, 1)
    grid = np.r_[0.0, np.cumsum(np.abs(np.diff(points[:, 0])))]
    grid /= grid[-1]
    with pytest.raises(ValueError, match="zero reachable speed.*refine"):
        ToppraTrajectory(points, velocity, acceleration, gridpoints=grid)
    trajectory = ToppraTrajectory(points, velocity, acceleration, grid_size=61)
    assert 0.0 < trajectory.duration < 100.0
    _, _, dq, ddq = trajectory.sample_uniform(1001)
    assert np.all(np.abs(dq) <= velocity + 1e-10)
    assert np.all(np.abs(ddq) <= acceleration + 1e-10)


@pytest.mark.parametrize("time_scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("boundary", [(0.0, 2.0), (2.0, 0.0)])
def test_infeasible_boundary_detection_is_independent_of_time_units(
    time_scale, boundary
):
    with pytest.raises(ValueError, match="infeasible|reach"):
        ToppraTrajectory(
            [[0.0], [1.0]],
            [10.0 * time_scale],
            [time_scale**2],
            start_path_velocity=boundary[0] * time_scale,
            end_path_velocity=boundary[1] * time_scale,
            grid_size=21,
        )


@pytest.mark.parametrize("time_scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("argument", ["start_path_velocity", "end_path_velocity"])
def test_joint_velocity_boundary_rejection_is_independent_of_time_units(
    time_scale, argument
):
    with pytest.raises(ValueError, match="joint velocity limits"):
        ToppraTrajectory(
            [[0.0], [1.0]],
            [time_scale],
            [2 * time_scale**2],
            **{argument: 1.01 * time_scale},
        )


@pytest.mark.parametrize("time_scale", [1e-4, 1.0, 1e4])
def test_roundoff_does_not_create_motion_at_a_projected_stop(time_scale):
    points = np.array(
        [
            0.311,
            0.032,
            -0.279,
            0.247,
            1.865,
            -0.702,
            0.206,
            0.602,
            -0.038,
            0.848,
            0.887,
            1.02,
        ]
    )[:, None]
    grid = np.r_[0.0, np.cumsum(np.abs(np.diff(points[:, 0])))]
    grid /= grid[-1]
    trajectory = ToppraTrajectory(
        points,
        [time_scale],
        [2 * time_scale**2],
        gridpoints=grid,
        start_path_velocity=0.002 * time_scale,
    )
    assert trajectory.result.path_speeds[8] == 0.0
    # Independently evaluating the half-plane solve at 80 decimal digits gives
    # 16.6798165269817138078...; old rounding inflated this stop to ~6.5e-8.
    assert trajectory.duration * time_scale == pytest.approx(
        16.6798165269817138, rel=1e-13
    )


@pytest.mark.parametrize("time_scale", [1e-4, 1.0, 1e4, 1e8])
@pytest.mark.parametrize("end_velocity", [0.0, 0.1])
def test_fast_and_slow_retiming_preserve_interval_dynamics(time_scale, end_velocity):
    points = np.array([[0.0, 0.0], [0.2, -0.3], [0.7, 0.1], [1.0, 0.0]])
    limits = np.array([1.0, 0.7])
    baseline = ToppraTrajectory(
        points, limits, [1.0, 1.0], grid_size=31, end_path_velocity=end_velocity
    )
    scaled = ToppraTrajectory(
        points,
        limits * time_scale,
        np.ones(2) * time_scale**2,
        grid_size=31,
        end_path_velocity=end_velocity * time_scale,
    )
    assert scaled.duration * time_scale == pytest.approx(baseline.duration, rel=1e-10)
    np.testing.assert_allclose(
        scaled.result.path_speeds / time_scale,
        baseline.result.path_speeds,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scaled.result.path_accelerations / time_scale**2,
        baseline.result.path_accelerations,
        rtol=1e-9,
        atol=1e-12,
    )
    times = np.linspace(0.0, baseline.duration, 301)
    expected = baseline.sample(times)
    actual = scaled.sample(times / time_scale)
    for order in range(3):
        np.testing.assert_allclose(
            actual[order] / time_scale**order, expected[order], rtol=1e-9, atol=1e-10
        )


@pytest.mark.parametrize("scale", [1.0, 1e8])
def test_scaled_dynamics_validation_still_rejects_inconsistent_braking(scale):
    with pytest.raises(ValueError, match="path dynamics"):
        ToppraResult(
            gridpoints=[0.0, 0.5],
            path_speeds=[scale, 0.0],
            path_accelerations=[-1.000001 * scale**2],
            times=[0.0, 1.0 / scale],
            duration=1.0 / scale,
        )


@pytest.mark.parametrize("offset", [-1e-13, 1e-13])
def test_grid_endpoint_roundoff_is_canonicalized_without_mutating_input(offset):
    grid = np.array([offset, 0.5, 1.0 - offset])
    before = grid.copy()
    result = ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], gridpoints=grid)
    np.testing.assert_array_equal(result.result.gridpoints, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(grid, before)
    q, _, _ = result.sample([-1.0, 0.0, result.duration, result.duration + 1.0])
    np.testing.assert_array_equal(q[:, 0], [0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "grid", [[-1e-13, -5e-14, 0.5, 1.0], [0.0, 0.5, 1.0 + 5e-14, 1.0 + 1e-13]]
)
def test_interior_gridpoints_outside_path_domain_rejected(grid):
    with pytest.raises(ValueError, match="gridpoints"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], gridpoints=grid)


@pytest.mark.parametrize("time_scale", [1e-155, 1e-100, 1e-20, 1e100, 1e153])
@pytest.mark.parametrize("end_velocity", [0.0, 0.05])
def test_extreme_time_units_preserve_retiming_and_sampling(time_scale, end_velocity):
    points = np.array([[0.0, 0.0], [0.2, -0.3], [0.7, 0.1], [1.0, 0.0]])
    limits = np.array([1.0, 0.7])
    expected = ToppraTrajectory(
        points, limits, np.ones(2), grid_size=31, end_path_velocity=end_velocity
    )
    actual = ToppraTrajectory(
        points,
        limits * time_scale,
        np.ones(2) * time_scale**2,
        grid_size=31,
        end_path_velocity=end_velocity * time_scale,
    )
    assert actual.duration * time_scale == pytest.approx(expected.duration, rel=1e-10)
    times = np.linspace(0.0, expected.duration, 301)
    reference = expected.sample(times)
    sampled = actual.sample(times / time_scale)
    for order, (before, after) in enumerate(zip(reference, sampled)):
        assert np.isfinite(after).all()
        np.testing.assert_allclose(
            after / time_scale**order, before, rtol=1e-9, atol=1e-10
        )


@pytest.mark.parametrize("argument", ["start_path_velocity", "end_path_velocity"])
@pytest.mark.parametrize("speed", [1e200, 1e308])
def test_unrepresentable_boundary_speed_raises_validation_error(argument, speed):
    with pytest.raises(ValueError, match="boundary path velocity"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0], **{argument: speed})


@pytest.mark.parametrize("time_scale", [1e-100, 1e100])
@pytest.mark.parametrize("inactive_limit", [1e-300, 1e300])
def test_inactive_joint_limits_do_not_break_speed_normalization(
    time_scale, inactive_limit
):
    with np.errstate(all="raise"):
        expected = ToppraTrajectory(
            [[0.0], [1.0]], [time_scale], [2.0 * time_scale**2], grid_size=31
        )
        actual = ToppraTrajectory(
            [[0.0, 0.0], [1.0, 0.0]],
            [time_scale, inactive_limit],
            [2.0 * time_scale**2, inactive_limit],
            grid_size=31,
        )
        assert actual.duration == pytest.approx(expected.duration, rel=1e-12)
        times = np.linspace(0.0, actual.duration, 65)
        for reference, sampled in zip(expected.sample(times), actual.sample(times)):
            np.testing.assert_allclose(sampled[:, :1], reference, rtol=1e-12, atol=0)
            np.testing.assert_array_equal(sampled[:, 1], 0.0)


@pytest.mark.parametrize("spatial_scale", [1e200, 1e250, 1e300])
@pytest.mark.parametrize("end_velocity", [0.0, 0.05])
def test_large_finite_chords_preserve_retiming_and_sampling(
    spatial_scale, end_velocity
):
    points = np.array([[0.0, 0.0], [1.0, 0.25], [2.0, -0.25]])
    reference = ToppraTrajectory(
        points,
        [1.0, 0.8],
        [2.0, 1.5],
        grid_size=25,
        end_path_velocity=end_velocity,
    )
    with np.errstate(all="raise"):
        actual = ToppraTrajectory(
            points * spatial_scale,
            np.array([1.0, 0.8]) * spatial_scale,
            np.array([2.0, 1.5]) * spatial_scale,
            grid_size=25,
            end_path_velocity=end_velocity,
        )
        assert actual.duration == pytest.approx(reference.duration, rel=1e-10)
        times = np.linspace(0.0, reference.duration, 301)
        for before, after in zip(reference.sample(times), actual.sample(times)):
            assert np.isfinite(after).all()
            np.testing.assert_allclose(
                after / spatial_scale, before, rtol=1e-9, atol=1e-10
            )


@pytest.mark.parametrize(
    "points, message",
    [
        ([[1e308], [-1e308]], "differences"),
        ([[0.0, 0.0], [1.7e308, 1.7e308]], "finite chord lengths"),
        ([[1e200], [1e200]], "distinct"),
    ],
)
def test_unrepresentable_or_duplicate_chords_raise_validation_error(points, message):
    dof = len(points[0])
    with np.errstate(all="raise"), pytest.raises(ValueError, match=message):
        ToppraTrajectory(points, np.ones(dof), np.ones(dof))


@pytest.mark.parametrize("velocity_limit", [1e155, 1e200, 1e300])
@pytest.mark.parametrize("end_velocity", [0.0, 0.2])
def test_loose_finite_velocity_caps_do_not_warn(velocity_limit, end_velocity):
    expected = ToppraTrajectory(
        [[0.0], [1.0]],
        [10.0],
        [1.0],
        grid_size=25,
        end_path_velocity=end_velocity,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = ToppraTrajectory(
            [[0.0], [1.0]],
            [velocity_limit],
            [1.0],
            grid_size=25,
            end_path_velocity=end_velocity,
        )
        assert actual.duration == pytest.approx(expected.duration, rel=1e-10)
        times = np.linspace(0.0, expected.duration, 129)
        for before, after in zip(expected.sample(times), actual.sample(times)):
            np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("spatial_scale", [1e100, 1e155, 1e200, 1e300])
@pytest.mark.parametrize("end_velocity", [0.0, 0.2])
def test_large_geometry_with_ordinary_limits_preserves_finite_sampling(
    spatial_scale, end_velocity
):
    # Both paths are velocity-limited on this grid. A unit acceleration is
    # effectively unbounded after scaling the path and its duration together.
    expected = ToppraTrajectory(
        [[0.0], [1.0]],
        [1.0],
        [1e20],
        grid_size=25,
        end_path_velocity=end_velocity,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = ToppraTrajectory(
            [[0.0], [spatial_scale]],
            [1.0],
            [1.0],
            grid_size=25,
            end_path_velocity=end_velocity / spatial_scale,
        )
        assert actual.duration / spatial_scale == pytest.approx(expected.duration)
        times = np.linspace(0.0, expected.duration, 129)
        q, velocity, acceleration = actual.sample(times * spatial_scale)
        reference_q, reference_v, reference_a = expected.sample(times)
        np.testing.assert_allclose(
            q / spatial_scale, reference_q, rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(velocity, reference_v, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            acceleration * spatial_scale, reference_a, rtol=1e-9, atol=1e-10
        )
        assert np.isfinite([q, velocity, acceleration]).all()
        assert np.max(np.abs(velocity)) <= 1.0 + 1e-10
        assert np.max(np.abs(acceleration)) <= 1.0


@pytest.mark.parametrize("end_velocity", [0.0, 0.05])
def test_combined_space_and_time_units_normalize_geometry_to_limit_ratios(end_velocity):
    points = np.array([[0.0, 0.0], [0.2, -0.3], [0.7, 0.1], [1.0, 0.0]])
    limits = np.array([1.0, 0.7])
    expected = ToppraTrajectory(
        points,
        limits,
        np.ones(2),
        grid_size=31,
        end_path_velocity=end_velocity,
    )
    geometry_scale, time_scale = 1e40, 1e-80
    with np.errstate(all="raise"):
        actual = ToppraTrajectory(
            points * geometry_scale,
            limits * geometry_scale * time_scale,
            np.ones(2) * geometry_scale * time_scale**2,
            grid_size=31,
            end_path_velocity=end_velocity * time_scale,
        )
        assert actual.duration * time_scale == pytest.approx(expected.duration)
        times = np.linspace(0.0, expected.duration, 129)
        for order, (before, after) in enumerate(
            zip(expected.sample(times), actual.sample(times / time_scale))
        ):
            np.testing.assert_allclose(
                after / (geometry_scale * time_scale**order),
                before,
                rtol=1e-9,
                atol=1e-10,
            )
