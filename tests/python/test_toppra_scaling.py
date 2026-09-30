"""Retiming should preserve endpoint and dynamics contracts across time units."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraResult, ToppraTrajectory


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
