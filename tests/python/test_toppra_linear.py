"""Independent straight-path envelopes on caller-supplied interval grids."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory


@pytest.mark.parametrize("dof", [1, 7, 32])
@pytest.mark.parametrize("grid_kind", ["uniform", "nonuniform", "endpoint_clustered"])
@pytest.mark.parametrize(
    "boundary", [(0.0, 0.0), (0.0, 0.02), (0.03, 0.0), (0.03, 0.02)]
)
def test_linear_profile_matches_independent_reachability_envelope(
    dof, grid_kind, boundary
):
    rng = np.random.default_rng(527 + dof)
    points = rng.uniform(-0.3, 0.3, (2, dof))
    points[:, 0] = [0.0, 1.0]
    if dof > 1:
        points[:, -1] = 0.2
    velocity = rng.uniform(0.7, 1.3, dof)
    acceleration = rng.uniform(1.4, 2.6, dof)
    if grid_kind == "uniform":
        grid = np.linspace(0.0, 1.0, 65)
    elif grid_kind == "nonuniform":
        grid = np.r_[0.0, np.sort(rng.uniform(0.0, 1.0, 63)), 1.0]
    else:
        grid = np.r_[0.0, 1e-10, 1e-7, 0.02, 0.13, 0.37, 0.7, 1.0 - 1e-9, 1.0]
    trajectory = ToppraTrajectory(
        points,
        velocity,
        acceleration,
        gridpoints=grid,
        start_path_velocity=boundary[0],
        end_path_velocity=boundary[1],
    )
    delta = np.abs(np.diff(points, axis=0)[0]).astype(np.longdouble)
    active = delta > 0.0
    speed_cap = np.min(velocity[active].astype(np.longdouble) / delta[active])
    path_acceleration = np.min(
        acceleration[active].astype(np.longdouble) / delta[active]
    )
    s = grid.astype(np.longdouble)
    expected = np.minimum(
        speed_cap**2,
        np.minimum(
            boundary[0] ** 2 + 2.0 * path_acceleration * s,
            boundary[1] ** 2 + 2.0 * path_acceleration * (1.0 - s),
        ),
    )
    np.testing.assert_allclose(
        trajectory.result.path_speeds**2, expected, rtol=2e-13, atol=2e-14
    )
    np.testing.assert_array_equal(trajectory.result.path_speeds[[0, -1]], boundary)
    np.testing.assert_allclose(trajectory.result.gridpoints, grid, rtol=0.0, atol=0.0)
    _, _, dq, ddq = trajectory.sample_uniform(1001)
    assert np.all(np.abs(dq) <= velocity * (1.0 + 2e-10))
    # The last few clustered intervals amplify subtraction roundoff in u.
    assert np.all(np.abs(ddq) <= acceleration * (1.0 + 2e-6))


@pytest.mark.parametrize(
    "coordinate,clock",
    [
        (1.0, 1e-120),
        (1.0, 1.0),
        (1.0, 1e120),
        (1e200, 1e-120),
        (1e200, 1.0),
        (1e300, 1e-100),
        (1e300, 1.0),
    ],
)
@pytest.mark.parametrize("boundary", [(0.0, 0.0), (0.03, 0.02)])
def test_linear_profile_preserves_independent_coordinate_and_time_units(
    coordinate, clock, boundary
):
    points = np.array([[0.1, -0.2, 0.3], [1.1, 0.1, 0.3]])
    velocity, acceleration = np.array([0.8, 1.0, 0.7]), np.array([1.4, 1.8, 1.2])
    options = {
        "gridpoints": [0.0, 0.002, 0.11, 0.37, 0.45, 0.7, 0.98, 1.0],
        "start_path_velocity": boundary[0],
        "end_path_velocity": boundary[1],
    }
    ordinary = ToppraTrajectory(points, velocity, acceleration, **options)
    scaled = ToppraTrajectory(
        points * coordinate,
        (velocity * coordinate) * clock,
        ((acceleration * coordinate) * clock) * clock,
        **{
            **options,
            "start_path_velocity": boundary[0] * clock,
            "end_path_velocity": boundary[1] * clock,
        },
    )
    assert scaled.duration * clock == pytest.approx(ordinary.duration, rel=2e-12)
    fractions = np.linspace(0.0037, 0.9961, 301)
    for order, (actual, expected) in enumerate(
        zip(
            scaled.sample(fractions * scaled.duration),
            ordinary.sample(fractions * ordinary.duration),
        )
    ):
        assert np.isfinite(actual).all()
        normalized = actual / coordinate
        for _ in range(order):
            normalized = normalized / clock
        np.testing.assert_allclose(normalized, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("boundary", [(0.0, 1.0), (1.0, 0.0), (0.1, 1.0), (1.0, 0.1)])
@pytest.mark.parametrize("clock", [1e-100, 1.0, 1e100])
def test_linear_profile_rejects_unreachable_boundaries(boundary, clock):
    with pytest.raises(ValueError, match="reach|infeasible"):
        ToppraTrajectory(
            [[0.0], [1.0]],
            [2.0 * clock],
            [0.1 * clock**2],
            start_path_velocity=boundary[0] * clock,
            end_path_velocity=boundary[1] * clock,
            gridpoints=[0.0, 0.001, 0.2, 0.9, 1.0],
        )


def test_two_node_rest_path_still_requires_refinement():
    with pytest.raises(ValueError, match="zero reachable speed.*refine"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0], grid_size=2)


def test_linear_sampling_assigns_nodes_to_the_right_interval():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0], grid_size=65)
    _, _, ddq = trajectory.sample(trajectory.result.times)
    expected = np.r_[
        trajectory.result.path_accelerations, trajectory.result.path_accelerations[-1]
    ]
    np.testing.assert_allclose(ddq[:, 0], expected, rtol=2e-13, atol=2e-13)
