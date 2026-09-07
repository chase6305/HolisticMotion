import numpy as np
import pytest
from holistic_motion.trajectory import ToppraResult, ToppraTrajectory, retime_path
from holistic_motion.trajectory.toppra import _NaturalCubicPath


@pytest.mark.parametrize("count", [2, 3, 17, 100])
def test_natural_spline_interpolates_and_preserves_c2_continuity(count):
    generator = np.random.default_rng(482)
    grid = np.cumsum(generator.uniform(0.01, 0.5, count))
    grid = (grid - grid[0]) / (grid[-1] - grid[0])
    values = generator.normal(size=(count, 4))

    path = _NaturalCubicPath(grid, values)

    np.testing.assert_allclose(path.evaluate(grid), values, atol=1e-12)
    np.testing.assert_allclose(path.evaluate(grid[[0, -1]], order=2), 0.0, atol=1e-9)
    # Evaluate each polynomial at its right endpoint; the next polynomial
    # starts at delta=0. This checks both derivatives without finite differences.
    delta = np.diff(grid)[:-1, None]
    left_first = path.b[:-1] + delta * (2.0 * path.c[:-1] + 3.0 * delta * path.d[:-1])
    left_second = 2.0 * path.c[:-1] + 6.0 * delta * path.d[:-1]
    np.testing.assert_allclose(left_first, path.b[1:], rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(left_second, 2.0 * path.c[1:], rtol=1e-11, atol=1e-9)


def test_natural_spline_two_waypoints_is_linear():
    path = _NaturalCubicPath(np.array([0.0, 1.0]), np.array([[1.0, 2.0], [3.0, -2.0]]))
    query = np.array([0.0, 0.25, 0.75, 1.0])
    np.testing.assert_allclose(
        path.evaluate(query), [1.0, 2.0] + query[:, None] * [2.0, -4.0]
    )
    np.testing.assert_allclose(
        path.evaluate(query, order=1), np.tile([2.0, -4.0], (4, 1))
    )
    np.testing.assert_allclose(path.evaluate(query, order=2), 0.0)


def test_toppra_retimes_multijoint_path_with_zero_boundaries():
    trajectory = retime_path(
        [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
        [1.0, 0.8],
        [2.0, 1.5],
        grid_size=120,
    )
    times, position, velocity, acceleration = trajectory.sample_uniform(300)
    assert trajectory.duration > 0.0
    np.testing.assert_allclose(position[0], [0.0, 0.0])
    np.testing.assert_allclose(position[-1], [1.0, 0.5])
    np.testing.assert_allclose(velocity[[0, -1]], 0.0, atol=1e-9)
    assert np.max(np.abs(velocity), axis=0)[0] <= 1.0 + 1e-6
    assert np.max(np.abs(velocity), axis=0)[1] <= 0.8 + 1e-6
    assert np.all(np.max(np.abs(acceleration), axis=0) <= [2.0 + 1e-6, 1.5 + 1e-6])
    assert np.all(np.diff(times) > 0.0)


def test_toppra_validates_inputs_and_boundary_velocity():
    with pytest.raises(ValueError, match="distinct"):
        ToppraTrajectory([[0.0], [0.0]], [1.0], [1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        ToppraTrajectory([[0.0], [1.0]], [0.0], [1.0])
    with pytest.raises(ValueError, match="boundary"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], start_path_velocity=2.0)
    with pytest.raises(ValueError, match="finite"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], start_path_velocity=float("nan"))
    with pytest.raises(TypeError, match="grid_size"):
        ToppraTrajectory([[0.0], [1.0]], [1.0], [1.0], grid_size=20.5)


def test_toppra_clamps_sample_times():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0])
    position, _, _ = trajectory.sample([-1.0, trajectory.duration + 1.0])
    np.testing.assert_allclose(position[:, 0], [0.0, 1.0])


def test_toppra_result_is_an_immutable_snapshot():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0])
    expected, _, _ = trajectory.sample_uniform(10)[1:]

    with pytest.raises(ValueError, match="read-only"):
        trajectory.result.path_speeds[0] = 10.0
    with pytest.raises(ValueError, match="read-only"):
        trajectory.waypoints[0, 0] = 10.0
    with pytest.raises(ValueError, match="read-only"):
        trajectory.max_velocity[0] = 10.0
    with pytest.raises(AttributeError):
        trajectory.result = trajectory.result

    actual, _, _ = trajectory.sample_uniform(10)[1:]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gridpoints": [0.0, 0.0]}, "gridpoints"),
        ({"path_speeds": [0.0, -1.0]}, "non-negative"),
        ({"times": [0.0, -1.0], "duration": -1.0}, "duration"),
        ({"duration": 2.0}, "final time"),
        ({"path_accelerations": [1.0]}, "path dynamics"),
    ],
)
def test_toppra_result_rejects_inconsistent_snapshots(kwargs, message):
    values = {
        "gridpoints": [0.0, 1.0],
        "path_speeds": [1.0, 1.0],
        "path_accelerations": [0.0],
        "times": [0.0, 1.0],
        "duration": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ToppraResult(**values)


def test_toppra_sampling_uses_constant_acceleration_interval_dynamics():
    trajectory = ToppraTrajectory([[0.0], [1.0]], [1.0], [2.0], grid_size=20)
    interval = 0
    start_time = trajectory.result.times[interval]
    end_time = trajectory.result.times[interval + 1]
    elapsed = 0.5 * (end_time - start_time)
    expected_s = (
        trajectory.result.gridpoints[interval]
        + trajectory.result.path_speeds[interval] * elapsed
        + 0.5 * trajectory.result.path_accelerations[interval] * elapsed**2
    )

    position, velocity, _ = trajectory.sample([start_time + elapsed])

    np.testing.assert_allclose(position[0, 0], expected_s, atol=1e-12)
    np.testing.assert_allclose(
        velocity[0, 0],
        trajectory.result.path_speeds[interval]
        + trajectory.result.path_accelerations[interval] * elapsed,
        atol=1e-12,
    )


@pytest.mark.parametrize("boundary", [(0.1, 0.1), (0.0, 0.2), (0.2, 0.0)])
def test_toppra_preserves_boundary_speeds_and_continuous_limits(boundary):
    trajectory = ToppraTrajectory(
        [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
        [1.0, 1.0],
        [2.0, 2.0],
        start_path_velocity=boundary[0],
        end_path_velocity=boundary[1],
        grid_size=20,
    )

    np.testing.assert_allclose(
        trajectory.result.path_speeds[[0, -1]], boundary, atol=1e-14
    )
    _, _, velocity, acceleration = trajectory.sample_uniform(10001)
    assert np.max(np.abs(velocity)) <= 1.0 + 1e-8
    assert np.max(np.abs(acceleration)) <= 2.0 + 1e-8


@pytest.mark.parametrize("boundary", [(0.0, 1.0), (1.0, 0.0)])
def test_toppra_rejects_unreachable_boundary_speed(boundary):
    # The requested speed change needs distance 1 / (2 * 0.1) = 5,
    # but this path has length 1. Both endpoints satisfy the velocity bound.
    with pytest.raises(ValueError, match="velocity|infeasible"):
        ToppraTrajectory(
            [[0.0], [1.0]],
            [2.0],
            [0.1],
            start_path_velocity=boundary[0],
            end_path_velocity=boundary[1],
            grid_size=20,
        )


def test_toppra_reaches_nonzero_end_speed_when_zero_cannot_reach_it_locally():
    trajectory = ToppraTrajectory(
        [[0.0], [1.0]],
        [1.0],
        [1.0],
        start_path_velocity=1.0,
        end_path_velocity=1.0,
        grid_size=20,
    )
    np.testing.assert_allclose(trajectory.result.path_speeds, 1.0, atol=1e-12)
    assert trajectory.duration == pytest.approx(1.0)


def test_toppra_limits_are_owned_snapshots():
    velocity = np.array([1.0])
    acceleration = np.array([2.0])
    trajectory = ToppraTrajectory([[0.0], [1.0]], velocity, acceleration)
    velocity[0] = 3.0
    acceleration[0] = 4.0
    np.testing.assert_array_equal(trajectory.max_velocity, [1.0])
    np.testing.assert_array_equal(trajectory.max_acceleration, [2.0])


def test_toppra_result_rejects_times_inconsistent_with_motion():
    with pytest.raises(ValueError, match="timing"):
        ToppraResult([0.0, 1.0], [1.0, 1.0], [0.0], [0.0, 2.0], 2.0)


@pytest.mark.parametrize("seed", [3, 12, 421])
def test_toppra_bounds_polynomial_extrema_on_nonuniform_intervals(seed):
    points = np.random.default_rng(seed).normal(size=(6, 3))
    trajectory = ToppraTrajectory(
        points,
        [0.7, 1.0, 1.5],
        [1.0, 2.0, 0.8],
        gridpoints=[0.0, 0.03, 0.2, 0.4, 0.7, 0.96, 1.0],
    )
    _assert_polynomial_limits(trajectory)


def _assert_polynomial_limits(trajectory):
    from numpy.polynomial import polynomial as poly

    result = trajectory.result
    path = trajectory._path_model

    def extrema(coefficients):
        roots = poly.polyroots(poly.polyder(coefficients))
        candidates = [0.0, 1.0]
        candidates.extend(
            root.real
            for root in roots
            if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0
        )
        return poly.polyval(candidates, coefficients)

    # Independently find the extrema in the monomial basis, rather than
    # inspecting the Bernstein bounds used by the implementation.
    for index, ds in enumerate(np.diff(result.gridpoints)):
        start = result.gridpoints[index]
        segment = np.clip(
            np.searchsorted(path.grid, start, side="right") - 1, 0, len(path.d) - 1
        )
        x = [
            result.path_speeds[index] ** 2,
            result.path_speeds[index + 1] ** 2 - result.path_speeds[index] ** 2,
        ]
        first = path.evaluate([start], order=1)[0]
        second = path.evaluate([start], order=2)[0]
        for joint in range(trajectory.dof):
            derivative = np.array(
                [first[joint], ds * second[joint], 3.0 * ds**2 * path.d[segment, joint]]
            )
            speed_squared = poly.polymul(poly.polymul(derivative, derivative), x)
            acceleration = poly.polyadd(
                poly.polymul(poly.polyder(derivative) / ds, x),
                derivative * result.path_accelerations[index],
            )
            assert (
                np.max(extrema(speed_squared))
                <= trajectory.max_velocity[joint] ** 2 + 1e-9
            )
            assert (
                np.max(np.abs(extrema(acceleration)))
                <= trajectory.max_acceleration[joint] + 1e-9
            )


@pytest.mark.parametrize(
    "spatial_scale,time_scale", [(1.0, 1.0), (0.25, 4.0), (4.0, 0.25)]
)
def test_toppra_handles_nearly_zero_half_plane_coefficients(spatial_scale, time_scale):
    # Minimized by Hypothesis: a coefficient around 2.4e-15 used to turn
    # cancellation in 1 - a*x into a false 0.029-wide feasibility gap.
    points = spatial_scale * np.array(
        [[0.125, 0.0, -0.125], [0.25, 0.125, 0.125], [0.375, 0.0, 0.0]]
    )
    trajectory = ToppraTrajectory(
        points,
        np.full(3, 0.25 * spatial_scale * time_scale),
        np.full(3, 0.5 * spatial_scale * time_scale**2),
        grid_size=20,
    )
    assert trajectory.duration > 0.0
    position, velocity, _ = trajectory.sample([0.0, trajectory.duration])
    np.testing.assert_allclose(position, points[[0, -1]], atol=1e-12)
    np.testing.assert_allclose(velocity, 0.0, atol=1e-9)
    _assert_polynomial_limits(trajectory)


def test_toppra_random_paths_do_not_stop_before_the_last_interval():
    generator = np.random.default_rng(421)
    for _ in range(20):
        points = generator.normal(
            size=(generator.integers(3, 10), generator.integers(1, 8))
        )
        trajectory = ToppraTrajectory(
            points,
            np.ones(points.shape[1]),
            np.full(points.shape[1], 2.0),
            grid_size=20,
        )
        assert np.all(np.diff(trajectory.result.times) > 0.0)
        np.testing.assert_allclose(
            trajectory.sample([trajectory.duration])[0][0], points[-1], atol=1e-12
        )
