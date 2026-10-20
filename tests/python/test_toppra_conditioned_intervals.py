"""Nearby caller grid points must not amplify reachability roundoff."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory


@pytest.mark.parametrize("dof", [1, 7, 32])
@pytest.mark.parametrize("delta", [-4e-13, 1e-10])
@pytest.mark.parametrize(
    "coordinate,clock", [(1.0, 1.0), (1e-6, 1e6), (1e100, 1e50), (1e200, 1e-80)]
)
@pytest.mark.parametrize(
    "start,end", [(0.0, 1e-4), (1e-4, 0.0), (1e-3, 0.01), (0.01, 0.01)]
)
def test_nearby_grid_nodes_preserve_limits_and_requested_endpoint_speeds(
    dof, delta, coordinate, clock, start, end
):
    direction = np.linspace(0.2, 1.0, dof)
    if dof > 1:
        direction[-1] = 0.0
    limits = np.where(direction != 0.0, direction, 1.0)
    points = np.array([0.0, 0.5 + delta, 1.0])[:, None] * direction * coordinate
    velocity = (10.0 * limits * coordinate) * clock
    acceleration = ((limits * coordinate) * clock) * clock
    grid = np.linspace(0.0, 1.0, 17)
    grid.setflags(write=False)
    trajectory = ToppraTrajectory(
        points,
        velocity,
        acceleration,
        gridpoints=grid,
        start_path_velocity=start * clock,
        end_path_velocity=end * clock,
    )
    result = trajectory.result
    assert np.isin(grid, result.gridpoints).all()
    assert np.isin(trajectory._waypoint_s, result.gridpoints).all()
    assert result.gridpoints.size == grid.size + 1
    assert 0.0 < np.min(np.diff(result.gridpoints)) < 1e-8
    assert result.path_speeds[0] / clock == pytest.approx(
        start, rel=4.0 * np.finfo(float).eps, abs=0.0
    )
    assert result.path_speeds[-1] / clock == pytest.approx(
        end, rel=4.0 * np.finfo(float).eps, abs=0.0
    )
    queries = np.r_[
        result.times,
        0.5 * (result.times[:-1] + result.times[1:]),
        trajectory.duration * np.linspace(0.0037, 0.9961, 301),
    ]
    position, speed, actual_acceleration = trajectory.sample(queries)
    assert np.isfinite(position).all()
    assert np.max(np.abs(speed) / velocity) <= 1.0 + 2e-8
    assert np.max(np.abs(actual_acceleration) / acceleration) <= 1.0 + 2e-8


@pytest.mark.parametrize("acceleration", [1e-20, 1e-100, 1e-280])
def test_large_equal_endpoint_speeds_keep_a_feasible_constant_profile(acceleration):
    trajectory = ToppraTrajectory(
        [[0.0], [0.5 - 4e-13], [1.0]],
        [20.0],
        [acceleration],
        gridpoints=np.linspace(0.0, 1.0, 17),
        start_path_velocity=10.0,
        end_path_velocity=10.0,
    )
    _, speed, actual_acceleration = trajectory.sample(trajectory.result.times)
    np.testing.assert_allclose(speed, 10.0, rtol=4.0 * np.finfo(float).eps, atol=0.0)
    assert np.max(np.abs(actual_acceleration)) <= acceleration * (1.0 + 2e-8)


@pytest.mark.parametrize("grid_size", [2, 200])
def test_unresolvable_endpoint_change_is_rejected_without_changing_requested_speed(
    grid_size,
):
    # Even this one-ulp velocity change exceeds the available acceleration.
    with pytest.raises(ValueError, match="numerically resolvable"):
        ToppraTrajectory(
            [[0.0], [1.0]],
            [20.0],
            [1e-20],
            grid_size=grid_size,
            start_path_velocity=10.0,
            end_path_velocity=np.nextafter(10.0, np.inf),
        )


@pytest.mark.parametrize(
    "coordinate,clock", [(1.0, 1e-100), (1.0, 1e100), (1e200, 1e-80), (1e200, 1.0)]
)
def test_physical_repair_keeps_nearby_grid_timing_consistent_across_units(
    coordinate, clock
):
    # Nearly uniform chords place caller grid points close to spline knots.
    points = np.array(
        [
            -0.06307516347147371,
            -0.05650483394170451,
            -0.04993450441494263,
            -0.04336417488503408,
            -0.03679384536016675,
            -0.030223515828062503,
            -0.023653186300261607,
            -0.017082856774261072,
            -0.010512527243772923,
            -0.0039421977163069276,
            0.002628131811064993,
            0.009198461338970405,
            0.01576879086985972,
            0.02233912039503327,
            0.028909449925722773,
            0.03547977945476884,
            0.042050108978605234,
            0.048620438509113684,
            0.055190768039264114,
            0.06176109756375541,
            0.06833142709409944,
            0.07490175662122404,
            0.08147208615194608,
            0.08804241567930961,
            0.09461274520538666,
            0.1011830747356254,
            0.10775340426208659,
            0.11432373379099402,
            0.1208940633183182,
            0.1274643928475402,
            0.1340347223761745,
            0.14060505190547554,
            0.14717538143276335,
        ]
    )[:, None]
    velocity = np.array([1.0242888920207975])
    acceleration = np.array([2.4279155224488376])
    grid = np.linspace(0.0, 1.0, 65)
    reference = ToppraTrajectory(
        points, velocity, acceleration, gridpoints=grid, start_path_velocity=1e-4
    )
    # A monotone one-dimensional chord path is a line. Its triangular timing
    # supplies an independent check against an unnecessary global slowdown.
    distance = points[-1, 0] - points[0, 0]
    initial_joint_velocity = distance * 1e-4
    peak = np.sqrt(acceleration[0] * distance + 0.5 * initial_joint_velocity**2)
    assert peak < velocity[0]
    analytical_duration = (2.0 * peak - initial_joint_velocity) / acceleration[0]
    assert reference.duration == pytest.approx(analytical_duration, rel=1e-7)
    transformed = ToppraTrajectory(
        points * coordinate,
        (velocity * coordinate) * clock,
        ((acceleration * coordinate) * clock) * clock,
        gridpoints=grid,
        start_path_velocity=1e-4 * clock,
    )
    assert transformed.duration * clock == pytest.approx(reference.duration, rel=1e-9)
    _, actual_velocity, actual_acceleration = transformed.sample(
        transformed.result.times
    )
    assert np.max(np.abs(actual_velocity) / ((velocity * coordinate) * clock)) <= (
        1.0 + 2e-8
    )
    assert np.max(
        np.abs(actual_acceleration) / (((acceleration * coordinate) * clock) * clock)
    ) <= (1.0 + 2e-8)
    assert transformed.result.path_speeds[0] / clock == pytest.approx(
        1e-4, rel=4.0 * np.finfo(float).eps
    )
    assert transformed.result.path_speeds[-1] == 0.0


@pytest.mark.parametrize("delta", [1e-10, 1e-12, 1e-14])
@pytest.mark.parametrize(
    "coordinate,clock", [(1.0, 1.0), (1e-6, 1e6), (1e200, 1e-80), (1.0, 1e-100)]
)
def test_acceleration_certificate_accounts_for_stationary_derivative_cancellation(
    delta, coordinate, clock
):
    points = np.array([0.10481811155206727, 1.1268294163149597, -0.31846619186668135])
    root = 0.4435483893773745
    grid = np.array([0.0, root - delta, root, root + delta, 1.0])
    trajectory = ToppraTrajectory(
        points[:, None] * coordinate,
        [10.0 * coordinate * clock],
        [(coordinate * clock) * clock],
        gridpoints=grid,
    )
    index = int(np.flatnonzero(trajectory.result.gridpoints == root - delta)[0])
    x = np.zeros(trajectory.result.gridpoints.size)
    x[index + 1] = 1.0
    bound = trajectory._acceleration_bounds(x, np.array([index]))[0]

    # Derive the quadratic acceleration directly from the original cubic
    # coefficients. Rounded endpoint derivatives lose digits near this root.
    path = trajectory._path_model
    ld = np.longdouble
    left, right = trajectory.result.gridpoints[index : index + 2].astype(ld)
    segment = int(np.searchsorted(path.grid, float(left), side="right") - 1)
    local, ds = left - ld(path.grid[segment]), right - left
    coefficients = path._normalized_coefficients
    scale = ld(1.0)
    if coefficients is None:
        coefficients = (path.b, path.c, path.d)
    else:
        scale = ld(path._coefficient_scale[0])
    b, c, d = (ld(block[segment, 0]) for block in coefficients)
    p0 = b + local * (2 * c + 3 * local * d)
    p1 = ds * (2 * c + 6 * local * d)
    p2 = 3 * ds * ds * d
    u = ld(1.0 / (2.0 * float(ds)))
    change = 2 * ds * u
    g0, g1, g2 = p0 * change, 3 * p1 * change, 5 * p2 * change
    vertex = np.clip(-g1 / (2 * g2), 0, 1)
    unit = scale * ld(trajectory._speed_scale) ** 2 / ld(trajectory.max_acceleration[0])
    maximum = max(
        abs((g0 + value * (g1 + value * g2)) / (2 * ds) * unit)
        for value in (ld(0), ld(1), vertex)
    )
    assert maximum <= ld(bound) + 4 * ld(np.finfo(float).eps) * max(1, maximum)


@pytest.mark.parametrize(
    "amplitude,ratio", [(1e-6, 0.16666666666660188), (1e-4, 0.1666666660185186)]
)
@pytest.mark.parametrize(
    "coordinate,clock", [(1.0, 1.0), (1e-6, 1e6), (1e100, 1e-50), (1e200, 1e-80)]
)
def test_local_repair_avoids_global_slowdown_near_flat_inflection(
    amplitude, ratio, coordinate, clock
):
    points = np.column_stack(
        (amplitude * np.array([-1.0, -ratio, 0.0, ratio, 1.0]), np.arange(5.0) - 2.0)
    )
    velocity = (np.array([10.0 * amplitude, 1e10]) * coordinate) * clock
    acceleration = ((np.array([amplitude, 1e15]) * coordinate) * clock) * clock
    coarse = np.linspace(0.0, 1.0, 17)
    fine = np.sort(np.r_[coarse, 0.5 - 1e-8, 0.5 + 1e-8])
    fine.setflags(write=False)
    reference = ToppraTrajectory(
        points * coordinate, velocity, acceleration, gridpoints=coarse
    )
    trajectory = ToppraTrajectory(
        points * coordinate, velocity, acceleration, gridpoints=fine
    )
    # Guard against a multi-fold global slowdown. Discretization and the
    # rounding certificate can still add some conservative timing.
    assert trajectory.duration <= 1.25 * reference.duration
    assert trajectory.duration * clock < 6.0
    assert np.isin(fine[[8, 10]], trajectory.result.gridpoints).all()
    assert trajectory.result.path_speeds[[0, -1]].tolist() == [0.0, 0.0]
    times = trajectory.result.times
    queries = np.r_[
        times,
        (
            times[:-1, None]
            + np.diff(times)[:, None] * np.array([0.0001, 0.1, 0.5, 0.9, 0.9999])
        ).reshape(-1),
    ]
    _, actual_velocity, actual_acceleration = trajectory.sample(queries)
    assert np.max(np.abs(actual_velocity) / velocity) <= 1.0 + 2e-8
    assert np.max(np.abs(actual_acceleration) / acceleration) <= 1.0 + 2e-8


def test_conditioned_acceleration_residual_defers_to_physical_check():
    a = np.array([[1e8, 2.0, 0.5, 0.0]])
    b = np.array([[-1e8, 0.0, 0.0, 0.5]])
    assert ToppraTrajectory._validate_speed_profile(a, b, np.ones(2)).tolist() == [0]


@pytest.mark.parametrize("kind", ["ordinary", "velocity", "velocity_second_row"])
def test_physical_deferral_keeps_velocity_and_ordinary_residual_checks(kind):
    a = np.array([[1e8, 2.0, 0.5, 0.0]])
    b = np.array([[-1e8, 0.0, 0.0, 0.5]])
    if kind == "ordinary":
        a[0, 0], b[0, 0] = 0.0, 0.0
    elif kind == "velocity":
        a[0, -2] = 2.0
    else:
        a = np.vstack((a, [0.0, 0.0, 2.0, 0.0]))
        b = np.vstack((b, [0.0, 0.0, 0.0, 0.5]))
    with pytest.raises(RuntimeError, match="violates interval constraints"):
        ToppraTrajectory._validate_speed_profile(a, b, np.ones(a.shape[0] + 1))


def test_conditioned_envelope_defers_affected_ordinary_interval():
    a = np.array([[1e8, 0.0, 0.5, 0.0], [0.0, 2.0, 0.5, 0.0]])
    b = np.array([[-1e8, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]])
    assert ToppraTrajectory._validate_speed_profile(a, b, np.ones(3)).tolist() == [0, 1]


@pytest.mark.parametrize("clock", [1.0, 1e40, 1e-80])
def test_conditioned_flat_inflection_with_heterogeneous_joint_units(clock):
    flat = 1e-6 * np.array([-1.0, -0.16666666666660188, 0.0, 0.16666666666660188, 1.0])
    points = flat[:, None] * np.linspace(0.2, 1.0, 16)[None, :]
    points[:, -1] = np.arange(5.0) - 2.0
    scales = np.array(
        [
            1.0,
            1.0,
            1e200,
            1e200,
            1e200,
            1e100,
            1e200,
            1e200,
            1e-6,
            1.0,
            1e100,
            1e-6,
            1.0,
            1.0,
            1.0,
            1e200,
        ]
    )
    velocity, acceleration = np.full(16, 1e-5), np.full(16, 1e-6)
    velocity[-1], acceleration[-1] = 1e10, 1e15
    velocity = (velocity * scales) * clock
    acceleration = ((acceleration * scales) * clock) * clock
    grid = np.sort(np.r_[np.linspace(0.0, 1.0, 17), 0.5 - 1e-8, 0.5 + 1e-8])
    trajectory = ToppraTrajectory(
        points * scales, velocity, acceleration, gridpoints=grid
    )
    assert np.isin(grid, trajectory.result.gridpoints).all()
    assert trajectory.result.path_speeds[[0, -1]].tolist() == [0.0, 0.0]
    assert 0.0 < trajectory.duration * clock < 25.0
    times = trajectory.result.times
    query = np.r_[
        times,
        times[:-1] + 0.5 * np.diff(times),
        np.linspace(0.0, trajectory.duration, 2001),
    ]
    position, actual_velocity, actual_acceleration = trajectory.sample(query)
    assert np.isfinite(position).all()
    assert np.max(np.abs(actual_velocity) / velocity) <= 1.0 + 2e-8
    assert np.max(np.abs(actual_acceleration) / acceleration) <= 1.0 + 2e-8
