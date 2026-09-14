"""Composition and boundary checks for shared trajectory segment queries."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.fixture(
    params=[(2, "double_s"), (7, "double_s"), (2, "trapezoidal"), (7, "trapezoidal")]
)
def trajectory(request):
    dof, profile = request.param
    waypoints = np.zeros((4, dof))
    waypoints[:, :2] = [[0, 0], [0.2, -0.1], [0.4, 0.2], [0.6, 0.1]]
    limits = np.ones(dof)
    result = hm.RnTrajectory(
        waypoints, limits, limits, limits, blend_tolerance=0.005, profile=profile
    )
    result.set_minimum_duration(1.7 * result.duration)
    return result


def test_combined_queries_match_scalar_queries_at_boundaries(trajectory):
    knots = trajectory.breakpoints
    offset = 1e-9 * trajectory.duration
    times = np.concatenate(
        ([-1.0, trajectory.duration + 1], knots, knots - offset, knots + offset)
    )
    samples = trajectory.sample(times)
    for index, time in enumerate(times):
        state = trajectory.state(time)
        for order, query in enumerate(
            (
                trajectory.position,
                trajectory.velocity,
                trajectory.acceleration,
                trajectory.jerk,
            )
        ):
            expected = query(time)
            np.testing.assert_allclose(state[order], expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                samples[order][index], expected, rtol=1e-12, atol=1e-12
            )


def test_state_derivatives_match_finite_differences(trajectory):
    knots = trajectory.breakpoints
    for left, right in zip(knots[:-1], knots[1:]):
        time = (left + right) / 2
        step = 1e-5 * min(right - left, 1.0)
        center = trajectory.state(time)
        before = trajectory.state(time - step)
        after = trajectory.state(time + step)
        for order in range(1, 4):
            difference = (after[order - 1] - before[order - 1]) / (2 * step)
            np.testing.assert_allclose(center[order], difference, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("scale,order", [(1e105, 3), (1e155, 2)])
def test_large_time_scaling_preserves_representable_derivatives(scale, order):
    waypoints = np.array([[0.0, 0.0], [0.2, -0.1], [0.4, 0.2], [0.6, 0.1]])
    limits = np.ones(2)
    trajectory = hm.RnTrajectory(
        waypoints, limits, limits, limits, blend_tolerance=0.005
    )
    time = 0.1 * trajectory.duration
    original = trajectory.state(time)
    trajectory.set_minimum_duration(trajectory.duration * scale)
    expected = original[order].copy()
    for _ in range(order):
        expected /= scale
    assert np.all(expected != 0.0)
    query = trajectory.acceleration if order == 2 else trajectory.jerk
    for actual in (query(time * scale), trajectory.state(time * scale)[order]):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-7, atol=np.nextafter(0.0, 1.0)
        )


@pytest.mark.parametrize(
    "velocity,acceleration,jerk,max_duration",
    [(1e4, 1e6, 1e12, 1e-3), (1e10, 1e15, 1e20, 1e-5)],
)
def test_fast_trajectory_retains_sub_microsecond_phases(
    velocity, acceleration, jerk, max_duration
):
    waypoints = np.array([[0.0, 0.0], [0.1, 0.0]])
    trajectory = hm.RnTrajectory(
        waypoints,
        np.full(2, velocity),
        np.full(2, acceleration),
        np.full(2, jerk),
    )
    assert 0.0 < trajectory.duration < max_duration
    np.testing.assert_allclose(trajectory.position(0.0), waypoints[0], atol=1e-12)
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[-1], atol=1e-12
    )
    np.testing.assert_allclose(trajectory.velocity(trajectory.duration), 0.0, atol=1e-7)
    assert trajectory.constraint_report()["within_limits"]


@pytest.mark.parametrize("samples", [3, 2001])
def test_constraint_report_supports_large_finite_duration(samples):
    trajectory = hm.RnTrajectory(
        np.array([[0.0, 0.0], [0.1, 0.0]]),
        np.ones(2),
        np.ones(2),
        np.ones(2),
    )
    original = trajectory.constraint_report(samples=samples)
    original_duration = trajectory.duration
    trajectory.set_minimum_duration(1e308)
    report = trajectory.constraint_report(samples=samples)
    assert report["within_limits"]
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"]
    expected_velocity = original["peak_velocity"] * (
        original_duration / trajectory.duration
    )
    assert expected_velocity[0] > 0.0
    np.testing.assert_allclose(
        report["peak_velocity"],
        expected_velocity,
        rtol=1e-12,
        atol=np.nextafter(0.0, 1.0),
    )
    np.testing.assert_array_equal(report["peak_acceleration"], 0.0)
    np.testing.assert_array_equal(report["peak_jerk"], 0.0)


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
@pytest.mark.parametrize("scale", [1.0, 1.7])
def test_stopped_corner_derivatives_use_the_active_time_phase(profile, scale):
    trajectory = hm.RnTrajectory(
        np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2]]),
        np.array([1.0, 0.2]),
        np.array([1.0, 0.2]),
        np.array([1.0, 0.2]),
        blend_tolerance=0.0,
        profile=profile,
    )
    trajectory.set_minimum_duration(scale * trajectory.duration)
    knots = trajectory.breakpoints
    corner_index = next(
        i
        for i, time in enumerate(knots[1:-1], 1)
        if np.linalg.norm(trajectory.position(time) - [0.1, 0.0]) < 1e-14
        and np.linalg.norm(trajectory.velocity(time)) < 1e-13
    )
    corner = knots[corner_index]
    step = 1e-8 * min(
        corner - knots[corner_index - 1], knots[corner_index + 1] - corner
    )
    times = [corner - step, corner, corner + step]
    samples = trajectory.sample(times)
    for index, time in enumerate(times):
        state = trajectory.state(time)
        for order, query in enumerate(
            (
                trajectory.position,
                trajectory.velocity,
                trajectory.acceleration,
                trajectory.jerk,
            )
        ):
            np.testing.assert_allclose(
                query(time), state[order], rtol=1e-12, atol=1e-30
            )
            np.testing.assert_allclose(
                samples[order][index], state[order], rtol=1e-12, atol=1e-30
            )
    # Before stopping, every derivative belongs to the incoming x segment;
    # the right-continuous knot and departure belong to the outgoing y segment.
    for order in range(1, 4):
        assert samples[order][0, 1] == 0.0
        assert samples[order][1, 0] == 0.0
        assert samples[order][2, 0] == 0.0
    # The tiny incoming speed can round to zero when the cubic is evaluated.
    assert samples[1][0, 0] >= 0.0
    assert samples[1][2, 1] > 0.0
    assert samples[2][0, 0] < 0.0
    assert samples[2][2, 1] > 0.0
    report = trajectory.constraint_report()
    assert report["within_limits"]
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"] == (profile == "double_s")


@pytest.mark.parametrize(
    "length,velocity,acceleration",
    [(0.1, 1e10, 1e15), (0.100005, 1.0, 10.0), (0.1, 1.0, 1e6)],
)
def test_trapezoidal_short_phases_match_analytic_motion(length, velocity, acceleration):
    trajectory = hm.RnTrajectory(
        np.array([[0.0, 0.0], [length, 0.0]]),
        np.full(2, velocity),
        np.full(2, acceleration),
        np.ones(2),
        profile="trapezoidal",
    )
    peak = min(velocity, np.sqrt(length * acceleration))
    ramp = peak / acceleration
    cruise = max(0.0, (length - peak * ramp) / peak)
    duration = 2.0 * ramp + cruise
    assert trajectory.duration == pytest.approx(
        duration * trajectory.time_scale, rel=1e-12
    )
    base_times = np.linspace(0.0, duration, 101)
    expected_position = np.where(
        base_times < ramp,
        0.5 * acceleration * base_times**2,
        np.where(
            base_times < ramp + cruise,
            peak * (base_times - 0.5 * ramp),
            length - 0.5 * acceleration * (duration - base_times) ** 2,
        ),
    )
    expected_velocity = (
        np.maximum(
            0.0,
            np.minimum(
                peak, acceleration * np.minimum(base_times, duration - base_times)
            ),
        )
        / trajectory.time_scale
    )
    positions, velocities, _, _ = trajectory.sample(base_times * trajectory.time_scale)
    np.testing.assert_allclose(
        positions[:, 0], expected_position, rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        velocities[:, 0], expected_velocity, rtol=1e-12, atol=1e-8
    )
    np.testing.assert_array_equal(positions[:, 1], 0.0)
    np.testing.assert_array_equal(velocities[:, 1], 0.0)
    assert trajectory.constraint_report()["within_limits"]
