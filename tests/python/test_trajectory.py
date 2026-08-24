import numpy as np
import pytest


@pytest.mark.parametrize("dof", [1, 3, 7, 10, 20, 21, 24, 31, 32])
def test_rn_trajectory_infers_dimension(dof):
    import holistic_motion as hm

    waypoints = np.zeros((2, dof))
    waypoints[1, -1] = 0.2
    limits = np.ones(dof)
    trajectory = hm.RnTrajectory(waypoints, limits, limits, limits)

    assert trajectory.dof == dof
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[-1], atol=1e-7
    )


def test_rn_trajectory_rejects_unsupported_dimension():
    import holistic_motion as hm

    waypoints = np.zeros((2, 33))
    waypoints[1, -1] = 0.2
    limits = np.ones(33)
    with pytest.raises(ValueError, match="1 through 32"):
        hm.RnTrajectory(waypoints, limits, limits, limits)


def test_six_axis_joint_trajectory():
    import holistic_motion as hm

    waypoints = np.zeros((2, 6))
    waypoints[1, 0] = 0.5
    limits = np.ones(6)
    trajectory = hm.JointTrajectory6(waypoints, limits, limits, limits)

    assert trajectory.duration > 0.0
    np.testing.assert_allclose(trajectory.position(0.0), waypoints[0])
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[1], atol=1e-6
    )
    assert trajectory.profile == "double_s"


def test_joint_trajectory_rejects_malformed_inputs():
    import holistic_motion as hm

    limits = np.ones(7)
    with pytest.raises(ValueError, match="shape"):
        hm.JointTrajectory7(np.zeros((2, 6)), limits, limits, limits)
    nonfinite = np.zeros((2, 7))
    nonfinite[1, 0] = np.nan
    with pytest.raises(ValueError, match="waypoints must be finite"):
        hm.JointTrajectory7(nonfinite, limits, limits, limits)
    moving = np.zeros((2, 7))
    moving[1, 0] = 0.1
    with pytest.raises(ValueError, match="match DOF"):
        hm.JointTrajectory7(moving, limits[:-1], limits, limits)
    with pytest.raises(ValueError, match="blend_tolerance"):
        hm.JointTrajectory7(
            moving, limits, limits, limits, blend_tolerance=np.inf
        )
    with pytest.raises(ValueError, match="minimum_duration"):
        hm.JointTrajectory7(
            moving, limits, limits, limits, minimum_duration=-1.0
        )


def test_trapezoidal_profile_is_selectable_and_constrained():
    import holistic_motion as hm

    waypoints = np.zeros((4, 7))
    waypoints[:, :2] = [[0.0, 0.0], [0.2, -0.1], [0.5, 0.2], [0.1, 0.0]]
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(
        waypoints, limits, limits, limits,
        blend_tolerance=0.01, profile="trapezoidal",
    )
    assert trajectory.profile == "trapezoidal"
    report = trajectory.constraint_report()
    assert report["within_limits"]
    assert report["velocity_continuous"]
    assert not report["acceleration_continuous"]
    np.testing.assert_allclose(trajectory.position(0.0), waypoints[0])
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[-1], atol=1e-6
    )

    with pytest.raises(ValueError, match="profile must be"):
        hm.JointTrajectory7(
            waypoints, limits, limits, limits, profile="unknown"
        )


def test_zero_blend_multi_waypoint_trajectory_reaches_last_waypoint():
    import holistic_motion as hm

    waypoints = np.zeros((4, 7))
    waypoints[1, :2] = [0.2, -0.1]
    waypoints[2, :2] = [0.4, 0.1]
    waypoints[3, :2] = [0.6, 0.0]
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(
        waypoints, limits, limits, limits, blend_tolerance=0.0
    )
    np.testing.assert_allclose(trajectory.waypoints, waypoints)
    np.testing.assert_allclose(trajectory.position(0.0), waypoints[0])
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[-1], atol=1e-6
    )
    assert trajectory.path_length > np.linalg.norm(waypoints[-1] - waypoints[0])


def test_collinear_waypoints_do_not_truncate_blended_path():
    import holistic_motion as hm

    waypoints = np.zeros((5, 7))
    waypoints[:, 0] = np.linspace(0.0, 0.8, len(waypoints))
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(
        waypoints, limits, limits, limits, blend_tolerance=0.02
    )
    np.testing.assert_allclose(trajectory.waypoints, waypoints)
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[-1], atol=1e-6
    )
    assert trajectory.path_length == pytest.approx(0.8)


def test_repeated_waypoints_are_filtered_and_breakpoints_are_ordered():
    import holistic_motion as hm

    waypoints = np.zeros((5, 7))
    waypoints[:, 0] = [0.0, 0.0, 0.2, 0.2, 0.4]
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(
        waypoints, limits, limits, limits, blend_tolerance=0.005
    )
    assert trajectory.path_length > 0.0
    np.testing.assert_allclose(
        trajectory.waypoints[:, 0], [0.0, 0.2, 0.4]
    )
    assert trajectory.breakpoints[0] == pytest.approx(0.0)
    assert trajectory.breakpoints[-1] == pytest.approx(trajectory.duration)
    assert np.all(np.diff(trajectory.breakpoints) > 0.0)
    np.testing.assert_allclose(
        trajectory.position(trajectory.duration), waypoints[-1], atol=1e-6
    )

    with pytest.raises(ValueError, match="at least two distinct"):
        hm.JointTrajectory7(
            np.zeros((2, 7)), limits, limits, limits,
            blend_tolerance=0.005,
        )


def test_blended_trajectory_enforces_all_joint_limits():
    import holistic_motion as hm

    waypoints = np.zeros((5, 7))
    waypoints[:, 0] = [0.0, 0.3, 0.5, 0.2, 0.6]
    waypoints[:, 1] = [0.0, -0.2, 0.1, 0.3, 0.0]
    max_velocity = np.full(7, 0.8)
    max_acceleration = np.full(7, 1.5)
    max_jerk = np.full(7, 4.0)
    trajectory = hm.JointTrajectory7(
        waypoints, max_velocity, max_acceleration, max_jerk,
        blend_tolerance=0.01,
    )
    assert trajectory.dof == 7
    assert trajectory.blend_tolerance == pytest.approx(0.01)

    samples = np.linspace(0.0, trajectory.duration, 2001)
    measured = [
        np.max(np.abs(np.asarray([query(t) for t in samples])), axis=0)
        for query in (
            trajectory.velocity, trajectory.acceleration, trajectory.jerk
        )
    ]
    assert np.all(measured[0] <= max_velocity)
    assert np.all(measured[1] <= max_acceleration)
    assert np.all(measured[2] <= max_jerk)


def test_minimum_duration_only_slows_trajectory():
    import holistic_motion as hm

    waypoints = np.zeros((2, 7))
    waypoints[1, 0] = 0.5
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(waypoints, limits, limits, limits)
    original_duration = trajectory.duration
    original_peak = np.max(np.abs(trajectory.velocity(original_duration / 2)))

    trajectory.set_minimum_duration(original_duration * 2.0)
    assert trajectory.duration == pytest.approx(original_duration * 2.0)
    assert np.max(np.abs(trajectory.velocity(trajectory.duration / 2))) \
        == pytest.approx(original_peak / 2.0)
    trajectory.set_minimum_duration(original_duration)
    assert trajectory.duration == pytest.approx(original_duration * 2.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        trajectory.set_minimum_duration(np.nan)

    constructed_slow = hm.JointTrajectory7(
        waypoints, limits, limits, limits,
        minimum_duration=original_duration * 3.0,
    )
    assert constructed_slow.duration == pytest.approx(original_duration * 3.0)


def test_trajectory_queries_reject_nonfinite_time():
    import holistic_motion as hm

    waypoints = np.zeros((2, 6))
    waypoints[1, 0] = 0.25
    limits = np.ones(6)
    trajectory = hm.JointTrajectory6(waypoints, limits, limits, limits)
    for query in (
        trajectory.position, trajectory.velocity,
        trajectory.acceleration, trajectory.jerk,
    ):
        with pytest.raises(ValueError, match="time must be finite"):
            query(np.nan)


def test_trajectory_derivatives_match_finite_differences():
    import holistic_motion as hm

    waypoints = np.zeros((4, 7))
    waypoints[:, 0] = [0.0, 0.2, 0.4, 0.6]
    waypoints[:, 1] = [0.0, -0.1, 0.1, 0.0]
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(
        waypoints, limits, limits, limits, blend_tolerance=0.005
    )
    step = 1e-5
    for fraction in (0.17, 0.31, 0.53, 0.77):
        time = trajectory.duration * fraction
        numerical_velocity = (
            trajectory.position(time + step)
            - trajectory.position(time - step)
        ) / (2.0 * step)
        numerical_acceleration = (
            trajectory.velocity(time + step)
            - trajectory.velocity(time - step)
        ) / (2.0 * step)
        numerical_jerk = (
            trajectory.acceleration(time + step)
            - trajectory.acceleration(time - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            trajectory.velocity(time), numerical_velocity, atol=1e-6
        )
        np.testing.assert_allclose(
            trajectory.acceleration(time), numerical_acceleration, atol=1e-6
        )
        np.testing.assert_allclose(
            trajectory.jerk(time), numerical_jerk, atol=1e-5
        )

    boundary_step = max(trajectory.duration, 1.0) * 1e-8
    for boundary in trajectory.breakpoints[1:-1]:
        left = trajectory.state(boundary - boundary_step)
        right = trajectory.state(boundary + boundary_step)
        for derivative_order in range(3):
            np.testing.assert_allclose(
                left[derivative_order], right[derivative_order], atol=5e-6
            )


def test_batch_sample_matches_scalar_queries():
    import holistic_motion as hm

    waypoints = np.zeros((3, 6))
    waypoints[:, :2] = [[0.0, 0.0], [0.2, -0.1], [0.4, 0.1]]
    limits = np.ones(6)
    trajectory = hm.JointTrajectory6(
        waypoints, limits, limits, limits, blend_tolerance=0.005
    )
    np.testing.assert_array_equal(trajectory.max_velocity, limits)
    np.testing.assert_array_equal(trajectory.max_acceleration, limits)
    np.testing.assert_array_equal(trajectory.max_jerk, limits)
    times = np.linspace(0.0, trajectory.duration, 31)
    sampled = trajectory.sample(times)
    assert len(sampled) == 4
    for values, query in zip(
        sampled,
        (
            trajectory.position, trajectory.velocity,
            trajectory.acceleration, trajectory.jerk,
        ),
    ):
        assert values.shape == (len(times), 6)
        np.testing.assert_allclose(
            values, np.asarray([query(time) for time in times])
        )
    state = trajectory.state(times[len(times) // 2])
    for state_value, sampled_values in zip(state, sampled):
        np.testing.assert_allclose(
            state_value, sampled_values[len(times) // 2]
        )
    with pytest.raises(ValueError, match="times must be finite"):
        trajectory.sample(np.array([0.0, np.nan]))

    uniform = trajectory.sample_uniform(17)
    assert len(uniform) == 5
    np.testing.assert_allclose(uniform[0],
                               np.linspace(0.0, trajectory.duration, 17))
    for actual, expected in zip(uniform[1:], trajectory.sample(uniform[0])):
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="at least 2 samples"):
        trajectory.sample_uniform(1)


def test_constraint_report_exposes_per_joint_peaks_and_utilization():
    import holistic_motion as hm

    waypoints = np.zeros((4, 7))
    waypoints[:, :2] = [[0.0, 0.0], [0.3, -0.2], [0.5, 0.1], [0.1, 0.0]]
    max_velocity = np.full(7, 0.7)
    max_acceleration = np.full(7, 1.4)
    max_jerk = np.full(7, 3.5)
    trajectory = hm.JointTrajectory7(
        waypoints, max_velocity, max_acceleration, max_jerk,
        blend_tolerance=0.01,
    )

    report = trajectory.constraint_report(samples=1001)
    assert report["within_limits"]
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"]
    assert report["maximum_utilization"] <= 1.0 + 1e-12
    for peak_name, usage_name, limits in (
        ("peak_velocity", "velocity_utilization", max_velocity),
        ("peak_acceleration", "acceleration_utilization", max_acceleration),
        ("peak_jerk", "jerk_utilization", max_jerk),
    ):
        peak = report[peak_name]
        usage = report[usage_name]
        assert peak.shape == (7,)
        assert usage.shape == (7,)
        np.testing.assert_allclose(usage, peak / limits)
    assert report["maximum_velocity_jump"].shape == (7,)
    assert report["maximum_acceleration_jump"].shape == (7,)

    with pytest.raises(ValueError, match="at least 2 samples"):
        trajectory.constraint_report(samples=1)


def test_constraint_report_scales_with_minimum_duration():
    import holistic_motion as hm

    waypoints = np.zeros((3, 7))
    waypoints[:, :2] = [[0.0, 0.0], [0.4, -0.2], [0.1, 0.2]]
    limits = np.ones(7)
    trajectory = hm.JointTrajectory7(
        waypoints, limits, limits, limits, blend_tolerance=0.01
    )
    before = trajectory.constraint_report()
    duration = trajectory.duration
    trajectory.set_minimum_duration(2.0 * duration)
    after = trajectory.constraint_report()

    for name, scale in (
        ("peak_velocity", 2.0),
        ("peak_acceleration", 4.0),
        ("peak_jerk", 8.0),
    ):
        np.testing.assert_allclose(after[name], before[name] / scale,
                                   rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
def test_randomized_trajectories_respect_limits_at_breakpoints(profile):
    import holistic_motion as hm

    random = np.random.default_rng(6305)
    for case in range(200):
        dof = 6 if case % 2 == 0 else 7
        waypoint_count = int(random.integers(2, 13))
        step_scale = random.uniform(0.005, 0.15)
        waypoints = np.cumsum(
            random.normal(0.0, step_scale, (waypoint_count, dof)), axis=0
        )
        max_velocity = random.uniform(0.3, 1.5, dof)
        max_acceleration = random.uniform(0.5, 3.0, dof)
        max_jerk = random.uniform(1.0, 10.0, dof)
        blend = float(random.choice([0.0, 0.001, 0.003, 0.005, 0.01, 0.02]))
        trajectory_type = (
            hm.JointTrajectory6 if dof == 6 else hm.JointTrajectory7
        )
        trajectory = trajectory_type(
            waypoints, max_velocity, max_acceleration, max_jerk, blend,
            profile=profile,
        )
        breakpoint_offset = (
            256.0 * np.finfo(float).eps * max(1.0, trajectory.duration)
        )
        left_breakpoints = trajectory.breakpoints[1:] - np.minimum(
            0.5 * np.diff(trajectory.breakpoints),
            np.maximum(
                breakpoint_offset, 1e-9 * np.diff(trajectory.breakpoints)
            ),
        )
        times = np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, trajectory.duration, 101),
                    trajectory.breakpoints,
                    left_breakpoints,
                    (trajectory.breakpoints[:-1]
                     + trajectory.breakpoints[1:]) / 2.0,
                )
            )
        )
        positions, velocities, accelerations, jerks = trajectory.sample(times)
        assert np.all(np.isfinite(positions))
        assert np.max(np.abs(velocities) / max_velocity) <= 1.0
        assert np.max(np.abs(accelerations) / max_acceleration) <= 1.0
        assert np.max(np.abs(jerks) / max_jerk) <= 1.0
        np.testing.assert_allclose(positions[-1], waypoints[-1], atol=1e-6)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_joint_trajectory_rejects_invalid_limits(invalid):
    import holistic_motion as hm

    waypoints = np.zeros((2, 6))
    waypoints[1, 0] = 0.5
    limits = np.ones(6)
    invalid_limits = limits.copy()
    invalid_limits[0] = invalid
    with pytest.raises(ValueError, match="finite and positive"):
        hm.JointTrajectory6(
            waypoints, invalid_limits, limits, limits
        )
