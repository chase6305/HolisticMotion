"""A short final leg must propagate a reduced entry speed to earlier phases."""

import holistic_motion as hm
import numpy as np
import pytest


def test_short_final_deceleration_preserves_velocity_continuity():
    inputs = {
        "waypoints": [
            [-0.9354447474771082, -2.4134574688379695],
            [-0.5241088043619658, -3.268812656509193],
            [-0.3579563815953131, -3.6467652794396783],
        ],
        "max_velocity": [0.5498234556472988, 1.902897427993734],
        "max_acceleration": [9.847619614681955, 1.4945934331795436],
        "max_jerk": [7.53225475425155, 2.72928577422683],
        "blend_tolerance": 0.03926977684513886,
        "profile": "double_s",
    }
    trajectory = hm.RnTrajectory(**inputs)
    report = trajectory.constraint_report(10001)
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"]
    assert report["within_limits"]
    points = np.asarray(inputs["waypoints"])
    q = trajectory.sample([0.0, trajectory.duration])[0]
    np.testing.assert_allclose(q, points[[0, -1]], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("length", [1000.0, 10000.0])
@pytest.mark.parametrize("jerk", [1e4, 1e5, 1e6])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_short_jerk_ramps_after_long_cruise(length, jerk, direction):
    trajectory = hm.RnTrajectory(
        [[0.0], [direction * length]],
        [0.1],
        [1.0],
        [jerk],
        profile="double_s",
        blend_tolerance=0.005,
    )
    for slowdown in [1.0, 1.7]:
        if slowdown != 1.0:
            trajectory.set_minimum_duration(slowdown * trajectory.duration)
        report = trajectory.constraint_report(2)
        assert report["within_limits"]
        assert report["velocity_continuous"]
        assert report["acceleration_continuous"]
        assert np.max(report["maximum_acceleration_jump"]) < 1e-12
        assert np.max(report["maximum_velocity_jump"]) < 1e-12
        q, v, a, _ = trajectory.sample([0.0, trajectory.duration])
        np.testing.assert_allclose(
            q[:, 0], [0.0, direction * length], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(v, 0.0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a, 0.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("length", [1000.0, 10000.0])
@pytest.mark.parametrize("acceleration", [1e4, 1e5, 1e6])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_short_acceleration_ramps_after_long_cruise(length, acceleration, direction):
    trajectory = hm.RnTrajectory(
        [[0.0], [direction * length]],
        [0.1],
        [acceleration],
        [1.0],
        profile="trapezoidal",
        blend_tolerance=0.005,
    )
    for slowdown in [1.0, 1.7]:
        if slowdown != 1.0:
            trajectory.set_minimum_duration(slowdown * trajectory.duration)
        report = trajectory.constraint_report(2)
        assert report["within_limits"]
        assert report["velocity_continuous"]
        assert np.max(report["maximum_velocity_jump"]) < 1e-12
        q, v, _, _ = trajectory.sample([0.0, trajectory.duration])
        np.testing.assert_allclose(
            q[:, 0], [0.0, direction * length], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(v, 0.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("slowdown", [1.0, 1.7])
def test_rounded_reversal_keeps_incoming_geometry_until_the_stop(slowdown):
    # The final jerk ramp's position used to round above its prescribed end.
    # That backwards state change discarded the incoming line's ownership
    # and selected the reversed tangent while acceleration was still nonzero.
    points = [
        [-0.012830227537131972, -0.0008199357936686536, -0.0026794418462452456],
        [-0.030407691545095896, -0.0019436288461013383, -0.006349077257330209],
        [-0.020714944055530322, -0.0013245030152944034, -0.004326544377877539],
        [-0.02415063183443394, -0.0015439108018388721, -0.0050434293883954975],
        [-0.029161170967618315, -0.0018632600007437664, -0.006089416874305451],
        [-0.02333757878383428, -0.0014920276539523757, -0.0048736393299556955],
        [-0.005334087734338719, -0.0003419606392368393, -0.0011162732379817078],
        [-0.021969036630447254, -0.0014054178649529382, -0.004590564278935549],
        [-0.03982887213459477, -0.0025480029727611743, -0.00832112615863753],
    ]
    velocity = [8.715212169968956, 0.028442530100144324, 2.5778565454627453e-05]
    acceleration = [1.8094572902058323e-05, 0.8912446705003166, 3.257446352174526]
    jerk = [9.347087455382233, 1.4906924637918515, 3.0515284602807538]
    trajectory = hm.RnTrajectory(
        points,
        velocity,
        acceleration,
        jerk,
        profile="double_s",
        blend_tolerance=0.0002853926670601288,
    )
    trajectory.set_minimum_duration(slowdown * trajectory.duration)
    report = trajectory.constraint_report(2)
    assert report["within_limits"]
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"]
    knots = np.asarray(trajectory.breakpoints)
    spans = np.diff(knots)[:-1]
    offset = np.minimum(
        0.5 * spans,
        np.maximum(1e-8 * spans, 256 * np.finfo(float).eps * np.abs(knots[1:-1])),
    )
    left_times = knots[1:-1] - offset
    left = trajectory.sample(left_times)[2]
    right = trajectory.sample(knots[1:-1])[2]
    elapsed = knots[1:-1] - left_times
    budget = np.outer(elapsed, jerk) + 1e-7 * np.maximum(1.0, acceleration)
    assert np.all(np.abs(right - left) <= budget)
