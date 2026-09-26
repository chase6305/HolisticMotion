"""A short final leg must propagate a reduced entry speed to earlier phases."""

import holistic_motion as hm
import numpy as np


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
