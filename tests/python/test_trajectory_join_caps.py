"""A blended join must honor the velocity caps of both neighboring lines."""

import json
from pathlib import Path

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "fixture", ["trapezoidal_cap_rejection.json", "double_s_join_cap.json"]
)
@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_blend_velocity_respects_adjacent_lines(fixture, scale, reverse):
    inputs = json.loads(
        (Path(__file__).resolve().parents[2] / "benchmarks/fixtures" / fixture).read_text()
    )
    points = np.asarray(inputs["waypoints"]) * scale
    inputs["waypoints"] = points[::-1] if reverse else points
    for name in ("max_velocity", "max_acceleration", "max_jerk"):
        inputs[name] = np.asarray(inputs[name]) * scale
    inputs["blend_tolerance"] *= scale
    trajectory = hm.RnTrajectory(**inputs)
    times = np.unique(
        np.concatenate(
            [
                np.linspace(start, end, 257)
                for start, end in zip(
                    trajectory.breakpoints[:-1], trajectory.breakpoints[1:]
                )
            ]
        )
    )
    states = trajectory.sample(times)
    assert all(np.isfinite(state).all() for state in states)
    np.testing.assert_allclose(
        states[0][[0, -1]], inputs["waypoints"][[0, -1]], rtol=1e-12, atol=scale * 1e-10
    )
    names = ["max_velocity", "max_acceleration"]
    if inputs["profile"] == "double_s":
        names.append("max_jerk")
    for values, name in zip(states[1:], names):
        assert np.max(np.abs(values) / inputs[name]) <= 1.0 + 1e-8
    report = trajectory.constraint_report(10001)
    assert report["within_limits"]
    assert report["velocity_continuous"]
    if inputs["profile"] == "double_s":
        assert report["acceleration_continuous"]
