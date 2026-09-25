"""Blended paths must preserve limits and continuity at every geometric join."""

import json
from pathlib import Path

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "fixture",
    [
        "trapezoidal_cap_rejection.json",
        "double_s_join_cap.json",
    ],
)
@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_blended_joins_preserve_limits_and_continuity(fixture, scale, reverse):
    inputs = json.loads(
        (
            Path(__file__).resolve().parents[2] / "benchmarks/fixtures" / fixture
        ).read_text()
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


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_final_jerk_phase_keeps_incoming_line_at_stop(scale):
    inputs = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks/fixtures/double_s_linear_join_continuity.json"
        ).read_text()
    )
    for name in ("waypoints", "max_velocity", "max_acceleration", "max_jerk"):
        inputs[name] = np.asarray(inputs[name]) * scale
    inputs["blend_tolerance"] *= scale
    trajectory = hm.RnTrajectory(**inputs)
    # The incoming final jerk phase travels only a few coordinate ulps.
    # Its terminal state belongs to the next line, whose tangent is reversed.
    report = trajectory.constraint_report(20001)
    assert report["within_limits"]
    assert report["velocity_continuous"]
    assert report["acceleration_continuous"]
    np.testing.assert_allclose(
        trajectory.sample([0.0, trajectory.duration])[0],
        np.asarray(inputs["waypoints"])[[0, -1]],
        rtol=0.0,
        atol=scale * 1e-12,
    )
