"""Opt-in NVIDIA cuRobo v2 checks using caller-provided robot/scene assets."""

import json
import os

import numpy as np
import pytest
from holistic_motion.kit.curobo_planner import CuroboPlanner


@pytest.fixture(scope="module")
def problem():
    path = os.environ.get("HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG")
    if not path:
        pytest.skip(
            "Set HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG for real CUDA planning"
        )
    coordinates = []
    for name in ("START", "GOAL"):
        text = os.environ.get("HOLISTICMOTION_TEST_CUROBO_" + name)
        if not text:
            pytest.fail(
                "Explicit HOLISTICMOTION_TEST_CUROBO_" + name + " JSON is required"
            )
        coordinates.append(np.asarray(json.loads(text), dtype=float))
    with CuroboPlanner(path, collision_cache={"cuboid": 4}) as planner:
        planner.warmup(1)
        yield planner, *coordinates


def test_real_curobo_rest_to_rest_motion_and_ownership(problem):
    planner, start, goal = problem
    result = planner.plan(start, goal)
    assert result.success, result.message
    trajectory = result.trajectory
    assert trajectory.joint_names == planner.joint_names
    np.testing.assert_allclose(trajectory.position[0], start, atol=1e-3)
    np.testing.assert_allclose(trajectory.position[-1], goal, atol=1e-3)
    assert trajectory.duration > 0
    assert np.all(np.diff(trajectory.time) > 0)
    assert all(
        np.isfinite(getattr(trajectory, name)).all()
        for name in ("position", "velocity", "acceleration", "jerk")
    )
    snapshot = trajectory.position.copy()
    second = planner.plan(goal, start)
    assert second.success, second.message
    np.testing.assert_array_equal(trajectory.position, snapshot)


def test_real_curobo_world_updates_reject_colliding_requests(problem):
    planner, start, goal = problem
    planner.update_world(
        {
            "cuboid": {
                "blocked": {
                    "dims": [100.0, 100.0, 100.0],
                    "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                }
            }
        }
    )
    try:
        result = planner.plan(start, goal, max_attempts=2)
        assert not result.success
        assert result.trajectory is None
    finally:
        planner.update_world({})
    recovered = planner.plan(start, goal)
    assert recovered.success, recovered.message


def test_real_curobo_pose_planning_uses_named_tool_transforms(problem):
    planner, start, goal = problem
    target = planner.forward_kinematics(goal)
    result = planner.plan_pose(start, target)
    assert result.success, result.message
    final = planner.forward_kinematics(result.trajectory.position[-1])
    for name, transform in target.items():
        np.testing.assert_allclose(final[name][:3, 3], transform[:3, 3], atol=0.005)
        np.testing.assert_allclose(final[name][:3, :3], transform[:3, :3], atol=0.05)
