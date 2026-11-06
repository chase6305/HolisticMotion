"""Opt-in NVIDIA cuRobo v2 checks using caller-provided robot/scene assets."""

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
from holistic_motion.kit.curobo_planner import CuroboPlanner

_COLLISION_CACHE = {
    "cuboid": 4,
    "capsule": 2,
    "cylinder": 2,
    "mesh": 2,
    "voxel": {"layers": 2, "dims": [3.9] * 3, "voxel_size": 0.3},
}


def _free_voxel_scene(features):
    return {
        "voxel": {
            "grid": {
                "dims": [8.0] * 3,
                "voxel_size": 1.0,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": features,
            }
        }
    }


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
    import torch

    with CuroboPlanner(
        path,
        scene=_free_voxel_scene(torch.ones((8, 8, 8)).transpose(0, 1)),
        collision_cache=_COLLISION_CACHE,
    ) as planner:
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


def test_real_curobo_rrt_initializes_a_successful_route(problem):
    planner, start, goal = problem
    result = planner.plan(start, goal, enable_graph_attempt=0)
    assert result.success, result.message
    np.testing.assert_allclose(result.trajectory.position[0], start, atol=1e-3)
    np.testing.assert_allclose(result.trajectory.position[-1], goal, atol=1e-3)


def test_real_curobo_stationary_request_preserves_the_requested_position(problem):
    planner, start, _goal = problem
    result = planner.plan(start, start)
    assert result.success, result.message
    assert result.trajectory.duration >= 0.0
    np.testing.assert_allclose(
        result.trajectory.position,
        np.broadcast_to(start, result.trajectory.position.shape),
        atol=1e-3,
    )


def test_real_curobo_payload_updates_preserve_old_results(problem):
    link = os.environ.get("HOLISTICMOTION_TEST_CUROBO_PAYLOAD_LINK")
    if not link:
        pytest.skip(
            "Set HOLISTICMOTION_TEST_CUROBO_PAYLOAD_LINK for a reserved massless link"
        )
    planner, start, goal = problem
    before = planner.plan(start, goal)
    assert before.success, before.message
    snapshot = before.trajectory.position.copy()
    planner.update_link_inertial(
        link, mass=1.0, com=[0.0, 0.0, 0.1], inertia=[0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
    )
    try:
        loaded = planner.plan(start, goal)
        assert loaded.success, loaded.message
        np.testing.assert_array_equal(before.trajectory.position, snapshot)
    finally:
        planner.update_link_inertial(
            link, mass=0.0, com=[0.0, 0.0, 0.0], inertia=[0.0] * 6
        )


def test_real_curobo_fk_and_torque_against_independent_pinocchio(problem):
    pin = pytest.importorskip("pinocchio")
    import yaml
    from holistic_motion.configuration import JointPositionMap

    config_path = Path(os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kin = config.get("robot_cfg", config)["kinematics"]
    urdf = Path(kin["urdf_path"])
    if not urdf.is_absolute():
        pytest.skip(
            "Independent Pinocchio check requires an explicit absolute URDF path"
        )
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    planner, start, goal = problem
    mapping = JointPositionMap.from_model(model, planner.joint_names)
    reference = pin.neutral(model)
    locked = kin.get("lock_joints", {})
    if locked:
        locked_mapping = JointPositionMap.from_model(model, list(locked))
        reference = locked_mapping.configuration_with_joint_positions(
            reference, list(locked.values())
        )
    base_id = model.getFrameId(kin["base_link"])
    for fraction in np.linspace(0.0, 1.0, 7):
        values = start + (goal - start) * fraction
        full = mapping.configuration_with_joint_positions(reference, values)
        pin.forwardKinematics(model, data, full)
        pin.updateFramePlacements(model, data)
        actual = planner.forward_kinematics(values)
        for name, matrix in actual.items():
            expected = data.oMf[base_id].inverse() * data.oMf[model.getFrameId(name)]
            np.testing.assert_allclose(matrix, expected.homogeneous, atol=2e-6)

    result = planner.plan(start, goal)
    assert result.success, result.message
    indices = [
        model.joints[model.getJointId(name)].idx_v for name in planner.joint_names
    ]
    for position, velocity, acceleration in zip(
        result.trajectory.position,
        result.trajectory.velocity,
        result.trajectory.acceleration,
    ):
        full = mapping.configuration_with_joint_positions(reference, position)
        full_velocity, full_acceleration = np.zeros(model.nv), np.zeros(model.nv)
        full_velocity[indices] = velocity
        full_acceleration[indices] = acceleration
        torque = pin.rnea(model, data, full, full_velocity, full_acceleration)[indices]
        assert np.isfinite(torque).all()
        assert np.all(np.abs(torque) <= model.effortLimit[indices] * 1.001 + 1e-5)


def test_real_curobo_repeated_pose_queries_keep_cuda_memory_bounded(problem):
    import torch

    planner, start, goal = problem
    targets = planner.forward_kinematics(goal)
    for _ in range(2):
        result = planner.plan_pose(start, targets)
        assert result.success, result.message
    torch.cuda.synchronize()
    gc.collect()
    before = torch.cuda.memory_allocated()
    for _ in range(128):
        result = planner.plan_pose(start, targets)
        assert result.success, result.message
    torch.cuda.synchronize()
    gc.collect()
    after = torch.cuda.memory_allocated()
    # The upstream grad-enabled LM input copy retained at least 1 KiB/request
    # on the qualified commit. Allow small lazy allocations, not that growth.
    assert after <= before + 64 * 1024


def test_real_curobo_link_collision_spheres_block_and_restore_planning(problem):
    link = os.environ.get("HOLISTICMOTION_TEST_CUROBO_PAYLOAD_LINK")
    if not link:
        pytest.skip("Set HOLISTICMOTION_TEST_CUROBO_PAYLOAD_LINK for a reserved link")
    planner, start, goal = problem
    previous = planner.plan(start, goal)
    assert previous.success, previous.message
    snapshot = previous.trajectory.position.copy()
    planner.update_link_collision_spheres(link, [[0.0, 0.0, 0.0, 100.0]])
    try:
        for query in (
            lambda: planner.plan(start, goal, max_attempts=2),
            lambda: planner.plan_pose(
                start, planner.forward_kinematics(goal), max_attempts=2
            ),
        ):
            result = query()
            assert not result.success
            assert result.trajectory is None
    finally:
        planner.update_link_collision_spheres(link, None)
    result = planner.plan(start, goal)
    assert result.success, result.message
    np.testing.assert_array_equal(previous.trajectory.position, snapshot)


@pytest.mark.parametrize(
    "kind, geometry",
    [
        ("sphere", {"radius": 100.0}),
        ("capsule", {"radius": 100.0, "base": [0, 0, -1], "tip": [0, 0, 1]}),
        ("cylinder", {"radius": 100.0, "height": 100.0}),
    ],
)
def test_real_curobo_analytic_world_primitives_are_not_omitted(problem, kind, geometry):
    planner, start, goal = problem
    obstacle = {"pose": [0, 0, 0, 1, 0, 0, 0], **geometry}
    planner.update_world({kind: {"blocked": obstacle}})
    try:
        result = planner.plan(start, goal, max_attempts=2)
        assert not result.success
        assert result.trajectory is None
        far = np.eye(4)
        far[:3, 3] = [200.0] * 3
        planner.update_obstacle_pose("blocked", far)
        moved = planner.plan(start, goal)
        assert moved.success, moved.message
        planner.update_obstacle_pose("blocked", np.eye(4))
        restored = planner.plan(start, goal, max_attempts=2)
        assert not restored.success and restored.trajectory is None
    finally:
        planner.update_world({})
    result = planner.plan(start, goal)
    assert result.success, result.message


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
@pytest.mark.parametrize("amp_dtype", [None, "float16", "bfloat16"])
def test_real_curobo_cold_planning_inside_torch_inference_context(
    problem, mode, amp_dtype
):
    import torch

    _existing, start, goal = problem
    dtype = torch.float16 if amp_dtype is None else getattr(torch, amp_dtype)
    with (
        getattr(torch, mode)(),
        torch.autocast("cuda", dtype=dtype, enabled=amp_dtype is not None),
    ):
        caller_state = (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled(),
        )
        with CuroboPlanner(
            os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"],
            scene=_free_voxel_scene(torch.ones((8, 8, 8))),
            collision_cache=_COLLISION_CACHE,
        ) as planner:
            planner.warmup(1)
            joint_result = planner.plan(start, goal)
            assert joint_result.success, joint_result.message
            targets = planner.forward_kinematics(goal)
            pose_result = planner.plan_pose(start, targets)
            assert pose_result.success, pose_result.message
        assert (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled(),
        ) == caller_state


def test_real_curobo_mesh_replacement_and_pose_updates_use_current_geometry(problem):
    import trimesh

    planner, start, goal = problem

    def mesh_scene(size):
        geometry = trimesh.creation.box(extents=[size] * 3)
        return {
            "mesh": {
                "shape": {
                    "vertices": geometry.vertices.tolist(),
                    "faces": geometry.faces.tolist(),
                    "pose": [2, 2, 2, 1, 0, 0, 0],
                }
            }
        }

    try:
        for size, expected_success in ((0.01, True), (100.0, False), (0.01, True)):
            planner.update_world(mesh_scene(size))
            result = planner.plan(start, goal, max_attempts=2)
            assert result.success == expected_success, result.message
        planner.update_world(mesh_scene(100.0))
        far = np.eye(4)
        far[:3, 3] = [200.0] * 3
        planner.update_obstacle_pose("shape", far)
        result = planner.plan(start, goal)
        assert result.success, result.message
        near = np.eye(4)
        near[:3, 3] = [2.0] * 3
        planner.update_obstacle_pose("shape", near)
        result = planner.plan(start, goal, max_attempts=2)
        assert not result.success and result.trajectory is None
    finally:
        planner.update_world({})
    result = planner.plan(start, goal)
    assert result.success, result.message


def test_real_curobo_instances_keep_worlds_dynamics_and_lifetimes_independent(problem):
    planner, start, goal = problem
    previous = planner.plan(start, goal)
    assert previous.success, previous.message
    snapshot = previous.trajectory.position.copy()
    with CuroboPlanner(
        os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"],
        collision_cache=_COLLISION_CACHE,
    ) as other:
        other.warmup(1)
        other.update_world(
            {
                "cuboid": {
                    "blocked": {
                        "dims": [100.0] * 3,
                        "pose": [0, 0, 0, 1, 0, 0, 0],
                    }
                }
            }
        )
        assert not other.plan(start, goal, max_attempts=2).success
        result = planner.plan(start, goal)
        assert result.success, result.message
        other.update_world({})
        link = os.environ.get("HOLISTICMOTION_TEST_CUROBO_PAYLOAD_LINK")
        if link:
            other.update_link_inertial(
                link, mass=1000.0, com=[0, 0, 0.1], inertia=[1, 1, 1, 0, 0, 0]
            )
            assert not other.plan(start, goal, max_attempts=2).success
            result = planner.plan(start, goal)
            assert result.success, result.message
    result = planner.plan(start, goal)
    assert result.success, result.message
    np.testing.assert_array_equal(previous.trajectory.position, snapshot)


@pytest.mark.parametrize("with_voxel_scene", [False, True])
def test_real_curobo_rejects_incompatible_live_capture_layout_before_kernel_launch(
    problem, with_voxel_scene
):
    import torch

    planner, start, goal = problem
    cache = {"cuboid": 2}
    scene = None
    if with_voxel_scene:
        cache = dict(_COLLISION_CACHE)
        cache["voxel"] = dict(cache["voxel"], layers=3)
        scene = _free_voxel_scene(torch.ones((8, 8, 8)))
    with pytest.raises(RuntimeError, match="share a collision cache layout"):
        CuroboPlanner(
            os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"],
            collision_cache=cache,
            scene=scene,
        )
    result = planner.plan(start, goal)
    assert result.success, result.message


def test_real_curobo_world_updates_finish_before_another_stream_plans(problem):
    import torch

    planner, start, goal = problem
    edit_stream, plan_stream = torch.cuda.Stream(), torch.cuda.Stream()
    try:
        with torch.cuda.stream(edit_stream):
            planner.update_world(
                {
                    "cuboid": {
                        "blocked": {
                            "dims": [100.0] * 3,
                            "pose": [0, 0, 0, 1, 0, 0, 0],
                        }
                    }
                }
            )
        with torch.cuda.stream(plan_stream):
            result = planner.plan(start, goal, max_attempts=2)
            assert not result.success and result.trajectory is None
        far = np.eye(4)
        far[:3, 3] = [200.0] * 3
        with torch.cuda.stream(edit_stream):
            planner.update_obstacle_pose("blocked", far)
        with torch.cuda.stream(plan_stream):
            result = planner.plan(start, goal)
            assert result.success, result.message
    finally:
        planner.update_world({})


@pytest.mark.parametrize(
    "scene, message",
    [
        ({"mesh": {"unplaced": {}}}, "explicit.*pose"),
        (
            {"cuboid": {"box": {"dims": [1e-50, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}}},
            "dims rounds to zero",
        ),
        (
            {"mesh": {"mesh": {"scale": [1, 1e-50, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}}},
            "scale rounds to zero",
        ),
    ],
)
def test_real_curobo_invalid_geometry_preserves_the_existing_world(
    problem, scene, message
):
    planner, start, goal = problem
    with pytest.raises(ValueError, match=message):
        planner.update_world(scene)
    result = planner.plan(start, goal)
    assert result.success, result.message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 70000.0])
def test_real_curobo_invalid_voxel_distances_preserve_the_existing_world(
    problem, value
):
    import torch

    planner, start, goal = problem
    scene = {
        "voxel": {
            "grid": {
                "dims": [1, 1, 1],
                "voxel_size": 0.5,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": torch.full((8,), value),
            }
        }
    }
    with pytest.raises(ValueError, match="ESDF"):
        planner.update_world(scene)
    result = planner.plan(start, goal)
    assert result.success, result.message


@pytest.mark.parametrize(
    "representation, public_scene",
    [
        ("cpu_float32", False),
        ("cuda_nonleaf", False),
        ("cpu_strided", True),
        ("cuda_float64", True),
    ],
)
def test_real_curobo_voxel_input_storage_and_ownership(
    problem, representation, public_scene
):
    import torch

    planner, start, goal = problem
    values = torch.ones((8, 8, 8))
    if representation == "cuda_nonleaf":
        values = values.cuda().requires_grad_(True) * 2
    elif representation == "cpu_strided":
        values = values.half().transpose(0, 1)
    elif representation == "cuda_float64":
        values = values.double().cuda()
    scene = _free_voxel_scene(values)
    if public_scene:
        from curobo.scene import Scene

        scene = Scene.create(scene)
    try:
        planner.update_world(scene)
        result = planner.plan(start, goal)
        assert result.success, result.message
        snapshot = result.trajectory.position.copy()
        with torch.inference_mode():
            values.fill_(-1)
        preserved = planner.plan(start, goal)
        assert preserved.success, preserved.message
        planner.update_world(scene)
        blocked = planner.plan(start, goal, max_attempts=2)
        assert not blocked.success and blocked.trajectory is None
        np.testing.assert_array_equal(result.trajectory.position, snapshot)
    finally:
        planner.update_world({})
    assert planner.plan(start, goal).success


@pytest.mark.parametrize("count", [1, 511, 513])
def test_real_curobo_invalid_voxel_counts_preserve_an_occupied_world(problem, count):
    import torch

    planner, start, goal = problem
    occupied = _free_voxel_scene(-torch.ones((8, 8, 8)))
    malformed = _free_voxel_scene(torch.ones(count))
    planner.update_world(occupied)
    try:
        assert not planner.plan(start, goal, max_attempts=2).success
        with pytest.raises(ValueError, match="feature count.*grid shape"):
            planner.update_world(malformed)
        blocked = planner.plan(start, goal, max_attempts=2)
        assert not blocked.success and blocked.trajectory is None
        with pytest.raises(ValueError, match="feature count.*grid shape"):
            CuroboPlanner(
                os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"],
                scene=malformed,
                collision_cache=_COLLISION_CACHE,
            )
    finally:
        planner.update_world({})
    assert planner.plan(start, goal).success


@pytest.mark.parametrize("count, size", [(13, 0.1), (3, 0.3)])
def test_real_curobo_voxel_grid_indexing_uses_integer_counts(problem, count, size):
    import torch

    planner, start, goal = problem
    scene = {
        "voxel": {
            "grid": {
                "dims": [count * size] * 3,
                "voxel_size": size,
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "feature_tensor": torch.ones((count,) * 3),
            }
        }
    }
    with CuroboPlanner(
        os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"],
        scene=scene,
        collision_cache=_COLLISION_CACHE,
    ) as other:
        data = other._planner.scene_collision_checker.data.voxels
        original_buffer = data.params.data_ptr()
        for replace in (False, True):
            if replace:
                other.update_world(scene)
            assert data.params.data_ptr() == original_buffer
            assert data.params[0, 0, :3].to(torch.int64).tolist() == [count] * 3
            assert other.plan(start, goal).success
    assert planner.plan(start, goal).success


@pytest.mark.parametrize("explicit_cache", [True, False])
def test_real_curobo_initial_voxels_preserve_reserved_capacity(problem, explicit_cache):
    import torch

    _planner, start, goal = problem
    small = {
        "dims": [0.9] * 3,
        "voxel_size": 0.3,
        "pose": [0, 0, 0, 1, 0, 0, 0],
        "feature_tensor": torch.ones((3,) * 3),
    }
    large = dict(small, dims=[3.9] * 3, feature_tensor=torch.ones((13,) * 3))
    scene = {"voxel": {"small": small}}
    cache = dict(_COLLISION_CACHE)
    if not explicit_cache:
        cache.pop("voxel")
        scene["voxel"]["large"] = large
    with CuroboPlanner(
        os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"],
        scene=scene,
        collision_cache=cache,
    ) as other:
        data = other._planner.scene_collision_checker.data.voxels
        assert data.max_n == 2 and data.features.shape[2] == 13**3
        pointers = (data.params.data_ptr(), data.features.data_ptr())
        other.update_world({"voxel": {"small": small, "large": large}})
        assert (data.params.data_ptr(), data.features.data_ptr()) == pointers
        result = other.plan(start, goal)
        assert result.success, result.message
        snapshot = result.trajectory.position.copy()
        other.update_world({"voxel": {"small": small}})
        assert other.plan(start, goal).success
        np.testing.assert_array_equal(result.trajectory.position, snapshot)


@pytest.mark.parametrize("overflow", ["layers", "cells"])
def test_real_curobo_voxel_capacity_errors_keep_occupied_world(problem, overflow):
    import torch

    planner, start, goal = problem
    grid = {
        "dims": [3.9] * 3,
        "voxel_size": 0.3,
        "pose": [0, 0, 0, 1, 0, 0, 0],
        "feature_tensor": -torch.ones((13,) * 3),
    }
    planner.update_world({"voxel": {"occupied": grid}})
    try:
        if overflow == "layers":
            replacement = {"voxel": {str(i): grid for i in range(3)}}
        else:
            replacement = {
                "voxel": {
                    "grid": dict(
                        grid, dims=[4.2] * 3, feature_tensor=torch.ones((14,) * 3)
                    )
                }
            }
        with pytest.raises(ValueError, match="voxel.*capacity"):
            planner.update_world(replacement)
        result = planner.plan(start, goal, max_attempts=2)
        assert not result.success and result.trajectory is None
    finally:
        planner.update_world({})
    assert planner.plan(start, goal).success


def test_real_curobo_unreserved_voxel_update_keeps_an_empty_world(problem):
    import torch

    _planner, start, goal = problem
    with CuroboPlanner(os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"]) as other:
        with pytest.raises(ValueError, match="voxel.*not allocated"):
            other.update_world(_free_voxel_scene(torch.ones((8, 8, 8))))
        result = other.plan(start, goal)
        assert result.success, result.message


@pytest.mark.parametrize("kind", ["cuboid", "sphere", "capsule", "cylinder", "mesh"])
def test_real_curobo_scene_capacity_errors_preserve_world_and_mesh_cache(problem, kind):
    import trimesh

    planner, start, goal = problem
    box = trimesh.creation.box(extents=[0.1] * 3)
    geometry = {
        "cuboid": {"dims": [0.1] * 3},
        "sphere": {"radius": 0.1},
        "capsule": {"radius": 0.1, "base": [0, 0, 0], "tip": [0, 0, 1]},
        "cylinder": {"radius": 0.1, "height": 0.2},
        "mesh": {"vertices": box.vertices.tolist(), "faces": box.faces.tolist()},
    }
    far_pose = [200, 200, 200, 1, 0, 0, 0]
    planner.update_world(
        {
            "cuboid": {
                "occupied": {"dims": [100.0] * 3, "pose": [0, 0, 0, 1, 0, 0, 0]}
            },
            "mesh": {"retained": dict(geometry["mesh"], pose=far_pose)},
        }
    )
    try:
        data = planner._planner.scene_collision_checker.data
        names = data.get_obstacle_names()
        cached_meshes = data.meshes.get_cached_mesh_names()
        assert cached_meshes
        count = _COLLISION_CACHE["capsule" if kind == "sphere" else kind] + 1
        bad = {
            kind: {
                str(index): dict(geometry[kind], pose=far_pose)
                for index in range(count)
            }
        }
        with pytest.raises(ValueError, match="capacity"):
            planner.update_world(bad)
        assert data.get_obstacle_names() == names
        assert data.meshes.get_cached_mesh_names() == cached_meshes
        result = planner.plan(start, goal, max_attempts=2)
        assert not result.success and result.trajectory is None
        far = np.eye(4)
        far[:3, 3] = 200
        planner.update_obstacle_pose("occupied", far)
        moved = planner.plan(start, goal)
        assert moved.success, moved.message
    finally:
        planner.update_world({})


def test_real_curobo_unreserved_geometry_keeps_an_empty_world(problem):
    _planner, start, goal = problem
    geometry = {
        "cuboid": {"dims": [0.1] * 3},
        "sphere": {"radius": 0.1},
        "capsule": {"radius": 0.1, "base": [0, 0, 0], "tip": [0, 0, 1]},
        "cylinder": {"radius": 0.1, "height": 0.2},
        "mesh": {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]]},
    }
    with CuroboPlanner(os.environ["HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG"]) as other:
        for kind, fields in geometry.items():
            with pytest.raises(ValueError, match="not allocated"):
                other.update_world(
                    {kind: {"unreserved": dict(fields, pose=[0, 0, 0, 1, 0, 0, 0])}}
                )
            result = other.plan(start, goal)
            assert result.success, result.message
