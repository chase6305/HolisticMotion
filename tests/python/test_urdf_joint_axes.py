"""Native and Python model importers use scalar axes as unit directions."""

import numpy as np
import pytest
from optional_dependencies import python_pinocchio_or_skip


OFFSET = np.array([0.4, -0.2, 0.3])
OBSTACLE = np.array([1.5, -0.4, 0.2])


def write_model(tmp_path, kind, axis, *, namespace=False):
    axis_element = "" if axis is None else f'<axis xyz="{axis}"/>'
    namespace = ' xmlns="urn:robot-description"' if namespace else ""
    path = tmp_path / "directions.urdf"
    path.write_text(
        f'<robot name="directions"{namespace}>'
        '<link name="base"><collision><origin xyz="1.5 -0.4 0.2"/>'
        '<geometry><sphere radius="0.1"/></geometry></collision></link>'
        '<link name="tip"><collision><origin xyz="0.4 -0.2 0.3"/>'
        '<geometry><sphere radius="0.1"/></geometry></collision></link>'
        f'<joint name="joint" type="{kind}"><parent link="base"/>'
        '<child link="tip"/><origin xyz="0.2 -0.1 0.3" '
        f'rpy="0 0 {np.pi / 2!r}"/>{axis_element}'
        '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        '<ros2_control name="metadata"><joint name="ignored" type="revolute">'
        '<axis xyz="0 0 0"/></joint></ros2_control></robot>'
    )
    return path


def expected_pose(kind, coordinate=0.7, axis=(0.0, 0.6, 0.8)):
    axis = np.asarray(axis)
    origin = np.array(
        [
            [0.0, -1.0, 0.0, 0.2],
            [1.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    motion = np.eye(4)
    if kind == "prismatic":
        motion[:3, 3] = coordinate * axis
    else:
        skew = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        motion[:3, :3] += np.sin(coordinate) * skew + (1 - np.cos(coordinate)) * (
            skew @ skew
        )
    return origin @ motion


def native_model(path, backend):
    import holistic_motion as hm

    if backend == "robot":
        if not hasattr(hm, "Robot"):
            pytest.skip("requires the native core")
        return hm.Robot(str(path))
    if not hasattr(hm, "CollisionModel"):
        pytest.skip("requires native collision support")
    if backend == "mesh":
        return hm.CollisionModel(str(path), exclude_adjacent=False)
    return hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("obstacle", "base", OBSTACLE, 0.1),
            hm.CollisionSphere("tool", "tip", OFFSET, 0.1),
        ],
    )


@pytest.mark.parametrize("backend", ["robot", "mesh", "sphere"])
@pytest.mark.parametrize("kind", ["revolute", "continuous", "prismatic"])
@pytest.mark.parametrize("scale", [1.5, 1e200, 1e-200])
def test_native_importers_preserve_joint_units_at_all_axis_scales(
    tmp_path, backend, kind, scale
):
    path = write_model(tmp_path, kind, f"0 {0.6 * scale} {0.8 * scale}")
    original = path.read_bytes()
    model = native_model(path, backend)
    expected = expected_pose(kind)
    if backend == "robot":
        np.testing.assert_allclose(
            model.kinematics.forward([0.7]), expected, atol=1e-14
        )
    else:
        q = model.configuration_from_joint_positions({"joint": 0.7})
        center = expected[:3, 3] + expected[:3, :3] @ OFFSET
        assert model.minimum_distance(q).distance == pytest.approx(
            np.linalg.norm(center - OBSTACLE) - 0.2, abs=2e-12
        )
        if backend == "sphere":
            np.testing.assert_allclose(
                model.world_spheres(q)[1, :3], center, atol=1e-14
            )
    assert path.read_bytes() == original


INVALID_AXES = [
    ("revolute", "0 0 0"),
    ("continuous", "0 0 0"),
    ("prismatic", "0 0 0"),
    ("revolute", "nan 0 1"),
    ("revolute", "inf 0 1"),
]


@pytest.mark.parametrize("kind", ["revolute", "continuous"])
@pytest.mark.parametrize("angle", [1e20, -1e200, 1e308])
@pytest.mark.parametrize(
    "axis",
    [
        (0.30471707975443135, -1.0399841062404955, 0.7504511958064572),
        (-0.6809295444039414, 1.2225413386740303, -0.15452948206880215),
        (1.0, 1.0, 1.0),
    ],
)
def test_imported_rotation_keeps_unit_angle_at_extreme_coordinates(
    tmp_path, kind, angle, axis
):
    path = write_model(tmp_path, kind, " ".join(map(str, axis)))
    robot = native_model(path, "robot")
    unit = np.asarray(axis) / np.linalg.norm(axis)
    expected = expected_pose(kind, angle, axis=unit)
    pose = robot.kinematics.forward([angle])
    np.testing.assert_allclose(pose, expected, atol=2e-14)
    import holistic_motion as hm

    if hasattr(hm, "SphereCollisionModel"):
        sphere = native_model(path, "sphere")
        q = sphere.configuration_from_joint_positions({"joint": angle})
        np.testing.assert_allclose(
            sphere.world_spheres(q)[1, :3],
            pose[:3, 3] + pose[:3, :3] @ OFFSET,
            atol=2e-14,
        )


@pytest.mark.parametrize("backend", ["robot", "mesh", "sphere"])
@pytest.mark.parametrize("kind,axis", INVALID_AXES)
def test_native_importers_reject_invalid_scalar_axes(tmp_path, backend, kind, axis):
    path = write_model(tmp_path, kind, axis)
    with pytest.raises((ValueError, RuntimeError), match="axis|URDF|urdf"):
        native_model(path, backend)


@pytest.mark.parametrize("backend", ["robot", "mesh", "sphere"])
@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
@pytest.mark.parametrize("branched", [False, True])
def test_native_importers_reject_reversed_limits_on_every_branch(
    tmp_path, backend, kind, branched
):
    path = write_model(tmp_path, kind, "1 0 0")
    source = path.read_text().replace('lower="-2" upper="2"', 'lower="1" upper="-1"')
    if branched:
        # The longer valid branch becomes Robot's default chain. Invalid
        # metadata on another branch still invalidates the complete model.
        branch = '<link name="long1"/><link name="long2"/>'
        for index, parent in enumerate(["base", "long1"], 1):
            branch += (
                f'<joint name="long{index}" type="prismatic">'
                f'<parent link="{parent}"/><child link="long{index}"/>'
                '<axis xyz="1 0 0"/>'
                '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint>'
            )
        source = source.replace("</robot>", branch + "</robot>")
    path.write_text(source)
    with pytest.raises(ValueError, match="position limits must be ordered"):
        native_model(path, backend)


@pytest.mark.parametrize("backend", ["robot", "mesh", "sphere"])
@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_native_importers_accept_locked_position_limits(tmp_path, backend, kind):
    path = write_model(tmp_path, kind, "1 0 0")
    path.write_text(
        path.read_text().replace('lower="-2" upper="2"', 'lower="0.4" upper="0.4"')
    )
    model = native_model(path, backend)
    expected = expected_pose(kind, 0.4, axis=(1.0, 0.0, 0.0))
    if backend == "robot":
        np.testing.assert_allclose(
            model.kinematics.forward([0.4]), expected, atol=1e-14
        )
    else:
        np.testing.assert_array_equal(model.joint_lower_limits(["joint"]), [0.4])
        np.testing.assert_array_equal(model.joint_upper_limits(["joint"]), [0.4])
        q = model.configuration_from_joint_positions({"joint": 0.4})
        center = expected[:3, 3] + expected[:3, :3] @ OFFSET
        assert model.minimum_distance(q).distance == pytest.approx(
            np.linalg.norm(center - OBSTACLE) - 0.2, abs=2e-12
        )


@pytest.mark.parametrize("backend", ["robot", "mesh", "sphere"])
def test_native_continuous_joints_ignore_position_limit_metadata(tmp_path, backend):
    path = write_model(tmp_path, "continuous", "1 0 0")
    path.write_text(
        path.read_text().replace('lower="-2" upper="2"', 'lower="1" upper="-1"')
    )
    model = native_model(path, backend)
    if backend == "robot":
        pose = model.kinematics.forward([0.7])
        np.testing.assert_allclose(
            pose, expected_pose("continuous", axis=(1.0, 0.0, 0.0)), atol=1e-14
        )
    else:
        np.testing.assert_allclose(model.joint_lower_limits(["joint"]), [-np.pi])
        np.testing.assert_allclose(model.joint_upper_limits(["joint"]), [np.pi])


def python_solver(path):
    python_pinocchio_or_skip()
    from holistic_motion.kit.retargeting import (
        PinocchioRetargetingSolver,
        RetargetingModeManager,
    )

    return PinocchioRetargetingSolver(
        path,
        {"left_hand": "tip"},
        {"left_arm": ["joint"]},
        mode_manager=RetargetingModeManager(initial_mode="left_arm"),
    )


@pytest.mark.parametrize("kind", ["revolute", "continuous", "prismatic"])
@pytest.mark.parametrize("scale", [1.5, 1e200, 1e-200])
def test_python_retargeting_import_preserves_joint_units(tmp_path, kind, scale):
    path = write_model(tmp_path, kind, f"0 {0.6 * scale} {0.8 * scale}")
    original = path.read_bytes()
    solver = python_solver(path)
    q = solver.create_joint_position_map(["joint"]).configuration_with_joint_positions(
        solver._neutral_q, [0.7]
    )
    result = solver.solve({"left_hand": expected_pose(kind)}, seed=q)
    assert result.success and result.iterations == 1
    assert result.residual < 1e-12
    np.testing.assert_allclose(result.configuration, q, atol=1e-14)
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind,axis", INVALID_AXES)
def test_python_retargeting_import_rejects_invalid_axes(tmp_path, kind, axis):
    path = write_model(tmp_path, kind, axis)
    with pytest.raises(ValueError, match="axis"):
        python_solver(path)


def test_python_retargeting_keeps_default_namespace_and_missing_axis_semantics(
    tmp_path,
):
    path = write_model(tmp_path, "revolute", None, namespace=True)
    solver = python_solver(path)
    result = solver.solve(
        {"left_hand": expected_pose("revolute", axis=(1.0, 0.0, 0.0))}, seed=[0.7]
    )
    assert result.success and result.iterations == 1


def test_python_retargeting_rejects_reversed_position_limits(tmp_path):
    path = write_model(tmp_path, "prismatic", "1 0 0")
    path.write_text(
        path.read_text().replace('lower="-2" upper="2"', 'lower="1" upper="-1"')
    )
    with pytest.raises(ValueError, match="position limits must be ordered"):
        python_solver(path)


def test_python_retargeting_keeps_locked_position_limits(tmp_path):
    path = write_model(tmp_path, "prismatic", "1 0 0")
    path.write_text(
        path.read_text().replace('lower="-2" upper="2"', 'lower="0.4" upper="0.4"')
    )
    solver = python_solver(path)
    result = solver.solve(
        {"left_hand": expected_pose("prismatic", 0.4, axis=(1.0, 0.0, 0.0))}
    )
    assert result.success
    np.testing.assert_array_equal(result.configuration, [0.4])
