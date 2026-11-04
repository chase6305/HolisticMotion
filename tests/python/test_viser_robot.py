"""Robot-tree transforms do not load mesh-processing dependencies."""

import builtins
import math
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def robot_helpers(monkeypatch):
    import holistic_motion as hm

    if not hasattr(hm, "JointType"):
        pytest.skip("robot-tree helpers require the native model types")
    import_module = builtins.__import__

    def forbid_trimesh(name, *args, **kwargs):
        if name == "trimesh" or name.startswith("trimesh."):
            raise AssertionError("robot-tree transforms must not import trimesh")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_trimesh)
    from holistic_motion.visualization.viser import robot

    return hm, robot


@pytest.mark.parametrize("kind", ["REVOLUTE", "CONTINUOUS"])
@pytest.mark.parametrize(
    "axis,angle,rotation",
    [
        ([0.0, 0.0, 5.0], 0.0, np.eye(3)),
        (
            [0.0, 0.0, 5.0],
            -math.pi / 2.0,
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        (
            [2.0, 2.0, 2.0],
            2.0 * math.pi / 3.0,
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        (
            [-2.0, -2.0, -2.0],
            -2.0 * math.pi / 3.0,
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
    ],
)
def test_joint_rotation_normalizes_axes_without_trimesh(
    robot_helpers, kind, axis, angle, rotation
):
    hm, helpers = robot_helpers
    axis = np.array(axis)
    original_axis = axis.copy()
    joint = SimpleNamespace(axis=axis, joint_type=getattr(hm.JointType, kind))
    expected = np.eye(4)
    expected[:3, :3] = rotation
    np.testing.assert_allclose(
        helpers._joint_motion(joint, angle), expected, atol=1e-15
    )
    np.testing.assert_array_equal(axis, original_axis)


@pytest.mark.parametrize("kind", ["REVOLUTE", "CONTINUOUS", "FIXED"])
def test_zero_axis_joint_motion_remains_identity(robot_helpers, kind):
    hm, helpers = robot_helpers
    joint = SimpleNamespace(axis=np.zeros(3), joint_type=getattr(hm.JointType, kind))
    np.testing.assert_array_equal(helpers._joint_motion(joint, -0.5), np.eye(4))


def test_prismatic_motion_preserves_axis_scale(robot_helpers):
    hm, helpers = robot_helpers
    joint = SimpleNamespace(
        axis=np.array([-2.0, 3.0, 4.0]), joint_type=hm.JointType.PRISMATIC
    )
    expected = np.eye(4)
    expected[:3, 3] = [0.5, -0.75, -1.0]
    np.testing.assert_array_equal(helpers._joint_motion(joint, -0.25), expected)


@pytest.mark.parametrize("cached_topology", [False, True])
def test_tree_transforms_without_trimesh_preserve_offsets_and_mimic(
    tmp_path, robot_helpers, cached_topology
):
    hm, helpers = robot_helpers
    path = tmp_path / "tree.urdf"
    path.write_text(
        f"""<robot name="tree">
        <link name="base"/><link name="mount"/><link name="arm"/>
        <link name="slide"/><link name="tool"/><link name="copy"/>
        <joint name="mount" type="fixed"><parent link="base"/><child link="mount"/>
          <origin xyz="1 2 0" rpy="0 0 {math.pi / 2.0}"/></joint>
        <joint name="turn" type="continuous"><parent link="mount"/><child link="arm"/>
          <axis xyz="0 0 7"/></joint>
        <joint name="slide" type="prismatic"><parent link="arm"/><child link="slide"/>
          <axis xyz="3 0 0"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
        <joint name="tool" type="fixed"><parent link="slide"/><child link="tool"/>
          <origin xyz="0 1 0"/></joint>
        <joint name="copy" type="revolute"><parent link="base"/><child link="copy"/>
          <axis xyz="0 0 -4"/><limit lower="-2" upper="2" velocity="1" effort="1"/>
          <mimic joint="turn" multiplier="-0.5" offset="0.1"/></joint>
        </robot>"""
    )
    model = hm.RobotModel(str(path))
    positions = {"turn": -math.pi / 2.0, "slide": -0.25, "copy": 1.234}
    original_positions = positions.copy()
    topology = helpers.tree_topology(model) if cached_topology else None
    transforms = helpers.tree_transforms(model, positions, topology)

    np.testing.assert_array_equal(transforms["base"], np.eye(4))
    expected_tool = np.eye(4)
    expected_tool[:3, 3] = [0.75, 3.0, 0.0]
    np.testing.assert_allclose(transforms["tool"], expected_tool, atol=1e-15)
    angle = -(math.pi / 4.0 + 0.1)
    expected_copy = np.eye(4)
    expected_copy[:2, :2] = [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ]
    np.testing.assert_allclose(transforms["copy"], expected_copy, atol=1e-15)
    assert helpers.chain_joint_names(model, "base", "tool") == ["turn", "slide"]
    assert positions == original_positions
