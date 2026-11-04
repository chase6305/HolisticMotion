"""Pure tree metadata is usable without constructing a chain solver."""

import numpy as np
import pytest


def write_model(path):
    path.write_text(
        '''<robot name="metadata">
        <link name="base"><visual><geometry><sphere radius="0.1"/></geometry></visual></link>
        <link name="arm"/><link name="tip"/><link name="free"/><link name="copy"/>
        <joint name="joint" type="revolute"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 2"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
        <joint name="terminal" type="fixed"><parent link="arm"/><child link="tip"/></joint>
        <joint name="floating" type="floating"><parent link="base"/><child link="free"/></joint>
        <joint name="mimic" type="revolute"><parent link="base"/><child link="copy"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
        <mimic joint="joint" multiplier="1" offset="0.2"/></joint></robot>'''
    )


def test_model_queries_include_nonscalar_and_mimic_paths(tmp_path):
    import holistic_motion as hm

    path = tmp_path / "metadata.urdf"
    write_model(path)
    model = hm.RobotModel(str(path))
    robot = hm.Robot(str(path))

    assert not hasattr(model, "kinematics")
    assert model.nq == model.nv == model.model_dof == robot.model_dof == 1
    assert [joint.name for joint in model.chain_joints("base", "tip")] == [
        "joint", "terminal"
    ]
    assert model.chain_joints("base", "free")[0].joint_type == hm.JointType.FLOATING
    assert model.chain_joints("base", "copy")[0].mimic_joint == "joint"
    assert robot.create_kinematics("base", "free") is None
    assert robot.create_kinematics("base", "copy") is None
    assert robot.has_kinematics and robot.dof == 1
    assert robot.chain_joint_names("base", "tip") == ["joint"]
    np.testing.assert_array_equal(model.get_joint("joint").axis, [0.0, 0.0, 1.0])
    mapping = model.create_joint_position_map(["joint"])
    assert mapping.nq == mapping.nv == 1


def test_model_reload_and_visuals_keep_the_loaded_snapshot(tmp_path):
    import holistic_motion as hm

    path = tmp_path / "metadata.urdf"
    write_model(path)
    model = hm.RobotModel(str(path))
    joint = model.get_joint("joint")
    assert not model.load_urdf(str(tmp_path / "missing.urdf"))
    assert model.name == "metadata" and model.get_joint("joint") is joint
    path.write_text('<robot name="invalid"><link name="base"/><link name="tip"/>'
                    '<joint name="bad" type="prismatic"><parent link="base"/>'
                    '<child link="tip"/><axis xyz="0 0 0"/>'
                    '<limit lower="-1" upper="1" effort="1" velocity="1"/>'
                    '</joint></robot>')
    with pytest.raises(ValueError, match="axis"):
        model.load_urdf(str(path))
    assert model.name == "metadata" and model.get_joint("joint") is joint
    path.unlink()
    assert model.load_visuals() and model.visuals_loaded
    assert model.get_link("base").visuals[0].radius == 0.1
