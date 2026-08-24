import numpy as np
import pytest


def test_robot_loads_explicit_urdf_path(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "two_joint.urdf"
    urdf.write_text(
        """<robot name="two_joint">
        <link name="base"/><link name="link1"/><link name="adapter"/>
        <link name="tool">
          <visual name="tool_visual">
            <origin xyz="0 0 0.2"/>
            <geometry><mesh filename="tool.stl"/></geometry>
            <material name="tool_blue"><color rgba="0 0 1 1"/></material>
          </visual>
          <visual name="tool_box">
            <geometry><box size="0.1 0.2 0.3"/></geometry>
          </visual>
        </link>
        <joint name="joint1" type="revolute">
          <parent link="base"/><child link="link1"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="2" effort="3"/>
        </joint>
        <joint name="adapter_joint" type="fixed">
          <parent link="link1"/><child link="adapter"/>
          <origin xyz="0 0 0.1"/>
        </joint>
        <joint name="joint2" type="prismatic">
          <parent link="adapter"/><child link="tool"/><axis xyz="1 0 0"/>
          <limit lower="0" upper="1" velocity="1" effort="2"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )

    robot = hm.Robot(str(urdf))
    assert robot.name == "two_joint"
    assert robot.dof == 2
    assert robot.model_dof == 2
    assert robot.has_kinematics
    assert robot.root_link_name == "base"
    assert not robot.visuals_loaded
    assert [link.name for link in robot.links] == [
        "base",
        "link1",
        "adapter",
        "tool",
    ]
    assert robot.links[0].child_joints == ["joint1"]
    assert robot.links[1].parent_joint == "joint1"
    assert len(robot.joints) == 3
    assert len(robot.actuated_joints) == 2
    assert len(robot.all_actuated_joints) == 2
    assert robot.get_link("tool").name == "tool"
    assert robot.get_joint("adapter_joint").joint_type == hm.JointType.FIXED
    np.testing.assert_allclose(
        robot.get_joint("adapter_joint").origin[:3, 3], [0, 0, 0.1]
    )
    assert robot.get_link("missing") is None
    assert robot.load_visuals()
    assert robot.visuals_loaded
    visual = robot.links[-1].visuals[0]
    assert visual.name == "tool_visual"
    assert visual.mesh_path == "tool.stl"
    assert visual.material_name == "tool_blue"
    np.testing.assert_allclose(visual.color, [0, 0, 1, 1])
    np.testing.assert_allclose(visual.origin[:3, 3], [0, 0, 0.2])
    primitive = robot.links[-1].visuals[1]
    assert primitive.type == hm.GeometryType.BOX
    np.testing.assert_allclose(primitive.size, [0.1, 0.2, 0.3])
    zero = np.zeros(2)
    pose = robot.kinematics.forward(zero)
    assert pose.shape == (4, 4)
    lower, upper = robot.kinematics.joint_limits
    np.testing.assert_allclose(lower, [-1.0, 0.0])
    np.testing.assert_allclose(upper, [1.0, 1.0])
    assert robot.kinematics.is_reachable(pose)
    robot.kinematics.home_joints = np.array([0.1, 0.2])
    np.testing.assert_allclose(robot.kinematics.home_joints, [0.1, 0.2])
    with pytest.raises(ValueError, match="home_joints"):
        robot.kinematics.home_joints = np.zeros(3)
    assert robot.kinematics.is_reachable(pose)
    assert robot.kinematics.jacobian(zero).shape == (6, 2)
    np.testing.assert_allclose(robot.kinematics.inverse(pose, zero), zero,
                               atol=1e-8)
    with pytest.raises(ValueError):
        robot.kinematics.forward(np.array([np.nan, 0.0]))
    invalid_pose = pose.copy()
    invalid_pose[0, 3] = np.nan
    with pytest.raises(ValueError):
        robot.kinematics.inverse(invalid_pose, zero)
    # Subchains are built from the already parsed Robot tree, not by reading
    # the URDF again.
    urdf.unlink()
    chain = robot.create_kinematics("base", "tool")
    assert chain is not None
    assert chain.dof() == 2
    np.testing.assert_allclose(chain.forward(zero), pose)
    assert robot.create_kinematics("tool", "base") is None


def test_robot_model_does_not_require_supported_kinematic_chain(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "planar.urdf"
    urdf.write_text(
        """<robot name="planar_model">
        <link name="world"/><link name="platform"/>
        <joint name="base_planar" type="planar">
          <parent link="world"/><child link="platform"/><axis xyz="0 0 1"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )

    robot = hm.Robot(str(urdf))
    assert robot.name == "planar_model"
    assert len(robot.links) == 2
    assert len(robot.joints) == 1
    assert robot.joints[0].joint_type == hm.JointType.PLANAR
    assert robot.dof == 0
    assert not robot.has_kinematics
    assert robot.kinematics is None


def test_branched_robot_exposes_complete_dof_and_independent_chains(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "branched.urdf"
    urdf.write_text(
        """<robot name="branched">
        <link name="base"/><link name="left"/><link name="right"/>
        <joint name="left_joint" type="revolute">
          <parent link="base"/><child link="left"/><axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        <joint name="right_joint" type="prismatic">
          <parent link="base"/><child link="right"/><axis xyz="1 0 0"/>
          <limit lower="0" upper="0.5" velocity="1" effort="1"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )

    robot = hm.Robot(str(urdf))
    assert robot.dof == 1
    assert robot.model_dof == 2
    assert {joint.name for joint in robot.all_actuated_joints} == {
        "left_joint",
        "right_joint",
    }
    left = robot.create_kinematics("base", "left")
    right = robot.create_kinematics("base", "right")
    assert left.dof() == 1
    assert right.dof() == 1
    np.testing.assert_allclose(left.forward(np.array([0.0])), np.eye(4))
    translated = right.forward(np.array([0.25]))
    np.testing.assert_allclose(translated[:3, 3], [0.25, 0.0, 0.0])


def test_mimic_joint_is_preserved_but_not_counted_as_independent_dof(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "mimic.urdf"
    urdf.write_text(
        """<robot name="mimic">
        <link name="base"/><link name="driver_link"/><link name="tool"/>
        <joint name="driver" type="revolute">
          <parent link="base"/><child link="driver_link"/>
          <axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
        </joint>
        <joint name="follower" type="revolute">
          <parent link="driver_link"/><child link="tool"/>
          <axis xyz="0 0 1"/>
          <limit lower="-1" upper="1" velocity="1" effort="1"/>
          <mimic joint="driver" multiplier="-1" offset="0.1"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )

    robot = hm.Robot(str(urdf))
    assert robot.model_dof == 1
    assert robot.get_joint("follower").mimic_joint == "driver"
    assert not robot.has_kinematics
    assert robot.create_kinematics("base", "tool") is None
