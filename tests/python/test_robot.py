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
    np.testing.assert_allclose(robot.kinematics.inverse(pose, zero), zero, atol=1e-8)
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


def test_default_chain_skips_unsupported_branches_and_preserves_leaf_ties(tmp_path):
    import holistic_motion as hm

    links = ['<link name="base"/>']
    joints = []
    for branch, kinds in (
        ("a", ["prismatic", "prismatic", "fixed"]),
        ("b", ["prismatic", "prismatic", "fixed"]),
        ("c", ["planar", "prismatic", "prismatic", "prismatic"]),
        ("d", ["prismatic", "prismatic", "prismatic"]),
    ):
        for index, kind in enumerate(kinds):
            child = f"{branch}{index}"
            parent = "base" if index == 0 else f"{branch}{index - 1}"
            links.append(f'<link name="{child}"/>')
            mimic = '<mimic joint="a_joint0"/>' if branch == "d" and index == 0 else ""
            joints.append(
                f'<joint name="{branch}_joint{index}" type="{kind}">'
                f'<parent link="{parent}"/><child link="{child}"/>'
                '<origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>'
                '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
                f"{mimic}</joint>"
            )
    path = tmp_path / "branches.urdf"
    path.write_text(
        f'<robot name="branches">{"".join(links)}{"".join(joints)}</robot>',
        encoding="utf-8",
    )
    robot = hm.Robot(str(path))
    assert robot.dof == 2
    assert [joint.name for joint in robot.actuated_joints] == ["a_joint0", "a_joint1"]
    np.testing.assert_allclose(robot.kinematics.forward([0.0, 0.0])[:3, 3], [0, 0, 0.3])
    assert robot.create_kinematics("base", "b2").dof() == 2
    assert robot.create_kinematics("base", "c3") is None
    assert robot.create_kinematics("base", "d2") is None


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


@pytest.mark.parametrize("dof", [1, 3, 6, 7])
def test_numerical_ik_iterates_with_rectangular_jacobian(tmp_path, dof):
    import holistic_motion as hm

    links = "".join(f'<link name="link{i}"/>' for i in range(dof + 1))
    joints = "".join(
        f'<joint name="joint{i}" type="prismatic">'
        f'<parent link="link{i}"/><child link="link{i + 1}"/>'
        '<axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint>"
        for i in range(dof)
    )
    urdf = tmp_path / "rectangular.urdf"
    urdf.write_text(f'<robot name="rectangular">{links}{joints}</robot>')
    solver = hm.Robot(str(urdf)).kinematics
    target = solver.forward(np.full(dof, 0.3 / dof))

    solution = solver.inverse(target, np.zeros(dof))

    assert np.isfinite(solution).all()
    np.testing.assert_allclose(solver.forward(solution), target, atol=5e-4)


def test_prismatic_ik_does_not_generate_angular_equivalents(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "linear.urdf"
    urdf.write_text(
        '<robot name="linear"><link name="base"/><link name="tool"/>'
        '<joint name="slide" type="prismatic">'
        '<parent link="base"/><child link="tool"/><axis xyz="1 0 0"/>'
        '<limit lower="-10" upper="10" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    solver = hm.Robot(str(urdf)).kinematics
    target = solver.forward(np.array([0.5]))

    solutions = solver.solve_all(target, np.array([0.5]))

    assert len(solutions) == 1
    np.testing.assert_allclose(solutions[0], [0.5])


@pytest.mark.parametrize(
    "limits",
    [
        "",
        '<limit velocity="2" effort="3"/>',
        '<limit lower="0" upper="0" velocity="2" effort="3"/>',
    ],
)
def test_continuous_joint_uses_full_turn_position_interval(tmp_path, limits):
    import holistic_motion as hm

    urdf = tmp_path / "continuous.urdf"
    urdf.write_text(
        '<robot name="continuous"><link name="base"/><link name="tool"/>'
        '<joint name="spin" type="continuous">'
        '<parent link="base"/><child link="tool"/><axis xyz="0 0 1"/>'
        f"{limits}</joint></robot>"
    )
    robot = hm.Robot(str(urdf))
    joint = robot.get_joint("spin")
    assert joint.joint_type == hm.JointType.CONTINUOUS
    np.testing.assert_allclose([joint.limit.lower, joint.limit.upper], [-np.pi, np.pi])
    if limits:
        assert joint.limit.max_velocity == 2.0
        assert joint.limit.max_effort == 3.0
    for solver in [robot.kinematics, robot.create_kinematics("base", "tool")]:
        lower, upper = solver.joint_limits
        np.testing.assert_allclose(lower, [-np.pi])
        np.testing.assert_allclose(upper, [np.pi])
        target = solver.forward(np.array([0.4]))
        solution = solver.inverse(target, np.array([0.0]))
        np.testing.assert_allclose(solver.forward(solution), target, atol=2e-3)


@pytest.mark.parametrize("offset", ["0 0 0", "100 -50 3"])
@pytest.mark.parametrize("angle", [0.001, 0.3])
def test_numerical_ik_uses_tool_position_error_in_world_frame(tmp_path, offset, angle):
    import holistic_motion as hm

    urdf = tmp_path / "offset_arm.urdf"
    urdf.write_text(
        '<robot name="offset_arm"><link name="base"/>'
        '<link name="arm"/><link name="tool"/>'
        '<joint name="turn" type="revolute">'
        '<parent link="base"/><child link="arm"/>'
        f'<origin xyz="{offset}" rpy="0.2 -0.3 0.4"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint>'
        '<joint name="tool_offset" type="fixed">'
        '<parent link="arm"/><child link="tool"/><origin xyz="10 0 0"/>'
        "</joint></robot>"
    )
    solver = hm.Robot(str(urdf)).kinematics
    target = solver.forward(np.array([angle]))
    solution = solver.inverse(target, np.array([0.0]))
    recovered = solver.forward(solution)

    assert np.linalg.norm(recovered[:3, 3] - target[:3, 3]) < 5e-4
    np.testing.assert_allclose(recovered[:3, :3], target[:3, :3], atol=2e-3)


@pytest.mark.parametrize(
    "kind,large_joint", [("prismatic", 1e308), ("revolute", 1e200)]
)
@pytest.mark.parametrize("method", ["forward", "forward_all", "jacobian"])
def test_numerical_fk_rejects_arithmetic_overflow(tmp_path, kind, large_joint, method):
    import holistic_motion as hm

    path = tmp_path / "overflow.urdf"
    path.write_text(
        '<robot name="overflow"><link name="base"/><link name="tool"/>'
        f'<joint name="joint" type="{kind}"><parent link="base"/>'
        '<child link="tool"/><axis xyz="2 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>',
        encoding="utf-8",
    )
    solver = hm.Robot(str(path)).kinematics
    expected = solver.forward([0.25])
    with pytest.raises(ValueError):
        getattr(solver, method)([large_joint])
    np.testing.assert_array_equal(solver.forward([0.25]), expected)


@pytest.mark.parametrize("seed", [-2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
def test_numerical_ik_preserves_periodic_limit_endpoints(tmp_path, seed):
    import holistic_motion as hm

    path = tmp_path / "periodic_endpoints.urdf"
    path.write_text(
        '<robot name="periodic"><link name="base"/><link name="tool"/>'
        '<joint name="joint" type="revolute"><parent link="base"/>'
        '<child link="tool"/><axis xyz="0 0 1"/>'
        f'<limit lower="0" upper="{2 * np.pi!r}" velocity="1" effort="1"/>'
        "</joint></robot>",
        encoding="utf-8",
    )
    solver = hm.Robot(str(path)).kinematics
    target = solver.forward([seed])
    solutions = solver.solve_all(target, [seed])
    np.testing.assert_array_equal(sorted(q[0] for q in solutions), [0, 2 * np.pi])
    for solution in solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-12)


@pytest.mark.parametrize("seed", [-1e20, 1e20])
def test_numerical_ik_reduces_extreme_periodic_seed(tmp_path, seed):
    import holistic_motion as hm

    path = tmp_path / "large_seed.urdf"
    path.write_text(
        '<robot name="periodic"><link name="base"/><link name="tool"/>'
        '<joint name="joint" type="revolute"><parent link="base"/>'
        '<child link="tool"/><axis xyz="0 0 1"/>'
        f'<limit lower="{-np.pi!r}" upper="{np.pi!r}" velocity="1" effort="1"/>'
        "</joint></robot>",
        encoding="utf-8",
    )
    solver = hm.Robot(str(path)).kinematics
    target = solver.forward([seed])
    solutions = solver.solve_all(target, [seed])
    assert len(solutions) == 1
    assert -np.pi <= solutions[0][0] <= np.pi
    np.testing.assert_allclose(solver.forward(solutions[0]), target, atol=1e-10)
