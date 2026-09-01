import numpy as np
import pytest


def _write_seven_revolute_urdf(path):
    axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"]
    links = "".join(f'<link name="link{i}"/>' for i in range(8))
    joints = []
    for i, axis in enumerate(axes):
        joints.append(f"""<joint name="joint{i}" type="revolute">
          <parent link="link{i}"/><child link="link{i + 1}"/>
          <origin xyz="0 0 0.2"/><axis xyz="{axis}"/>
          <limit lower="-3" upper="3" velocity="2" effort="10"/>
        </joint>""")
    path.write_text(
        f'<robot name="seven_r">{links}{"".join(joints)}</robot>',
        encoding="utf-8",
    )


def _write_ideal_srs_urdf(path):
    origins = [
        ("0 0 0", "0 0 0", "0 0 1"),
        ("0 0 0.1025", "1.5707963268 0 3.1415926536", "0 0 1"),
        ("0 0.260 0", "1.5707963268 -1.5707963268 3.1415926536", "0 0 1"),
        ("0 0 0", "1.5707963268 0 0", "0 0 1"),
        ("0 0.166 0", "-1.5707963268 3.1415926536 0", "0 0 1"),
        ("0 0 0.098", "1.5707963268 0 0", "0 0 -1"),
        ("0 0 0", "-1.5707963268 3.1415926536 1.5707963268", "0 0 1"),
    ]
    links = "".join(f'<link name="link{i}"/>' for i in range(9))
    joints = []
    for i, (xyz, rpy, axis) in enumerate(origins):
        joints.append(f'''<joint name="joint{i}" type="revolute">
          <parent link="link{i}"/><child link="link{i + 1}"/>
          <origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{axis}"/>
          <limit lower="-3" upper="3" velocity="2" effort="10"/>
        </joint>''')
    joints.append("""<joint name="tool" type="fixed">
      <parent link="link7"/><child link="link8"/>
      <origin xyz="-0.066 0 0" rpy="0 -1.5707963268 0"/>
    </joint>""")
    path.write_text(
        f'<robot name="ideal_srs">{links}{"".join(joints)}</robot>',
        encoding="utf-8",
    )


def _write_offset_srs_urdf(path):
    origins = [
        ("0 0 0", "-1.5707963268 0 -1.5707963268"),
        ("0 0 0", "1.5707963268 0 0"),
        ("0 0.287 0", "-1.5707963268 0 0"),
        ("0.018 0 0", "-1.5707963268 0 3.1415926536"),
        ("0.018 -0.314 0", "-1.5707963268 0 3.1415926536"),
        ("0 0 0", "1.5707963268 -1.5707963268 0"),
        ("0 0 0", "1.5707963268 -1.5707963268 0"),
    ]
    links = "".join(f'<link name="link{i}"/>' for i in range(9))
    joints = []
    for i, (xyz, rpy) in enumerate(origins):
        joints.append(f'''<joint name="joint{i}" type="revolute">
          <parent link="link{i}"/><child link="link{i + 1}"/>
          <origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="0 0 1"/>
          <limit lower="-3.1" upper="3.1" velocity="2" effort="10"/>
        </joint>''')
    joints.append("""<joint name="tool" type="fixed">
      <parent link="link7"/><child link="link8"/>
      <origin xyz="0 -0.095 0" rpy="1.5707963268 -1.5707963268 0"/>
    </joint>""")
    path.write_text(
        f'<robot name="offset_srs">{links}{"".join(joints)}</robot>',
        encoding="utf-8",
    )


def test_srs_null_space_projection_and_path(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "seven_r.urdf"
    _write_seven_revolute_urdf(urdf)
    robot = hm.Robot(str(urdf))
    assert isinstance(robot.kinematics, hm.SRSKinematics)
    assert robot.kinematics.compatible
    geometry = robot.kinematics.analyze_geometry()
    assert geometry.structurally_compatible
    assert not geometry.closed_form_compatible
    assert geometry.shoulder_axis_residual > 0.0

    joints = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
    direction = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    velocity = robot.kinematics.null_space_velocity(joints, direction)
    jacobian = robot.kinematics.jacobian(joints)
    assert velocity.shape == (7,)
    assert np.linalg.norm(jacobian @ velocity) < 1e-10
    with pytest.raises(ValueError):
        robot.kinematics.null_space_velocity(np.full(7, np.nan), direction)
    with pytest.raises(ValueError):
        robot.kinematics.null_space_velocity(joints, np.full(7, np.inf))

    target = robot.kinematics.forward(joints)
    planner = hm.NullSpacePlanner(robot.kinematics)
    path = planner.plan(joints, direction, 2, 0.01)
    assert len(path) == 3
    for point in path:
        np.testing.assert_allclose(robot.kinematics.forward(point), target, atol=1e-3)
    with pytest.raises(ValueError, match="planning failed"):
        planner.plan(joints, direction, 2, float("nan"))
    with pytest.raises(ValueError, match="planning failed"):
        planner.plan(joints, np.full(7, np.inf), 2, 0.01)


def test_srs_distinct_solve_methods(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "seven_r.urdf"
    _write_seven_revolute_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    joints = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
    target = solver.forward(joints)

    config = solver.configuration(joints)
    assert (config.shoulder, config.elbow, config.wrist) == (-1, -1, 1)
    assert config.redundancy == joints[2]

    for method in (
        hm.SRSSolveMethod.SEEDED_NUMERICAL,
        hm.SRSSolveMethod.CONFIGURATION,
        hm.SRSSolveMethod.ALL_CONFIGURATIONS,
        hm.SRSSolveMethod.NEAREST_REDUNDANCY,
    ):
        solutions = solver.solve(target, joints, method)
        assert solutions
        np.testing.assert_allclose(solver.forward(solutions[0]), target, atol=1e-3)

    configured = solver.solve_configuration(target, config, joints)
    np.testing.assert_allclose(solver.forward(configured), target, atol=1e-3)

    report = solver.solve_detailed(
        target, joints, hm.SRSSolveMethod.ALL_CONFIGURATIONS
    )
    assert report.status == hm.SRSSolveStatus.SUCCESS
    assert report.method == hm.SRSSolveMethod.ALL_CONFIGURATIONS
    assert len(report.solutions) == len(report.configurations)
    assert len(report.solutions) == len(report.minimum_singular_values)
    assert len(report.solutions) == len(report.minimum_joint_limit_margins)
    assert len(report.solutions) == len(report.joint_limit_margins)
    assert len(report.solutions) == len(report.near_singularities)
    assert len(report.solutions) == len(report.joint_limit_hits)
    assert all(value >= 0.0 for value in report.minimum_singular_values)
    assert all(value >= 0.0 for value in report.minimum_joint_limit_margins)
    assert all(
        np.asarray(value).shape == (7,) for value in report.joint_limit_margins
    )

    invalid = solver.solve_detailed(target, np.zeros(6))
    assert invalid.status == hm.SRSSolveStatus.INVALID_INPUT
    assert not invalid.solutions
    malformed_target = target.copy()
    malformed_target[3, 3] = 2.0
    invalid = solver.solve_detailed(malformed_target, joints)
    assert invalid.status == hm.SRSSolveStatus.INVALID_INPUT


def test_srs_explicit_tool_and_user_frame_boundaries(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "seven_r.urdf"
    _write_seven_revolute_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    joints = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])

    user_frame = np.eye(4)
    user_frame[:3, 3] = [0.4, -0.2, 0.1]
    solver.set_user_frame(user_frame)
    np.testing.assert_allclose(solver.user_frame, user_frame)
    base_target = solver.forward(joints)
    user_target = solver.forward_user(joints)
    np.testing.assert_allclose(user_target, user_frame @ base_target)

    solution = solver.solve_user(user_target, joints)[0]
    np.testing.assert_allclose(solver.forward(solution), base_target, atol=1e-3)
    detailed = solver.solve_detailed_user(user_target, joints)
    assert detailed.status == hm.SRSSolveStatus.SUCCESS

    solver.clear_user_frame()
    np.testing.assert_allclose(solver.user_frame, np.eye(4))
    np.testing.assert_allclose(solver.forward_user(joints), base_target)

    tool = np.eye(4)
    tool[2, 3] = 0.05
    solver.set_tcp(tool)
    np.testing.assert_allclose(solver.tcp, tool)
    assert not np.allclose(solver.forward(joints), base_target)
    solver.clear_tcp()
    np.testing.assert_allclose(solver.tcp, np.eye(4))

    malformed = np.eye(4)
    malformed[3, 3] = 2.0
    with pytest.raises(ValueError, match="rigid transform"):
        solver.set_user_frame(malformed)


def test_srs_closed_form_solution_from_urdf_parameters(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "ideal_srs.urdf"
    _write_ideal_srs_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    assert solver.analyze_geometry().closed_form_compatible
    joints = np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15])
    target = solver.forward(joints)
    configuration = solver.configuration(joints)
    solution = solver.analytic_solution(target, configuration, joints)
    np.testing.assert_allclose(solution, joints, atol=1e-9)
    np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10)

    all_configurations = solver.solve(
        target, joints, hm.SRSSolveMethod.ALL_CONFIGURATIONS
    )
    assert all_configurations
    for candidate in all_configurations:
        candidate_redundancy = solver.configuration(candidate).redundancy
        redundancy_delta = np.arctan2(
            np.sin(candidate_redundancy - configuration.redundancy),
            np.cos(candidate_redundancy - configuration.redundancy),
        )
        assert abs(redundancy_delta) < 1e-6


def test_srs_closed_form_candidate_refines_offset_chain(tmp_path):
    import holistic_motion as hm

    urdf = tmp_path / "offset_srs.urdf"
    _write_offset_srs_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    assert solver.analyze_geometry().closed_form_compatible
    joints = np.array(
        [
            0.51478342,
            -0.48319332,
            -0.47513737,
            -1.72500699,
            -2.74124467,
            0.75606588,
            -0.68688189,
        ]
    )
    target = solver.forward(joints)
    configuration = solver.configuration(joints)
    solution = solver.analytic_solution(target, configuration, joints)
    np.testing.assert_allclose(solver.forward(solution), target, atol=1e-8)
    solved = solver.configuration(solution)
    assert (solved.shoulder, solved.elbow, solved.wrist) == (
        configuration.shoulder,
        configuration.elbow,
        configuration.wrist,
    )
