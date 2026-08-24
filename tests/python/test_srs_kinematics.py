import numpy as np


def _write_seven_revolute_urdf(path):
    axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0",
            "1 0 0", "0 1 0", "1 0 0"]
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
    joints.append('''<joint name="tool" type="fixed">
      <parent link="link7"/><child link="link8"/>
      <origin xyz="-0.066 0 0" rpy="0 -1.5707963268 0"/>
    </joint>''')
    path.write_text(
        f'<robot name="ideal_srs">{links}{"".join(joints)}</robot>',
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

    target = robot.kinematics.forward(joints)
    planner = hm.NullSpacePlanner(robot.kinematics)
    path = planner.plan(joints, direction, 2, 0.01)
    assert len(path) == 3
    for point in path:
        np.testing.assert_allclose(robot.kinematics.forward(point), target,
                                   atol=1e-3)


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
        np.testing.assert_allclose(solver.forward(solutions[0]), target,
                                   atol=1e-3)

    configured = solver.solve_configuration(target, config, joints)
    np.testing.assert_allclose(solver.forward(configured), target, atol=1e-3)


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
