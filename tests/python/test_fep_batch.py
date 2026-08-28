import numpy as np
import pytest


def _seven_axis_urdf():
    links = "".join(f'<link name="link{i}"/>' for i in range(8))
    axes = ("0 0 1", "0 1 0", "1 0 0", "0 1 0",
            "1 0 0", "0 1 0", "1 0 0")
    joints = "".join(
        f'''<joint name="joint{i}" type="revolute">
        <parent link="link{i}"/><child link="link{i + 1}"/>
        <origin xyz="0 0 {0.1 if i else 0.0}"/>
        <axis xyz="{axes[i]}"/>
        <limit lower="-2" upper="2" velocity="1" effort="1"/>
        </joint>'''
        for i in range(7)
    )
    return f'<robot name="fep_batch">{links}{joints}</robot>'


def test_fep_batch_fk_matches_scalar_and_validates_input(tmp_path):
    import holistic_motion as hm

    path = tmp_path / "seven_axis.urdf"
    path.write_text(_seven_axis_urdf(), encoding="utf-8")
    robot = hm.Robot(str(path))
    solver = robot.create_fep_kinematics("link0", "link7")
    assert isinstance(solver, hm.FEPKinematics)
    joints = np.random.default_rng(7).uniform(-1.5, 1.5, size=(32, 7))
    batch = solver.forward_batch(joints, hm.FEPBackend.CPU)
    expected = np.stack([solver.forward(row) for row in joints])
    np.testing.assert_allclose(batch, expected, atol=1e-12)
    assert solver.resolve_backend(hm.FEPBackend.CPU, 4096) == hm.FEPBackend.CPU
    with pytest.raises(ValueError):
        solver.forward_batch(joints[:, :6])
    with pytest.raises(ValueError):
        solver.forward_batch(np.full((1, 7), np.nan))
    with pytest.raises(ValueError):
        solver.forward_batch(np.full((1, 7), 3.0))
    if not solver.cuda_available:
        with pytest.raises(ValueError):
            solver.forward_batch(joints, hm.FEPBackend.CUDA)


def test_fep_ik_round_trip_is_strict_and_rejects_invalid_targets(tmp_path):
    import holistic_motion as hm

    path = tmp_path / "seven_axis.urdf"
    path.write_text(_seven_axis_urdf(), encoding="utf-8")
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    expected = np.array([0.2, -0.35, 0.25, -0.6, 0.3, 0.4, -0.2])
    target = solver.forward(expected)
    seed = expected + np.array([0.03, -0.02, 0.01, 0.02, -0.01, 0.02, 0.0])
    solutions = solver.solve(target, seed, hm.FEPSolveMethod.SEEDED_NUMERICAL)
    assert solutions
    actual = solver.forward(solutions[0])
    np.testing.assert_allclose(actual[:3, 3], target[:3, 3], atol=1e-5)
    np.testing.assert_allclose(actual[:3, :3], target[:3, :3], atol=1e-5)

    invalid = target.copy()
    invalid[0, 3] = np.nan
    with pytest.raises(ValueError):
        solver.solve(invalid, seed)


def test_fep_continuous_tracker_preserves_branch_and_precision(tmp_path):
    import holistic_motion as hm
    from holistic_motion.kinematics import FEPContinuousTracker

    path = tmp_path / "seven_axis.urdf"
    path.write_text(_seven_axis_urdf(), encoding="utf-8")
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    initial = np.array([0.2, -0.35, 0.25, -0.6, 0.3, 0.4, -0.2])
    tracker = FEPContinuousTracker(solver, initial)
    branch = tracker.configuration
    previous = initial
    for offset in np.linspace(0.002, 0.02, 6):
        expected = initial.copy()
        expected[0] += offset
        result = tracker.solve(solver.forward(expected), dt=0.02)
        assert result.configuration == branch
        assert not result.branch_changed
        assert result.position_error < 1e-5
        assert result.angle_error < 1e-5
        assert np.max(np.abs(result.joints - previous)) < 0.05
        previous = result.joints
