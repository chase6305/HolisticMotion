import numpy as np
import pytest


def _seven_axis_urdf():
    links = "".join(f'<link name="link{i}"/>' for i in range(8))
    axes = ("0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0")
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


@pytest.mark.parametrize("backend_name", ["CPU", "AUTO", "CUDA"])
def test_oblique_large_angles_keep_scalar_and_batch_phase(tmp_path, backend_name):
    import holistic_motion as hm

    path = tmp_path / "large_oblique_turn.urdf"
    path.write_text(
        _seven_axis_urdf()
        .replace('xyz="0 0 1"', 'xyz="1 1 1"', 1)
        .replace('lower="-2" upper="2"', 'lower="-1e308" upper="1e308"')
    )
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    if backend_name == "CUDA" and not solver.cuda_available:
        pytest.skip("native CUDA backend is unavailable")
    states = np.zeros((129, 7))
    states[:, 0] = np.resize([1e20, -1e20, 1e155, -1e155, 1e200, 1e308, -1e308], 129)
    axis = np.ones(3) / np.sqrt(3.0)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    expected = np.tile(np.eye(4), (states.shape[0], 1, 1))
    expected[:, :3, :3] += np.sin(states[:, 0, None, None]) * skew + (
        1 - np.cos(states[:, 0, None, None])
    ) * (skew @ skew)
    expected[:, :3, 3] = expected[:, :3, :3] @ np.array([0.0, 0.0, 0.6])
    scalar = np.stack([solver.forward(row) for row in states])
    np.testing.assert_allclose(scalar, expected, rtol=0, atol=3e-15)
    batch = solver.forward_batch(states, getattr(hm.FEPBackend, backend_name))
    np.testing.assert_allclose(batch, expected, rtol=0, atol=3e-15)


@pytest.mark.parametrize("backend_name", ["CPU", "AUTO", "CUDA"])
@pytest.mark.parametrize(
    "axis",
    [
        "-0.36318104719500244 1.0634338443935962 -1.634851455255571",
        "1.571415911246069 -1.2244654067099312 -0.2982382350002157",
        "-1.1244560386652949 0.9442287122141714 1.0885412608548457",
    ],
)
def test_large_oblique_phase_uses_same_axis_scale_on_every_backend(
    tmp_path, backend_name, axis
):
    import holistic_motion as hm

    path = tmp_path / "host_axis_scale.urdf"
    path.write_text(
        _seven_axis_urdf()
        .replace('xyz="0 0 1"', f'xyz="{axis}"', 1)
        .replace('lower="-2" upper="2"', 'lower="-1e308" upper="1e308"')
    )
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    if backend_name == "CUDA" and not solver.cuda_available:
        pytest.skip("native CUDA backend is unavailable")
    states = np.zeros((129, 7))
    states[:, 0] = np.resize([1e20, -1e20, 1e200, -1e200, 1e308, -1e308], 129)
    expected = solver.forward_batch(states, hm.FEPBackend.CPU)
    scalar = np.stack([solver.forward(row) for row in states])
    np.testing.assert_allclose(scalar, expected, rtol=0, atol=3e-15)
    actual = solver.forward_batch(states, getattr(hm.FEPBackend, backend_name))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-15)


@pytest.mark.parametrize("backend_name", ["CPU", "AUTO", "CUDA"])
def test_fep_batch_rejects_nonfinite_output_after_finite_input(tmp_path, backend_name):
    import holistic_motion as hm

    path = tmp_path / "overflow.urdf"
    path.write_text(_seven_axis_urdf(), encoding="utf-8")
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    if backend_name == "CUDA" and not solver.cuda_available:
        pytest.skip("native CUDA backend is unavailable")
    tool = np.eye(4)
    tool[:2, 3] = 1.5e308
    assert solver.set_tcp(tool)
    states = np.zeros((2, 7))
    states[1, 0] = np.pi / 4.0
    assert np.isfinite(solver.forward(states[0])).all()
    with pytest.raises(ValueError):
        solver.forward(states[1])
    with pytest.raises(ValueError):
        solver.forward_batch(states, getattr(hm.FEPBackend, backend_name))


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("batch_size", [0, 1, 32, 129])
@pytest.mark.parametrize("backend_name", ["CPU", "AUTO", "CUDA"])
@pytest.mark.parametrize("axis_scale", [1.0, 1.5])
def test_fep_batch_layouts_and_tcp_updates(
    tmp_path, layout, batch_size, backend_name, axis_scale
):
    import holistic_motion as hm

    path = tmp_path / "oblique_axis.urdf"
    path.write_text(
        _seven_axis_urdf().replace(
            'xyz="0 1 0"', f'xyz="{0.6 * axis_scale} 0 {0.8 * axis_scale}"'
        ),
        encoding="utf-8",
    )
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    if backend_name == "CUDA" and not solver.cuda_available:
        pytest.skip("native CUDA backend is unavailable")
    states = np.random.default_rng(17).uniform(-1.5, 1.5, size=(batch_size, 7))
    if layout == "strided":
        storage = np.empty((batch_size, 14))
        storage[:, ::2] = states
        states = storage[:, ::2]
    else:
        states = np.array(states, order=layout)
    tool = np.array(
        [
            [0.0, -1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    for tcp in (np.eye(4), tool, np.eye(4)):
        assert solver.set_tcp(tcp)
        actual = solver.forward_batch(states, getattr(hm.FEPBackend, backend_name))
        assert actual.shape == (batch_size, 4, 4)
        if batch_size:
            expected = np.stack([solver.forward(state) for state in states])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


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


@pytest.mark.parametrize(
    "method_name",
    ["SEEDED_NUMERICAL", "CONFIGURATION", "ALL_CONFIGURATIONS", "NEAREST_REDUNDANCY"],
)
def test_fep_ik_refinement_preserves_tcp_updates(tmp_path, method_name):
    import holistic_motion as hm

    path = tmp_path / "seven_axis.urdf"
    path.write_text(_seven_axis_urdf(), encoding="utf-8")
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    expected = np.array([0.2, -0.35, 0.25, -0.6, 0.3, 0.4, -0.2])
    seed = expected + [0.03, -0.02, 0.01, 0.02, -0.01, 0.02, 0.0]
    tool = np.array(
        [
            [0.0, -1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    for tcp in (tool, np.eye(4), tool):
        assert solver.set_tcp(tcp)
        target = solver.forward(expected)
        # The regular solver leaves a residual that requires FEP's strict pass.
        coarse = solver.forward(solver.inverse(target, seed))
        assert np.linalg.norm(coarse[:3, 3] - target[:3, 3]) > 1e-5
        solutions = solver.solve(target, seed, getattr(hm.FEPSolveMethod, method_name))
        assert solutions
        for solution in solutions:
            actual = solver.forward(solution)
            assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 1e-5
            relative = actual[:3, :3].T @ target[:3, :3]
            angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            assert angle < 1e-5


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
