"""Metamorphic checks on public APIs, using bounded physical input domains."""

from pathlib import Path
from tempfile import TemporaryDirectory

import holistic_motion as hm
import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory

pytest.importorskip("hypothesis", reason="install the test extra for property tests")
from hypothesis import example, given
from hypothesis import strategies as st

pytestmark = pytest.mark.property


@st.composite
def optimization_paths(draw):
    dof = draw(st.integers(1, 5))
    count = draw(st.integers(3, 8))
    points = (
        np.array(
            draw(
                st.lists(
                    st.lists(st.integers(-20, 20), min_size=dof, max_size=dof),
                    min_size=count,
                    max_size=count,
                )
            ),
            dtype=float,
        )
        / 8.0
    )
    weights = 2.0 ** np.array(
        draw(st.lists(st.integers(-3, 3), min_size=dof, max_size=dof))
    )
    continuous = draw(st.lists(st.booleans(), min_size=dof, max_size=dof))
    return points, weights, continuous


@given(
    case=optimization_paths(),
    length_weight=st.integers(0, 3),
    smoothness_weight=st.integers(1, 4),
    use_state_cost=st.booleans(),
)
def test_path_optimizer_preserves_endpoints_and_decreases_full_objective(
    case, length_weight, smoothness_weight, use_state_cost
):
    points, weights, continuous = case
    optimizer = hm.PathOptimizer(
        np.full(points.shape[1], -np.pi), np.full(points.shape[1], np.pi)
    )
    optimizer.set_joint_weights(weights)
    optimizer.set_continuous_joints(np.flatnonzero(continuous).tolist())
    options = hm.PathOptimizationOptions()
    options.max_iterations = 8
    options.timeout_seconds = 5.0
    options.step_size = 0.15
    options.length_weight = length_weight
    options.smoothness_weight = smoothness_weight
    if use_state_cost:
        options.state_cost_weight = 0.25
        optimizer.set_state_cost(lambda q: np.dot(q, q))
        optimizer.set_state_cost_gradient(lambda q: 2.0 * np.asarray(q))

    def objective(path):
        differences = np.diff(path, axis=0)
        differences[:, continuous] = (differences[:, continuous] + np.pi) % (
            2 * np.pi
        ) - np.pi
        length = np.sqrt(np.sum(weights * differences**2, axis=1)).sum()
        smoothness = np.sum(weights * np.diff(differences, axis=0) ** 2)
        state_cost = options.state_cost_weight * np.sum(path[1:-1] ** 2)
        return length_weight * length + smoothness_weight * smoothness + state_cost

    result = optimizer.optimize(points, options)
    assert result.success
    assert result.status != hm.PathOptimizationStatus.TIMEOUT
    output = np.asarray(result.path)
    np.testing.assert_array_equal(output[[0, -1]], points[[0, -1]])
    assert output.shape == points.shape
    assert np.isfinite(output).all()
    assert np.all(np.abs(output) <= np.pi)
    initial, final = objective(points), objective(output)
    assert final <= initial + 2e-10
    assert result.statistics.initial_objective == pytest.approx(initial, abs=2e-10)
    assert result.statistics.final_objective == pytest.approx(final, abs=2e-10)


@st.composite
def joint_spaces(draw):
    dof = draw(st.integers(1, 6))
    start = (
        np.array(draw(st.lists(st.integers(-40, 40), min_size=dof, max_size=dof)))
        / 16.0
    )
    goal = (
        np.array(draw(st.lists(st.integers(-40, 40), min_size=dof, max_size=dof)))
        / 16.0
    )
    weights = 2.0 ** np.array(
        draw(st.lists(st.integers(-3, 3), min_size=dof, max_size=dof))
    )
    continuous = draw(st.lists(st.booleans(), min_size=dof, max_size=dof))
    turns = np.array(draw(st.lists(st.integers(-4, 4), min_size=dof, max_size=dof)))
    shift = 2.0 * np.pi * turns * continuous
    return start, goal, weights, continuous, shift


@given(joint_spaces())
def test_periodic_metric_is_invariant_under_whole_turns(case):
    start, goal, weights, continuous, shift = case
    options = hm.PlanningOptions()
    options.timeout_seconds = 5.0
    options.interpolate_path = True
    options.interpolation_points = 17
    results = []
    for offset in (np.zeros_like(start), shift):
        planner = hm.SamplingPlanner(offset - np.pi, offset + np.pi)
        planner.set_joint_weights(weights)
        planner.set_continuous_joints(np.flatnonzero(continuous).tolist())
        result = planner.plan(start + offset, goal + offset, options)
        assert result.success, result.message
        results.append(result)
    delta = goal - start
    delta[continuous] = (delta[continuous] + np.pi) % (2.0 * np.pi) - np.pi
    expected = np.sqrt(np.dot(weights, delta**2))
    for result in results:
        assert result.statistics.final_path_length == pytest.approx(expected, abs=1e-12)
    difference = np.asarray(results[1].path) - shift - np.asarray(results[0].path)
    difference[:, continuous] = (difference[:, continuous] + np.pi) % (
        2.0 * np.pi
    ) - np.pi
    np.testing.assert_allclose(difference, 0.0, atol=1e-12)


@given(
    joints=st.lists(st.floats(-0.5, 0.5, width=32), min_size=4, max_size=4),
    translation=st.lists(st.floats(-2.0, 2.0, width=32), min_size=3, max_size=3),
    angle=st.floats(-1.0, 1.0, width=32),
    axis_components=st.tuples(
        st.integers(1, 4), st.integers(-4, 4), st.integers(-4, 4)
    ),
)
def test_mixed_joint_fk_ik_and_jacobian_transform_with_the_base(
    joints, translation, angle, axis_components
):
    axis_vector = np.asarray(axis_components, dtype=float)
    axis_vector /= np.linalg.norm(axis_vector)
    links = '<link name="world"/>' + "".join(
        f'<link name="link{i}"/>' for i in range(5)
    )
    mount = (
        '<joint name="mount" type="fixed"><parent link="world"/><child link="link0"/>'
        f'<origin xyz="{" ".join(map(str, translation))}" rpy="0 0 {angle}"/></joint>'
    )
    chain = "".join(
        f'<joint name="joint{i}" type="{kind}"><parent link="link{i}"/>'
        f'<child link="link{i + 1}"/><axis xyz="{axis}"/>'
        '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        for i, (kind, axis) in enumerate(
            [
                ("revolute", " ".join(map(str, axis_vector))),
                ("prismatic", "1 0 0"),
                ("prismatic", "0 1 0"),
                ("prismatic", "0 0 1"),
            ]
        )
    )
    with TemporaryDirectory() as directory:
        urdf = Path(directory) / "mixed.urdf"
        urdf.write_text(
            f'<robot name="mixed">{links}{mount}{chain}</robot>', encoding="utf-8"
        )
        robot = hm.Robot(str(urdf))
    local = robot.create_kinematics("link0", "link4")
    world = robot.create_kinematics("world", "link4")
    base = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    base[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    base[:3, 3] = translation
    target = world.forward(joints)
    x, y, z = axis_vector
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = (
        np.eye(3) + np.sin(joints[0]) * skew + (1.0 - np.cos(joints[0])) * (skew @ skew)
    )
    reference = np.eye(4)
    reference[:3, :3] = rotation
    reference[:3, 3] = rotation @ joints[1:]
    np.testing.assert_allclose(target, base @ reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(target, base @ local.forward(joints), atol=1e-12)
    jacobian = world.jacobian(joints)
    local_jacobian = local.jacobian(joints)
    for block in (slice(0, 3), slice(3, 6)):
        np.testing.assert_allclose(
            jacobian[block], base[:3, :3] @ local_jacobian[block], atol=1e-12
        )
    solution = world.inverse(target, np.asarray(joints) + [0.1, -0.1, 0.1, -0.1])
    actual = world.forward(solution)
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= 5e-4
    np.testing.assert_allclose(actual[:3, :3], target[:3, :3], atol=1e-3)


@st.composite
def waypoint_paths(draw):
    count = draw(st.integers(2, 5))
    dof = draw(st.integers(1, 4))
    points = (
        np.array(
            draw(
                st.lists(
                    st.lists(st.integers(-8, 8), min_size=dof, max_size=dof),
                    min_size=count,
                    max_size=count,
                )
            ),
            dtype=float,
        )
        / 8.0
    )
    # A strictly increasing first coordinate ensures distinct waypoints without
    # filtering away most generated examples.
    points[:, 0] = (
        np.cumsum(draw(st.lists(st.integers(1, 8), min_size=count, max_size=count)))
        / 8.0
    )
    velocity = (
        np.array(draw(st.lists(st.integers(1, 8), min_size=dof, max_size=dof))) / 4.0
    )
    acceleration = (
        np.array(draw(st.lists(st.integers(1, 8), min_size=dof, max_size=dof))) / 2.0
    )
    return points, velocity, acceleration


@given(
    case=waypoint_paths(),
    spatial_power=st.integers(-2, 2),
    time_power=st.integers(-2, 2),
)
def test_toppra_preserves_units_and_time_scaling(case, spatial_power, time_power):
    points, velocity, acceleration = case
    spatial_scale, time_scale = 2.0**spatial_power, 2.0**time_power
    original = ToppraTrajectory(points, velocity, acceleration, grid_size=20)
    transformed = ToppraTrajectory(
        spatial_scale * points + 2.0,
        spatial_scale * time_scale * velocity,
        spatial_scale * time_scale**2 * acceleration,
        grid_size=20,
    )
    assert transformed.duration == pytest.approx(
        original.duration / time_scale, rel=1e-8
    )
    times = np.linspace(0.0, original.duration, 101)
    q, dq, ddq = original.sample(times)
    mapped_q, mapped_dq, mapped_ddq = transformed.sample(times / time_scale)
    np.testing.assert_allclose(mapped_q, spatial_scale * q + 2.0, atol=1e-8)
    np.testing.assert_allclose(mapped_dq, spatial_scale * time_scale * dq, atol=1e-8)
    np.testing.assert_allclose(
        mapped_ddq, spatial_scale * time_scale**2 * ddq, atol=1e-7
    )
    np.testing.assert_allclose(q[[0, -1]], points[[0, -1]], atol=1e-10)
    np.testing.assert_allclose(dq[[0, -1]], 0.0, atol=1e-9)
    assert np.all(np.abs(dq) <= velocity + 1e-8)
    assert np.all(np.abs(ddq) <= acceleration + 1e-8)


@pytest.mark.parametrize(
    "algorithm", [hm.SamplingAlgorithm.RRT_STAR, hm.SamplingAlgorithm.INFORMED_RRT_STAR]
)
@example(seed=0, budget=155, extra=5)
@given(seed=st.integers(0, 50), budget=st.integers(100, 600), extra=st.integers(1, 100))
def test_rrt_star_more_iterations_do_not_worsen_the_incumbent(
    algorithm, seed, budget, extra
):
    planner = hm.SamplingPlanner(
        [-1.0, -1.0],
        [1.0, 1.0],
        lambda q: not (-0.2 < q[0] < 0.2 and -0.75 < q[1] < 0.75),
    )
    options = hm.PlanningOptions()
    options.algorithm = algorithm
    options.timeout_seconds = 10.0
    options.extension_range = 0.15
    options.edge_resolution = 0.02
    options.goal_bias = 0.0
    options.simplify_path = False
    options.random_seed = seed
    options.max_iterations = budget
    earlier = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
    options.max_iterations += extra
    later = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
    assert earlier.statistics.iterations == budget
    assert later.statistics.iterations == budget + extra
    if earlier.success:
        assert later.success
        assert (
            later.statistics.final_path_length
            <= earlier.statistics.final_path_length + 1e-10
        )
