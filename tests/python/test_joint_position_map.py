"""Configuration/tangent mapping across scalar chains and manifold models."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from holistic_motion.configuration import JointCoordinate, JointPositionMap
from optional_dependencies import python_pinocchio_or_skip


def test_mapping_preserves_unselected_manifold_blocks_and_uses_tangent_indices():
    # A free flyer occupies q[0:7]/v[0:6], followed by a continuous turn
    # and a slider. Select them in reverse model order.
    mapping = JointPositionMap(
        10,
        8,
        [
            JointCoordinate("slide", 9, 1, 7, 1),
            JointCoordinate("turn", 7, 2, 6, 1),
        ],
    )
    reference = np.array([1, 2, 3, 0, 0, 0, 1, 1, 0, 0], dtype=float)
    original = reference.copy()
    q = mapping.configuration_with_joint_positions(reference, [0.3, 0.7])
    np.testing.assert_array_equal(reference, original)
    np.testing.assert_array_equal(q[:7], reference[:7])
    np.testing.assert_allclose(q[7:], [np.cos(0.7), np.sin(0.7), 0.3])
    np.testing.assert_allclose(mapping.joint_positions(q), [0.3, 0.7])
    np.testing.assert_array_equal(mapping.joint_velocities(np.arange(8)), [7, 6])
    with pytest.raises(ValueError, match="size 8"):
        mapping.joint_velocities(q)
    with pytest.raises(FrozenInstanceError):
        mapping.nq = 8


@pytest.mark.parametrize(
    "joints",
    [
        [JointCoordinate("turn", 0, 2, 0, 1)] * 2,
        [JointCoordinate("turn", 0, 2, 0, 1), JointCoordinate("slide", 1, 1, 1, 1)],
        [JointCoordinate("turn", 0, 2, 0, 1), JointCoordinate("slide", 2, 1, 0, 1)],
        [JointCoordinate("floating", 0, 7, 0, 6)],
        [JointCoordinate("outside", 3, 1, 0, 1)],
    ],
)
def test_mapping_rejects_ambiguous_or_unsupported_blocks(joints):
    with pytest.raises(ValueError):
        JointPositionMap(3, 2, joints)


def test_empty_mapping_and_invalid_continuous_pair():
    empty = JointPositionMap(0, 0, [])
    assert empty.joint_positions([]).shape == (0,)
    assert empty.joint_velocities([]).shape == (0,)
    assert empty.configuration_with_joint_positions([], []).shape == (0,)
    mapping = JointPositionMap(2, 1, [JointCoordinate("turn", 0, 2, 0, 1)])
    for value in ([0, 0], [2, 0], [1, np.nan]):
        with pytest.raises(ValueError):
            mapping.joint_positions(value)
    for value in ([0.1 + 0j], np.array([0.1 + 1j], dtype=object)):
        with pytest.raises(TypeError, match="real"):
            mapping.configuration_with_joint_positions([1, 0], value)


def test_numpy_integer_metadata_is_canonical_and_cannot_underflow():
    with np.errstate(all="raise"), pytest.raises(ValueError, match="outside"):
        JointPositionMap(
            np.uint64(0),
            np.uint64(0),
            [
                JointCoordinate(
                    "outside", np.uint64(0), np.uint64(1), np.uint64(0), np.uint64(1)
                ),
            ],
        )
    mapping = JointPositionMap(
        np.uint64(3),
        np.int64(2),
        [
            JointCoordinate(
                "turn", np.uint64(0), np.uint64(2), np.int64(0), np.int64(1)
            ),
            JointCoordinate(
                "slide", np.uint64(2), np.uint64(1), np.int64(1), np.int64(1)
            ),
        ],
    )
    assert type(mapping.nq) is type(mapping.nv) is int
    assert all(
        type(j.idx_q) is type(j.nq) is type(j.idx_v) is type(j.nv) is int
        for j in mapping.joints
    )
    assert mapping == JointPositionMap(3, 2, mapping.joints)
    assert hash(mapping) == hash(JointPositionMap(3, 2, mapping.joints))
    for indices in (
        mapping._q_indices,
        mapping._v_indices,
        mapping._pair_positions,
        mapping._pair_q_indices,
    ):
        assert not indices.flags.writeable
    with pytest.raises(ValueError, match="index range"):
        JointPositionMap(int(np.iinfo(np.intp).max) + 1, 0, [])


def test_model_adapter_rejects_fractional_coordinate_metadata():
    model = SimpleNamespace(
        nq=1,
        nv=1,
        names=["universe", "slide"],
        idx_qs=[0, 0.5],
        nqs=[0, 1],
        idx_vs=[0, 0],
        nvs=[0, 1],
    )
    with pytest.raises(TypeError, match="integers"):
        JointPositionMap.from_model(model, ["slide"])


def test_model_adapter_rejects_ambiguous_names_outside_the_selected_chain():
    model = SimpleNamespace(
        nq=3,
        nv=3,
        joint_coordinates=[
            JointCoordinate("duplicate", 0, 1, 0, 1),
            JointCoordinate("duplicate", 1, 1, 1, 1),
            JointCoordinate("selected", 2, 1, 2, 1),
        ],
    )
    with pytest.raises(ValueError, match="duplicate model joint name"):
        JointPositionMap.from_model(model, ["selected"])


@pytest.mark.parametrize("count", [1, 2, 7])
def test_mapping_preserves_subnormal_phases_with_strict_numpy_errors(count):
    mapping = JointPositionMap(
        2 * count,
        count,
        [JointCoordinate(str(i), 2 * i, 2, i, 1) for i in range(count)],
    )
    reference = np.tile([1.0, 0.0], count)
    positions = np.full(count, 1e-320)
    positions[::2] *= -1.0
    with np.errstate(all="raise"):
        full = mapping.configuration_with_joint_positions(reference, positions)
        np.testing.assert_array_equal(mapping.joint_positions(full), positions)
        invalid = full.copy()
        invalid[:2] = np.finfo(float).max
        with pytest.raises(ValueError, match="unit pair"):
            mapping.joint_positions(invalid)


@pytest.mark.parametrize("count,selected_count", [(7, 1), (7, 7), (33, 7), (33, 33)])
def test_prepared_mapping_preserves_context_and_pulls_back_tangent_gradients(
    count, selected_count
):
    generator = np.random.default_rng(120 + count + selected_count)
    coordinates = []
    nq, nv = 7, 6  # An unselected floating base precedes the scalar joints.
    for index in range(count):
        size = 2 if index % 3 == 0 else 1
        coordinates.append(JointCoordinate(f"joint{index}", nq, size, nv, 1))
        nq += size
        nv += 1
    reference = np.zeros(nq)
    reference[6] = 1.0
    for joint in coordinates:
        reference[joint.idx_q] = 1.0 if joint.nq == 2 else 0.4
    selected = generator.permutation(count)[:selected_count]
    mapping = JointPositionMap(nq, nv, [coordinates[i] for i in selected])
    positions = generator.uniform(-2, 2, selected_count)
    reference.setflags(write=False)
    full = mapping.configuration_with_joint_positions(reference, positions)
    np.testing.assert_allclose(mapping.joint_positions(full), positions, atol=1e-14)
    np.testing.assert_array_equal(full[:7], reference[:7])
    weights = generator.normal(size=nq)
    gradient = np.zeros(nv)
    for joint in coordinates:
        if joint.nq == 1:
            gradient[joint.idx_v] = weights[joint.idx_q]
        else:
            c, s = full[joint.idx_q : joint.idx_q + 2]
            gradient[joint.idx_v] = (
                -weights[joint.idx_q] * s + weights[joint.idx_q + 1] * c
            )
    selected_gradient = mapping.joint_velocities(gradient)
    expected = []
    for index in range(selected_count):
        delta = np.eye(selected_count)[index] * 1e-6
        positive = mapping.configuration_with_joint_positions(
            reference, positions + delta
        )
        negative = mapping.configuration_with_joint_positions(
            reference, positions - delta
        )
        expected.append(weights @ (positive - negative) / 2e-6)
    np.testing.assert_allclose(selected_gradient, expected, rtol=1e-6, atol=1e-8)
    assert not np.shares_memory(full, reference)
    assert not np.shares_memory(selected_gradient, gradient)
    selected_gradient[:] = 9.0
    np.testing.assert_allclose(
        mapping.joint_velocities(gradient), expected, rtol=1e-6, atol=1e-8
    )


URDF = """<robot name="coordinate_mapping">
  <link name="base"/>
  <link name="arm"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
  <link name="tool"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
  <link name="other"/>
  <joint name="turn" type="continuous"><parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/><limit velocity="2" effort="1"/></joint>
  <joint name="slide" type="prismatic"><parent link="arm"/><child link="tool"/>
    <axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
  <joint name="other_slide" type="prismatic"><parent link="base"/><child link="other"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
</robot>"""


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / "mapping.urdf"
    path.write_text(URDF)
    return path


@pytest.mark.parametrize("backend", ["mesh", "sphere"])
def test_native_and_python_mapping_agree_for_continuous_joint(urdf, backend):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    model = (
        hm.CollisionModel(str(urdf))
        if backend == "mesh"
        else hm.SphereCollisionModel(
            str(urdf),
            [
                hm.CollisionSphere("arm", "arm", [0, 0, 0], 0.1),
                hm.CollisionSphere("tool", "tool", [0, 0, 0], 0.1),
            ],
        )
    )
    assert model.nq == 4 and model.nv == 3
    names = ["slide", "turn"]
    native = model.create_joint_position_map(names)
    python = JointPositionMap.from_model(model, names)
    reference = model.configuration_from_joint_positions({"other_slide": 0.6})
    for mapping in (native, python):
        q = mapping.configuration_with_joint_positions(reference, [0.4, 1.2])
        np.testing.assert_allclose(mapping.joint_positions(q), [0.4, 1.2])
        np.testing.assert_allclose(
            q, model.configuration_with_joint_positions(reference, names, [0.4, 1.2])
        )
        np.testing.assert_allclose(
            q,
            model.configuration_from_joint_positions(
                {"other_slide": 0.6, "slide": 0.4, "turn": 1.2}
            ),
        )
        np.testing.assert_array_equal(
            mapping.joint_velocities(np.arange(model.nv)),
            [np.arange(model.nv)[j.idx_v] for j in native.joints],
        )
    np.testing.assert_allclose(model.joint_lower_limits(names), [-2, -np.pi])
    np.testing.assert_allclose(model.joint_upper_limits(names), [2, np.pi])
    for factory in (
        model.create_joint_position_map,
        lambda names: JointPositionMap.from_model(model, names),
    ):
        with pytest.raises(ValueError):
            factory(["turn", "turn"])
        with pytest.raises(ValueError):
            factory(["missing"])


def test_robot_chain_names_embed_into_full_scalar_model(urdf):
    import holistic_motion as hm

    if not hasattr(hm, "Robot"):
        pytest.skip("native extension is disabled")
    robot = hm.Robot(str(urdf))
    assert robot.nq == robot.nv == robot.model_dof == 3
    assert robot.chain_joint_names("base", "tool") == ["turn", "slide"]
    assert robot.chain_joint_names("tool", "base") == []
    urdf.unlink()  # Names and mapping use the parsed tree.
    native = robot.create_joint_position_map(robot.chain_joint_names("base", "tool"))
    python = JointPositionMap.from_model(robot, ["turn", "slide"])
    for mapping in (native, python):
        reference = np.full(robot.nq, 0.7)
        full = mapping.configuration_with_joint_positions(reference, [0.2, 0.3])
        np.testing.assert_allclose(mapping.joint_positions(full), [0.2, 0.3])
        other = next(
            j.idx_q for j in robot.joint_coordinates if j.name == "other_slide"
        )
        assert full[other] == 0.7


def test_named_collision_planner_uses_continuous_angles(urdf):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    model = hm.CollisionModel(str(urdf))
    context = model.configuration_from_joint_positions({"other_slide": 0.6})
    planner = hm.SamplingPlanner.from_collision_joints(
        model, ["turn", "slide"], context
    )
    planner.set_joint_weights([1.0, 1.0])
    options = hm.PlanningOptions()
    options.random_seed = 7
    result = planner.plan([3.0, 0.4], [-3.0, 0.4], options)
    assert result.success
    assert result.statistics.final_path_length == pytest.approx(2 * np.pi - 6)
    optimizer = hm.PathOptimizer.from_collision_joints(
        model, ["turn", "slide"], context
    )
    optimizer.set_joint_weights([1.0, 1.0])
    optimized = optimizer.optimize(result.path)
    assert optimized.statistics.final_path_length < 0.3
    np.testing.assert_array_equal(
        context, model.configuration_from_joint_positions({"other_slide": 0.6})
    )


def test_named_sphere_adapter_maps_tangent_gradients_and_preserves_context(urdf):
    import holistic_motion as hm

    if not hasattr(hm, "SphereCollisionModel"):
        pytest.skip("native collision support is disabled")
    model = hm.SphereCollisionModel(
        str(urdf),
        [
            hm.CollisionSphere("arm", "arm", [1, 0, 0], 0.1),
            hm.CollisionSphere("obstacle", "base", [0, 1.5, 0], 0.1),
        ],
    )
    context = model.configuration_from_joint_positions({"other_slide": 0.6})
    names = ["slide", "turn"]
    mapping = model.create_joint_position_map(names)
    positions = np.array([0.4, 0.3])
    full = mapping.configuration_with_joint_positions(context, positions)
    analytic = mapping.joint_velocities(model.minimum_distance_gradient(full))
    difference = []
    for index in range(2):
        delta = np.eye(2)[index] * 1e-6
        positive = mapping.configuration_with_joint_positions(
            context, positions + delta
        )
        negative = mapping.configuration_with_joint_positions(
            context, positions - delta
        )
        difference.append(
            (
                model.minimum_distance(positive).distance
                - model.minimum_distance(negative).distance
            )
            / 2e-6
        )
    np.testing.assert_allclose(analytic, difference, rtol=1e-6, atol=1e-8)
    planner = hm.SamplingPlanner.from_sphere_collision_joints(model, names, context)
    result = planner.plan([0.4, 0.0], [0.4, 0.8])
    assert result.success
    optimizer = hm.PathOptimizer.from_sphere_collision_joints(
        model, names, context, clearance=2.0
    )
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 2.0
    options.max_iterations = 5
    optimized = optimizer.optimize([[0.4, 0.0], positions, [0.4, 0.8]], options)
    assert optimized.success
    assert optimized.statistics.state_cost_gradient_evaluations > 0
    assert (
        optimized.statistics.final_objective <= optimized.statistics.initial_objective
    )
    np.testing.assert_array_equal(
        context, model.configuration_from_joint_positions({"other_slide": 0.6})
    )
    for factory in (
        hm.SamplingPlanner.from_sphere_collision_model,
        hm.PathOptimizer.from_sphere_collision_model,
    ):
        with pytest.raises(ValueError, match="from_sphere_collision_joints"):
            factory([-1.0] * model.nq, [1.0] * model.nq, model)


@pytest.mark.parametrize("backend", ["mesh", "sphere"])
@pytest.mark.parametrize("initial_pairs", [False, True])
def test_named_clearance_optimizer_observes_pair_changes(urdf, backend, initial_pairs):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    if backend == "mesh":
        model = hm.CollisionModel(str(urdf), exclude_adjacent=False)
        factory = hm.PathOptimizer.from_collision_joints
    else:
        model = hm.SphereCollisionModel(
            str(urdf),
            [
                hm.CollisionSphere("arm", "arm", [0, 0, 0], 0.1),
                hm.CollisionSphere("tool", "tool", [0, 0, 0], 0.1),
            ],
        )
        factory = hm.PathOptimizer.from_sphere_collision_joints
    assert model.pair_count == 1
    context = model.neutral_configuration()
    if not initial_pairs:
        model.clear_collision_pairs()
        with pytest.raises(ValueError, match="positive clearance requires active"):
            factory(model, ["slide"], context, clearance=0.6)
        return

    optimizer = factory(model, ["slide"], context, clearance=0.6)
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 1.0
    options.max_iterations = 2
    path = [[0.4], [0.7], [1.0]]
    assert optimizer.optimize(path, options).success
    model.clear_collision_pairs()
    result = optimizer.optimize(path, options)
    options.state_cost_weight = 0.0
    ordinary = factory(model, ["slide"], context).optimize(path, options)
    assert result.success and ordinary.success
    assert result.statistics.final_objective == pytest.approx(
        ordinary.statistics.final_objective
    )
    assert model.reset_collision_pairs(False) == 1
    options.state_cost_weight = 1.0
    assert optimizer.optimize(path, options).success


@pytest.mark.parametrize("named", [False, True])
def test_sphere_clearance_query_failure_cannot_reuse_a_previous_distance(
    tmp_path, named
):
    import holistic_motion as hm

    if not hasattr(hm, "SphereCollisionModel"):
        pytest.skip("native collision support is disabled")
    path = tmp_path / "large_offset.urdf"
    path.write_text(
        '<robot name="large_offset"><link name="base"/><link name="tip"/>'
        '<joint name="turn" type="revolute"><parent link="base"/>'
        '<child link="tip"/><origin xyz="1e308 0 0"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" velocity="1" effort="1"/></joint></robot>'
    )
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("fixed", "base", [0, 0, 0], 0.1),
            hm.CollisionSphere("moving", "tip", [1e308, 0, 0], 0.1),
        ],
    )
    if named:
        optimizer = hm.PathOptimizer.from_sphere_collision_joints(
            model, ["turn"], [0.0], clearance=1.0
        )
    else:
        optimizer = hm.PathOptimizer.from_sphere_collision_model(
            [-3.0], [3.0], model, clearance=1.0
        )
    options = hm.PathOptimizationOptions()
    options.max_iterations = 1
    options.line_search_steps = 1
    options.step_size = 0.7
    options.edge_resolution = 10.0
    options.state_cost_weight = 1.0
    options.timeout_seconds = 10.0
    # The original path is valid. Its improving trial has finite coordinates,
    # but the physical sphere distance exceeds float64's range.
    with pytest.raises(OverflowError, match="sphere distance is not finite"):
        optimizer.optimize([[1.0], [1.4], [1.0]], options)
    invalid_start = 1.4 - options.step_size
    with pytest.raises(OverflowError, match="sphere distance is not finite"):
        model.minimum_distance([invalid_start])
    options.state_cost_weight = 0.0
    # A two-waypoint retry validates this exact start once. It must not accept
    # the preceding successful query's distance from the failed trial cache.
    with pytest.raises(OverflowError, match="sphere distance is not finite"):
        optimizer.optimize([[invalid_start], [1.0]], options)
    assert optimizer.optimize([[1.0], [1.4]], options).success


def test_full_configuration_planning_does_not_treat_unit_pairs_as_scalars(urdf):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    model = hm.CollisionModel(str(urdf))
    with pytest.raises(ValueError, match="from_collision_joints"):
        hm.SamplingPlanner.from_collision_model(
            [-1.0] * model.nq, [1.0] * model.nq, model
        )
    context = model.neutral_configuration()
    turn = next(j for j in model.joint_coordinates if j.name == "turn")
    with pytest.raises(ValueError, match="scalar joints"):
        hm.SamplingPlanner.from_collision_model(
            [-1.0], [1.0], model, [turn.idx_q], context
        )
    slide = next(j for j in model.joint_coordinates if j.name == "slide")
    planner = hm.SamplingPlanner.from_collision_model(
        [-1.0], [1.0], model, [slide.idx_q], context
    )
    assert planner.plan([0.4], [0.8]).success


@pytest.mark.parametrize("kind", ["mesh", "sphere"])
def test_native_collision_queries_reject_non_normalized_continuous_pairs(urdf, kind):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    model = (
        hm.CollisionModel(str(urdf))
        if kind == "mesh"
        else hm.SphereCollisionModel(
            str(urdf), [hm.CollisionSphere("arm", "arm", [1, 0, 0], 0.1)]
        )
    )
    full = model.neutral_configuration()
    turn = next(j for j in model.joint_coordinates if j.name == "turn")
    full[turn.idx_q : turn.idx_q + 2] = [2.0, 0.0]
    with pytest.raises(ValueError, match="manifold"):
        model.in_collision(full)
    valid = model.configuration_with_joint_positions(full, ["turn"], [0.3])
    assert not model.in_collision(valid)


@pytest.mark.parametrize("kind", ["mesh", "sphere"])
def test_named_chain_preserves_and_validates_floating_base_context(tmp_path, kind):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    urdf = tmp_path / "floating.urdf"
    urdf.write_text(
        URDF.replace(
            '<link name="base"/>',
            """
      <link name="world"/><link name="base"/>
      <joint name="floating_base" type="floating">
        <parent link="world"/><child link="base"/>
      </joint>""",
        )
        .replace('xyz="0 0 1"', 'xyz="0 0 1.5"')
        .replace('xyz="1 0 0"', 'xyz="2 0 0"')
    )
    if kind == "mesh":
        model = hm.CollisionModel(str(urdf))
        factories = (
            hm.SamplingPlanner.from_collision_joints,
            hm.PathOptimizer.from_collision_joints,
        )
    else:
        model = hm.SphereCollisionModel(
            str(urdf),
            [
                hm.CollisionSphere("tool", "tool", [0, 0, 0], 0.1),
            ],
        )
        factories = (
            hm.SamplingPlanner.from_sphere_collision_joints,
            hm.PathOptimizer.from_sphere_collision_joints,
        )
    assert model.nq == 11 and model.nv == 9
    floating = next(j for j in model.joint_coordinates if j.name == "floating_base")
    assert floating.nq == 7 and floating.nv == 6
    robot = hm.Robot(str(urdf))
    assert robot.nq == robot.nv == 3
    names = robot.chain_joint_names("base", "tool")
    assert names == ["turn", "slide"]
    context = model.neutral_configuration()
    context[floating.idx_q : floating.idx_q + 3] = [0.3, 0.4, 0.5]
    yaw = 0.4
    context[floating.idx_q + 5 : floating.idx_q + 7] = [
        np.sin(yaw / 2),
        np.cos(yaw / 2),
    ]
    original = context.copy()
    mapping = model.create_joint_position_map(names)
    full = mapping.configuration_with_joint_positions(context, [0.2, 0.3])
    np.testing.assert_array_equal(full[:7], context[:7])
    np.testing.assert_array_equal(context, original)
    assert model.validate_configuration(full) is None
    world_T_base = np.eye(4)
    world_T_base[:3, :3] = [
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ]
    world_T_base[:3, 3] = [0.3, 0.4, 0.5]
    expected = world_T_base @ robot.create_kinematics("base", "tool").forward(
        [0.2, 0.3]
    )
    if kind == "sphere":
        np.testing.assert_allclose(
            model.world_spheres(full)[0, :3], expected[:3, 3], atol=1e-12
        )
    else:
        model.add_box_obstacle("world_box", [0.05, 0.05, 0.05], expected)
        # Obstacle registration preserves the explicit collision-pair policy.
        assert (
            model.set_collision_groups(
                {"robot": ["tool"], "environment": ["world_box"]},
                [("robot", "environment")],
            )
            == 1
        )
        assert model.in_collision(full)
        away = full.copy()
        away[floating.idx_q] += 3.0
        assert not model.in_collision(away)
    for factory in factories:
        assert factory(model, names, context) is not None
    invalid = context.copy()
    invalid[floating.idx_q + 6] = 2.0
    with pytest.raises(ValueError, match="manifold"):
        model.validate_configuration(invalid)
    with pytest.raises(ValueError, match="manifold"):
        model.configuration_with_joint_positions(invalid, names, [0.2, 0.3])
    for factory in factories:
        with pytest.raises(ValueError, match="manifold"):
            factory(model, names, invalid)
    with pytest.raises(ValueError, match="scalar"):
        model.create_joint_position_map(["floating_base"])
    with pytest.raises(TypeError, match="real"):
        model.validate_configuration(full.astype(complex))


def test_python_pinocchio_mapping_supports_free_flyer_context(urdf):
    pin = python_pinocchio_or_skip()
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    mapping = JointPositionMap.from_model(model, ["turn", "slide"])
    reference = pin.neutral(model)
    reference[:3] = [0.3, 0.4, 0.5]
    q = mapping.configuration_with_joint_positions(reference, [0.6, 0.7])
    assert pin.isNormalized(model, q)
    np.testing.assert_array_equal(q[:7], reference[:7])
    np.testing.assert_allclose(mapping.joint_positions(q), [0.6, 0.7])
    with pytest.raises(ValueError, match="scalar"):
        JointPositionMap.from_model(model, ["root_joint"])


def test_python_pinocchio_fk_matches_named_chains_with_frozen_branch_context(tmp_path):
    import holistic_motion as hm

    if not hasattr(hm, "Robot"):
        pytest.skip("requires native core kinematics")
    pin = python_pinocchio_or_skip()
    links = ['<link name="world"/>', '<link name="base"/>', '<link name="torso"/>']
    joints = [
        '<joint name="floating_base" type="floating">'
        '<parent link="world"/><child link="base"/></joint>',
        '<joint name="torso_turn" type="revolute">'
        '<parent link="base"/><child link="torso"/><axis xyz="0 0 1"/>'
        '<origin xyz="0 0 0.3" rpy="0.03 -0.02 0.01"/>'
        '<limit lower="-2" upper="2" effort="1" velocity="1"/></joint>',
    ]
    # Reverse XML branch order and use names unrelated to scalar indices.
    # All branches mix scalar, continuous-pair and prismatic coordinates.
    for branch in ("z_arm", "m_leg", "a_arm"):
        parent = "torso"
        for index in range(7):
            child = f"{branch}_link_{index}"
            links.append(f'<link name="{child}"/>')
            kind, axis = (
                ("continuous", "0 0 1.5"),
                ("revolute", "0 -2 0"),
                ("prismatic", "0.3 0.4 0"),
            )[index % 3]
            joints.append(
                f'<joint name="{branch}_joint_{index}" type="{kind}">'
                f'<parent link="{parent}"/><child link="{child}"/>'
                f'<axis xyz="{axis}"/><origin xyz="0.13 0.03 -0.01" '
                'rpy="0.07 -0.03 0.02"/>'
                '<limit lower="-2" upper="2" effort="1" velocity="1"/></joint>'
            )
            parent = child
    urdf = tmp_path / "branches.urdf"
    urdf.write_text('<robot name="branches">' + "".join(links + joints) + "</robot>")
    robot = hm.Robot(str(urdf))
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    assert robot.nq == robot.nv == 22
    assert model.nv == 28 and model.nq == 38
    generator = np.random.default_rng(451)
    context = pin.integrate(
        model, pin.neutral(model), generator.uniform(-0.4, 0.4, model.nv)
    )
    original = context.copy()
    scalar_reference = generator.uniform(-0.4, 0.4, robot.nq)
    for branch in ("a_arm", "m_leg", "z_arm"):
        tip = f"{branch}_link_6"
        names = robot.chain_joint_names("torso", tip)
        assert names == [f"{branch}_joint_{index}" for index in range(7)]
        chain = robot.create_kinematics("torso", tip)
        mapping = JointPositionMap.from_model(model, names)
        scalar_map = robot.create_joint_position_map(names)
        changed_indices = {
            index
            for joint in mapping.joints
            for index in range(joint.idx_q, joint.idx_q + joint.nq)
        }
        frozen_indices = sorted(set(range(model.nq)) - changed_indices)
        for positions in generator.uniform(-0.7, 0.7, (8, 7)):
            full = mapping.configuration_with_joint_positions(context, positions)
            np.testing.assert_array_equal(full[frozen_indices], context[frozen_indices])
            assert pin.isNormalized(model, full)
            pin.computeJointJacobians(model, data, full)
            pin.updateFramePlacements(model, data)
            world_T_torso = data.oMf[model.getFrameId("torso")].homogeneous
            world_T_tip = data.oMf[model.getFrameId(tip)].homogeneous
            np.testing.assert_allclose(
                world_T_torso @ chain.forward(positions), world_T_tip, atol=2e-12
            )
            native_jacobian = chain.jacobian(positions)
            native_jacobian[:3] = world_T_torso[:3, :3] @ native_jacobian[:3]
            native_jacobian[3:] = world_T_torso[:3, :3] @ native_jacobian[3:]
            full_jacobian = pin.getFrameJacobian(
                model, data, model.getFrameId(tip), pin.LOCAL_WORLD_ALIGNED
            )
            np.testing.assert_allclose(
                native_jacobian,
                full_jacobian[:, [joint.idx_v for joint in mapping.joints]],
                atol=2e-12,
            )
            scalar_full = scalar_map.configuration_with_joint_positions(
                scalar_reference, positions
            )
            np.testing.assert_allclose(
                scalar_map.joint_positions(scalar_full), positions
            )
        np.testing.assert_array_equal(context, original)
