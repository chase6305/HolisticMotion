"""A shared planning-space snapshot preserves metric, topology and vector order."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "lower,direction", [(-np.pi, 1.0), (0.0, 1.0), (-2 * np.pi, -1.0)]
)
@pytest.mark.parametrize("distance", [0.1, 1e-10, 1e-16, 1e-100])
def test_continuous_interpolation_preserves_small_in_range_angles(
    lower, direction, distance
):
    import holistic_motion as hm

    planner = hm.SamplingPlanner([lower], [lower + 2 * np.pi])
    planner.set_continuous_joints([0])
    options = hm.PlanningOptions()
    options.interpolate_path = True
    options.interpolation_points = 3
    options.simplify_path = False
    goal = direction * distance
    result = planner.plan([0.0], [goal], options)

    assert result.success
    np.testing.assert_array_equal(result.path, [[0.0], [goal / 2], [goal]])


@pytest.mark.parametrize("scale", [1e-10, 1e-20, 1e-100, 1e-150])
def test_continuous_optimizer_preserves_the_requested_small_step(scale):
    import holistic_motion as hm

    optimizer = hm.PathOptimizer([-np.pi] * 2, [np.pi] * 2)
    optimizer.set_continuous_joints([0, 1])
    options = hm.PathOptimizationOptions()
    options.length_weight = 1.0
    options.smoothness_weight = 0.0
    options.minimum_improvement = 0.0
    options.step_size = 0.35 * scale
    options.max_iterations = 1
    path = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]) * scale
    result = optimizer.optimize(path, options)

    assert result.success
    assert result.statistics.accepted_updates == 1
    np.testing.assert_allclose(
        np.asarray(result.path)[1], np.array([0.5, 0.65]) * scale, rtol=1e-14, atol=0.0
    )
    np.testing.assert_array_equal(np.asarray(result.path)[[0, -1]], path[[0, -1]])


def test_unwrap_rejects_lifting_that_changes_a_source_angle_phase():
    import holistic_motion as hm

    origin = 1e16
    space = hm.JointSpace([origin - 8], [origin + 8]).with_continuous_joints([0])
    source = [[origin], [origin - 8]]
    # A nonzero shortest step rounds to origin-2 at this offset. That result
    # has a different sine/cosine pair from the requested origin-8 endpoint.
    with pytest.raises(OverflowError, match="phase"):
        space.unwrap_path(source)


@pytest.mark.parametrize("origin", [0.0, 1e16])
def test_unwrap_keeps_representable_source_steps_at_large_offsets(origin):
    import holistic_motion as hm

    space = hm.JointSpace([origin - 8], [origin + 8]).with_continuous_joints([0])
    source = [[origin], [origin - 2]]
    np.testing.assert_array_equal(space.unwrap_path(source), source)


@pytest.mark.parametrize("operation", ["plan", "optimize"])
def test_result_space_survives_its_request_and_producer(operation):
    import gc

    import holistic_motion as hm

    space = hm.JointSpace([-np.pi], [np.pi], ["turn"])
    space = space.with_continuous_joints([0]).with_joint_weights([4.0])
    path = [[3.0], [-3.0]]
    if operation == "plan":
        producer = hm.SamplingPlanner(space)
        result = producer.plan(*path)
    else:
        producer = hm.PathOptimizer(space)
        result = producer.optimize(path)
    assert result.success
    retained = result.joint_space
    del space, producer, result
    gc.collect()
    assert retained.joint_names == ["turn"]
    assert retained.continuous_joint_indices == [0]
    np.testing.assert_array_equal(retained.joint_weights, [4.0])
    np.testing.assert_allclose(retained.unwrap_path(path), [[3.0], [2 * np.pi - 3.0]])


@pytest.mark.parametrize("operation", ["plan", "optimize"])
def test_invalid_requests_still_report_their_coordinate_contract(operation):
    import holistic_motion as hm

    space = hm.JointSpace([-1.0], [1.0], ["slide"]).with_joint_weights([2.0])
    result = (
        hm.SamplingPlanner(space).plan([-2.0], [0.0])
        if operation == "plan"
        else hm.PathOptimizer(space).optimize([[-2.0], [0.0]])
    )
    assert not result.success
    assert result.path == []
    assert result.joint_space.joint_names == ["slide"]
    np.testing.assert_array_equal(result.joint_space.joint_weights, [2.0])


def test_joint_space_retains_the_same_metric_between_planning_and_optimization():
    import holistic_motion as hm

    names = ["turn", "slide"]
    lower = np.array([-np.pi, -1.0])
    upper = np.array([np.pi, 1.0])
    weights = np.array([4.0, 0.25])
    space = hm.JointSpace(lower, upper, names)
    configured = space.with_joint_weights(weights).with_continuous_joints([0, 0])
    assert space.continuous_joint_indices == []
    assert configured.continuous_joint_indices == [0]
    assert configured.joint_names == names
    assert configured.dof == 2
    names.reverse()
    lower.fill(0)
    upper.fill(0)
    weights.fill(0)
    configured.lower_limits.fill(0)
    configured.joint_weights.fill(0)
    assert configured.joint_names == ["turn", "slide"]
    np.testing.assert_array_equal(configured.joint_weights, [4.0, 0.25])
    np.testing.assert_array_equal(configured.lower_limits, [-np.pi, -1.0])

    checked = []

    def valid(q):
        checked.append(q.copy())
        # The shortest continuous arc stays close to +/-pi. A bounded-angle
        # interpretation would incorrectly pass through the forbidden origin.
        return abs(q[0]) > 2.5

    planner = hm.SamplingPlanner(configured, state_validator=valid)
    snapshot = planner.joint_space
    optimizer = hm.PathOptimizer(snapshot, state_validator=valid)
    start, goal = [np.pi - 0.1, 0.0], [-np.pi + 0.1, 0.4]
    result = planner.plan(start, goal)
    assert result.success and checked
    assert result.joint_space.joint_names == ["turn", "slide"]
    assert result.statistics.final_path_length == pytest.approx(np.sqrt(0.2))
    options = hm.PathOptimizationOptions()
    options.smoothness_weight = 0.0
    optimized = optimizer.optimize(result.path, options)
    assert optimized.success
    assert optimized.statistics.initial_path_length == pytest.approx(
        result.statistics.final_path_length
    )

    # Legacy setters publish a replacement snapshot only on that consumer.
    # An optimizer built from an earlier snapshot keeps its original metric.
    planner.set_joint_weights([1.0, 1.0])
    planner.set_continuous_joints([])
    np.testing.assert_array_equal(snapshot.joint_weights, [4.0, 0.25])
    assert snapshot.continuous_joint_indices == [0]
    assert optimizer.joint_space.continuous_joint_indices == [0]
    assert result.joint_space.continuous_joint_indices == [0]
    assert optimized.joint_space.continuous_joint_indices == [0]
    np.testing.assert_array_equal(result.joint_space.joint_weights, [4.0, 0.25])
    assert planner.joint_space.continuous_joint_indices == []
    with pytest.raises(ValueError):
        optimizer.set_joint_weights([1.0, 0.0])
    with pytest.raises(IndexError):
        optimizer.set_continuous_joints([2])
    assert optimizer.joint_space.continuous_joint_indices == [0]
    np.testing.assert_array_equal(optimizer.joint_space.joint_weights, [4.0, 0.25])


@pytest.mark.parametrize("names", [["x"], ["x", "x"], ["x", ""]])
def test_joint_space_rejects_ambiguous_joint_order(names):
    import holistic_motion as hm

    with pytest.raises(ValueError, match="joint names"):
        hm.JointSpace([-1, -1], [1, 1], names)


@pytest.mark.parametrize("change", ["weights", "periodic"])
def test_joint_space_builders_reject_complex_inputs(change):
    import holistic_motion as hm

    space = hm.JointSpace([-np.pi], [np.pi])
    method = (
        space.with_joint_weights
        if change == "weights"
        else space.with_continuous_joints
    )
    with pytest.raises(TypeError, match="real"):
        method([1 + 0j])


def test_joint_space_preserves_legacy_limit_and_period_validation():
    import holistic_motion as hm

    for lower, upper in [([], []), ([0], [0]), ([1], [-1]), ([0], [np.inf])]:
        with pytest.raises(ValueError, match="joint limits"):
            hm.JointSpace(lower, upper)
    space = hm.JointSpace([-1], [1])
    with pytest.raises(ValueError, match="2\\*pi"):
        space.with_continuous_joints([0])
    with pytest.raises(ValueError, match="weights"):
        space.with_joint_weights([np.nan])
    assert space.continuous_joint_indices == []


def test_unwrapped_periodic_plan_stays_on_the_checked_arc_during_retiming():
    import holistic_motion as hm

    space = hm.JointSpace([-np.pi, -1], [np.pi, 1], ["turn", "slide"])
    space = space.with_continuous_joints([0]).with_joint_weights([1, 1])

    def valid(q):
        phase = np.arctan2(np.sin(q[0]), np.cos(q[0]))
        return abs(phase) > 2.5

    planned = hm.SamplingPlanner(space, valid).plan(
        [np.pi - 0.1, 0.0], [-np.pi + 0.1, 0.4]
    )
    assert planned.success
    path = np.asarray(space.unwrap_path(planned.path))
    np.testing.assert_allclose(np.diff(path[:, 0]), [0.2], atol=1e-14)
    np.testing.assert_array_equal(path[:, 1], np.asarray(planned.path)[:, 1])
    assert path[-1, 0] > np.pi
    trajectory = hm.RnTrajectory(path, [1, 1], [2, 2], [4, 4], blend_tolerance=0.0)
    positions = trajectory.sample_uniform(101)[1]
    assert all(valid(q) for q in positions)
    np.testing.assert_allclose(positions[[0, -1]], path[[0, -1]], atol=1e-14)
    # Using the representative angles as Euclidean scalars would cross the
    # forbidden origin even though the planner validated the short circle arc.
    assert not valid(np.mean(planned.path, axis=0))


def test_unwrap_path_keeps_winding_and_rejects_invalid_source_states():
    import holistic_motion as hm

    space = hm.JointSpace([-np.pi, -1], [np.pi, 1]).with_continuous_joints([0])
    angles = np.arange(25) * 0.5
    source = np.column_stack((np.arctan2(np.sin(angles), np.cos(angles)), np.zeros(25)))
    np.testing.assert_allclose(np.asarray(space.unwrap_path(source))[:, 0], angles)
    np.testing.assert_array_equal(space.unwrap_path(source[:1]), source[:1])
    for bad in ([], [[0]], [[np.nan, 0]], [[0, 2]], [[4, 0]]):
        with pytest.raises(ValueError):
            space.unwrap_path(bad)
    with pytest.raises(TypeError, match="real"):
        space.unwrap_path([[0 + 0j, 0]])
    # After one full turn, a subnormal step can no longer be represented in
    # the lifted scalar. Return an explicit failure instead of a frozen step.
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(OverflowError, match="represented"):
        space.unwrap_path([[0, 0], [2, 0], [-2, 0], [0, 0], [tiny, 0]])


@pytest.mark.parametrize("kind", ["mesh", "sphere"])
def test_named_collision_factories_preserve_joint_order(tmp_path, kind):
    import holistic_motion as hm

    if not hasattr(hm, "CollisionModel"):
        pytest.skip("collision component is unavailable")
    urdf = tmp_path / "named.urdf"
    urdf.write_text(
        "<robot name='named'><link name='base'/><link name='arm'/><link name='tip'>"
        "<collision><geometry><sphere radius='0.1'/></geometry></collision></link>"
        "<joint name='turn' type='continuous'><parent link='base'/><child link='arm'/>"
        "<axis xyz='0 0 1'/><limit effort='1' velocity='1'/></joint>"
        "<joint name='slide' type='prismatic'><parent link='arm'/><child link='tip'/>"
        "<axis xyz='1 0 0'/><limit lower='-1' upper='1' velocity='1' effort='1'/>"
        "</joint></robot>"
    )
    if kind == "mesh":
        model = hm.CollisionModel(str(urdf))
        factory = "from_collision_joints"
    else:
        model = hm.SphereCollisionModel(
            str(urdf), [hm.CollisionSphere("tip", "tip", [0, 0, 0], 0.1)]
        )
        factory = "from_sphere_collision_joints"
    names = ["slide", "turn"]
    for consumer in (hm.SamplingPlanner, hm.PathOptimizer):
        instance = getattr(consumer, factory)(
            model, names, model.neutral_configuration()
        )
        assert instance.joint_space.joint_names == names
        assert instance.joint_space.continuous_joint_indices == [1]
