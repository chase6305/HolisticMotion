"""Planning inputs must not silently discard imaginary components."""

import warnings

import holistic_motion as hm
import numpy as np
import pytest

COMPLEX_VECTORS = [
    np.array([0.25 + 1j]),
    np.array([0.25 + 1j], dtype=object),
    [0.25 + 1j],
    (np.complex128(0.25 + 1j),),
]


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("bad", COMPLEX_VECTORS)
@pytest.mark.parametrize("argument", ["lower_limits", "upper_limits"])
def test_planning_rejects_complex_limits(factory, bad, argument):
    args = {"lower_limits": [0.0], "upper_limits": [1.0], argument: bad}
    with (
        warnings.catch_warnings(record=True) as emitted,
        pytest.raises(TypeError, match="real"),
    ):
        factory(**args)
    assert not emitted


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("bad", COMPLEX_VECTORS)
def test_planning_rejects_complex_weights(factory, bad):
    planner = factory([0.0], [1.0])
    with pytest.raises(TypeError, match="real"):
        planner.set_joint_weights(bad)


@pytest.mark.parametrize("bad", COMPLEX_VECTORS)
@pytest.mark.parametrize("argument", ["start", "goal"])
def test_sampling_rejects_complex_endpoints(bad, argument):
    planner = hm.SamplingPlanner([0.0], [1.0])
    args = {"start": [0.0], "goal": [1.0], argument: bad}
    with pytest.raises(TypeError, match="real"):
        planner.plan(**args)


@pytest.mark.parametrize("bad", COMPLEX_VECTORS)
def test_optimizer_rejects_complex_waypoints(bad):
    optimizer = hm.PathOptimizer([0.0], [1.0])
    with pytest.raises(TypeError, match="real"):
        optimizer.optimize([[0.0], bad, [1.0]])


@pytest.mark.parametrize(
    "options_type,field",
    [
        (hm.PlanningOptions, field)
        for field in (
            "timeout_seconds",
            "extension_range",
            "goal_bias",
            "edge_resolution",
        )
    ]
    + [
        (hm.PathOptimizationOptions, field)
        for field in (
            "timeout_seconds",
            "step_size",
            "line_search_decay",
            "edge_resolution",
            "length_weight",
            "smoothness_weight",
            "state_cost_weight",
            "finite_difference_step",
            "state_cost_step_size",
            "minimum_improvement",
        )
    ],
)
@pytest.mark.parametrize("bad", [np.complex128(0.25 + 1j), np.array(0.25 + 1j)])
def test_planning_rejects_complex_options_without_mutation(options_type, field, bad):
    options = options_type()
    previous = getattr(options, field)
    with pytest.raises(TypeError, match="real"):
        setattr(options, field, bad)
    assert getattr(options, field) == previous


@pytest.mark.parametrize("bad", COMPLEX_VECTORS)
def test_optimizer_rejects_complex_cost_gradient(bad):
    optimizer = hm.PathOptimizer([0.0], [1.0])
    optimizer.set_state_cost(lambda _state: 1.0)
    optimizer.set_state_cost_gradient(lambda _state: bad)
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 1.0
    with pytest.raises(TypeError, match="real"):
        optimizer.optimize([[0.0], [0.6], [1.0]], options)


@pytest.mark.parametrize("bad", [np.complex128(1.0 + 1j), np.array(1.0 + 1j)])
def test_optimizer_rejects_complex_state_cost(bad):
    optimizer = hm.PathOptimizer([0.0], [1.0])
    optimizer.set_state_cost(lambda _state: bad)
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 1.0
    with pytest.raises(TypeError, match="real"):
        optimizer.optimize([[0.0], [0.6], [1.0]], options)


@pytest.mark.parametrize("factory", [hm.PathOptimizer, hm.SamplingPlanner])
def test_planning_still_accepts_real_strided_and_integer_inputs(factory):
    planner = factory(np.array([0, 9])[::2], (1,))
    planner.set_joint_weights(np.array([1.0, 9.0])[::2])
    if factory is hm.PathOptimizer:
        planner.set_state_cost(None)
        planner.set_state_cost_gradient(None)
        result = planner.optimize(np.array([[0, 9], [1, 9]])[:, ::2])
    else:
        result = planner.plan(np.array([0.0, 9.0])[::2], [1])
    assert result.success


@pytest.fixture()
def collision_models(tmp_path):
    if not hasattr(hm, "SphereCollisionModel"):
        pytest.skip("native collision support is disabled")
    from test_sphere_collision import URDF

    urdf = tmp_path / "collision.urdf"
    urdf.write_text(
        URDF.replace(
            '<link name="left"/>',
            '<link name="left"><collision><geometry><sphere radius="0.1"/>'
            "</geometry></collision></link>",
        ).replace(
            '<link name="right"/>',
            '<link name="right"><collision><geometry><sphere radius="0.1"/>'
            "</geometry></collision></link>",
        )
    )
    return hm.SphereCollisionModel(
        str(urdf), [hm.CollisionSphere("left", "left", [0.0, 0.0, 0.0], 0.1)]
    ), hm.CollisionModel(str(urdf))


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize(
    "argument", ["lower_limits", "upper_limits", "security_margin"]
)
def test_sphere_planning_factory_rejects_complex_inputs(
    collision_models, factory, argument
):
    sphere, _ = collision_models
    args = {"lower_limits": [-1.0], "upper_limits": [1.0], "collision_model": sphere}
    args[argument] = (
        np.complex128(0.25 + 1j)
        if argument == "security_margin"
        else COMPLEX_VECTORS[0]
    )
    with pytest.raises(TypeError, match="real"):
        factory.from_sphere_collision_model(**args)


def test_sphere_optimizer_factory_rejects_complex_clearance(collision_models):
    sphere, _ = collision_models
    with pytest.raises(TypeError, match="real"):
        hm.PathOptimizer.from_sphere_collision_model(
            [-1.0], [1.0], sphere, clearance=np.complex128(1 + 1j)
        )


@pytest.mark.parametrize(
    "argument", ["lower_limits", "upper_limits", "context_configuration"]
)
def test_full_collision_planning_factory_rejects_complex_inputs(
    collision_models, argument
):
    _, collision = collision_models
    args = {
        "lower_limits": [-1.0],
        "upper_limits": [1.0],
        "collision_model": collision,
        "active_indices": [0],
        "context_configuration": [0.0],
    }
    args[argument] = COMPLEX_VECTORS[0]
    with pytest.raises(TypeError, match="real"):
        hm.SamplingPlanner.from_collision_model(**args)


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
@pytest.mark.parametrize("argument", ["context_configuration", "security_margin"])
def test_named_collision_planning_factory_rejects_complex_inputs(
    collision_models, factory, argument
):
    _, collision = collision_models
    args = {
        "collision_model": collision,
        "joint_names": ["left_slide"],
        "context_configuration": [0.0],
    }
    args[argument] = (
        np.complex128(0.25 + 1j)
        if argument == "security_margin"
        else COMPLEX_VECTORS[0]
    )
    with pytest.raises(TypeError, match="real"):
        factory.from_collision_joints(**args)


@pytest.mark.parametrize("argument", ["start", "preferred_direction", "step_size"])
def test_null_space_planning_rejects_complex_inputs(tmp_path, argument):
    from test_srs_kinematics import _write_ideal_srs_urdf

    urdf = tmp_path / "srs.urdf"
    _write_ideal_srs_urdf(urdf)
    robot = hm.Robot(str(urdf))
    planner = hm.NullSpacePlanner(robot.kinematics)
    args = {"start": np.zeros(7), "preferred_direction": np.ones(7), "steps": 1}
    args[argument] = (
        np.complex128(0.01 + 1j)
        if argument == "step_size"
        else np.ones(7, dtype=complex) * (0.1 + 1j)
    )
    with pytest.raises(TypeError, match="real"):
        planner.plan(**args)


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
def test_named_collision_factories_preserve_real_configuration_support(
    collision_models, factory
):
    _, collision = collision_models
    planner = factory.from_collision_joints(collision, ["left_slide"], np.array([0.0]))
    if factory is hm.PathOptimizer:
        result = planner.optimize([[0.0], [0.1]])
    else:
        result = planner.plan([0.0], [0.1])
    assert result.success


def test_full_collision_factory_preserves_default_context(collision_models):
    _, collision = collision_models
    planner = hm.SamplingPlanner.from_collision_model([-1.0], [1.0], collision)
    assert planner.plan([0.0], [0.1]).success


def test_optimizer_rejects_complex_waypoints_nested_in_object_arrays():
    path = np.empty(3, dtype=object)
    for index, position in enumerate((0.0, 0.5, 1.0)):
        path[index] = np.array([np.complex128(position + 1j)], dtype=object)
    optimizer = hm.PathOptimizer([0.0], [1.0])
    with (
        warnings.catch_warnings(record=True) as emitted,
        pytest.raises(TypeError, match="real"),
    ):
        optimizer.optimize(path)
    assert not emitted


def test_optimizer_rejects_cyclic_object_path_without_recursing_forever():
    path = np.empty(1, dtype=object)
    path[0] = path
    optimizer = hm.PathOptimizer([0.0], [1.0])
    with pytest.raises(TypeError, match="shape"):
        optimizer.optimize(path)


def test_optimizer_still_accepts_real_object_waypoints():
    path = np.empty(3, dtype=object)
    for index, position in enumerate((0.0, 0.5, 1.0)):
        path[index] = np.array([position], dtype=object)
    assert hm.PathOptimizer([0.0], [1.0]).optimize(path).success
