"""Collision queries must preserve real configuration and geometry inputs."""

import holistic_motion as hm
import numpy as np
import pytest
from test_planning_real_inputs import collision_models as _collision_models

collision_models = _collision_models

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


@pytest.mark.parametrize(
    "model_index,method",
    [
        (1, method)
        for method in ("in_collision", "collisions", "minimum_distance", "evaluate")
    ]
    + [
        (0, method)
        for method in (
            "world_spheres",
            "in_collision",
            "minimum_distance",
            "minimum_distance_gradient",
            "minimum_distance_with_gradient",
        )
    ],
)
@pytest.mark.parametrize("kind", ["complex", "object"])
def test_collision_queries_reject_complex_configuration(
    collision_models, model_index, method, kind
):
    model = collision_models[model_index]
    bad = np.array(
        [np.complex128(0.1 + 1j)], dtype=complex if kind == "complex" else object
    )
    with pytest.raises(TypeError, match="real"):
        getattr(model, method)(bad)


def test_sphere_batch_query_rejects_complex_rows(collision_models):
    sphere, _ = collision_models
    with pytest.raises(TypeError, match="real"):
        sphere.batch_minimum_distances(np.array([[0.1 + 1j], [0.2]]))


@pytest.mark.parametrize("argument", ["configuration", "maximum_distance"])
def test_sphere_distances_reject_complex_inputs(collision_models, argument):
    sphere, _ = collision_models
    args = {"configuration": [0.0], "maximum_distance": 1.0}
    args[argument] = (
        np.array([0.1 + 1j]) if argument == "configuration" else np.complex128(1 + 1j)
    )
    with pytest.raises(TypeError, match="real"):
        sphere.distances(**args)


@pytest.mark.parametrize("argument", ["configuration", "security_margin"])
def test_collision_margin_query_rejects_complex_inputs(collision_models, argument):
    _, model = collision_models
    args = {"configuration": [0.0], "security_margin": 0.1}
    args[argument] = (
        np.array([0.1 + 1j]) if argument == "configuration" else np.complex128(0.1 + 1j)
    )
    with pytest.raises(TypeError, match="real"):
        model.is_within_distance(**args)


@pytest.mark.parametrize("argument", ["center", "radius"])
def test_collision_sphere_constructor_rejects_complex_inputs(argument):
    args = {"name": "sphere", "link_name": "left", "center": [0.0] * 3, "radius": 0.1}
    args[argument] = (
        np.ones(3, dtype=complex) * (0.1 + 1j)
        if argument == "center"
        else np.complex128(0.1 + 1j)
    )
    with pytest.raises(TypeError, match="real"):
        hm.CollisionSphere(**args)


@pytest.mark.parametrize("argument", ["center", "radius"])
def test_collision_sphere_rejected_assignment_preserves_previous_value(argument):
    sphere = hm.CollisionSphere("sphere", "left", [0.0] * 3, 0.1)
    previous = np.array(getattr(sphere, argument), copy=True)
    bad = (
        np.ones(3, dtype=complex) * (0.1 + 1j)
        if argument == "center"
        else np.complex128(0.1 + 1j)
    )
    with pytest.raises(TypeError, match="real"):
        setattr(sphere, argument, bad)
    np.testing.assert_array_equal(getattr(sphere, argument), previous)


@pytest.mark.parametrize("argument", ["size", "pose"])
def test_obstacle_rejects_complex_geometry_without_adding_it(
    collision_models, argument
):
    _, model = collision_models
    args = {"name": "obstacle", "size": [0.1] * 3, "pose": np.eye(4)}
    args[argument] = (
        np.ones(3, dtype=complex) * (0.1 + 1j)
        if argument == "size"
        else np.eye(4, dtype=complex) + 1j
    )
    count = model.geometry_count
    with pytest.raises(TypeError, match="real"):
        model.add_box_obstacle(**args)
    assert model.geometry_count == count


def test_collision_sphere_center_view_remains_read_only():
    sphere = hm.CollisionSphere("sphere", "left", [0.0] * 3, 0.1)
    view = sphere.center
    assert not view.flags.writeable
    sphere.center = [0.25, 0.0, 0.0]
    assert sphere.center[0] == 0.25


@pytest.mark.parametrize("argument", ["reference", "joint_positions"])
def test_collision_configuration_update_rejects_complex_values(
    collision_models, argument
):
    _, model = collision_models
    args = {"reference": [0.0], "joint_names": ["left_slide"], "joint_positions": [0.1]}
    args[argument] = np.array([0.1 + 1j])
    with pytest.raises(TypeError, match="real"):
        model.configuration_with_joint_positions(**args)


def test_collision_joint_position_mapping_rejects_complex_scalar(collision_models):
    _, model = collision_models
    with pytest.raises(TypeError, match="real"):
        model.configuration_from_joint_positions(
            {"left_slide": np.complex128(0.1 + 1j)}
        )


def test_sphere_collision_margin_rejects_complex_scalar(collision_models):
    sphere, _ = collision_models
    with pytest.raises(TypeError, match="real"):
        sphere.in_collision([0.0], security_margin=np.complex128(0.1 + 1j))


def test_collision_configuration_helpers_preserve_real_inputs(collision_models):
    _, model = collision_models
    mapped = model.configuration_from_joint_positions({"left_slide": np.float64(0.25)})
    np.testing.assert_array_equal(mapped, [0.25])
    updated = model.configuration_with_joint_positions(
        np.array([0.25, 9.0])[::2], ["left_slide"], (0,)
    )
    np.testing.assert_array_equal(updated, [0.0])
