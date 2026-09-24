"""Sphere queries should agree across length scales and translated frames."""

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


def slider_model(tmp_path, radius, offset=0.0, second_radius=None):
    path = tmp_path / "slider.urdf"
    path.write_text(
        '<robot name="slider"><link name="base"/><link name="moving"/>'
        '<joint name="slide" type="prismatic">'
        '<parent link="base"/><child link="moving"/><axis xyz="0 1 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    return hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("base", "base", [offset, 0.0, 0.0], radius),
            hm.CollisionSphere(
                "moving",
                "moving",
                [offset, 0.0, 0.0],
                radius if second_radius is None else second_radius,
            ),
        ],
    )


@pytest.mark.parametrize("scale", [1e-200, 1e-160, 1e-14, 1.0, 1e160, 1e200])
def test_distance_collision_normal_and_gradient_follow_length_scale(tmp_path, scale):
    model = slider_model(tmp_path, 0.25 * scale)
    for factor in (0.25, 0.5, 1.0):
        q = [factor * scale]
        result = model.minimum_distance_with_gradient(q)
        expected = factor - 0.5
        assert result.distance_result.distance / scale == pytest.approx(expected)
        np.testing.assert_allclose(result.distance_result.normal, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(result.gradient, [1.0])
        assert model.in_collision(q) == (factor <= 0.5)
        assert model.in_collision(q, stop_at_first=False) == (factor <= 0.5)
        assert model.in_collision(q, security_margin=0.6 * scale)
        assert len(model.distances(q, 0.0)) == int(factor <= 0.5)
    np.testing.assert_allclose(
        model.batch_minimum_distances([[0.25 * scale], [scale]]) / scale,
        [-0.25, 0.5],
    )


def test_gradient_uses_center_separation_before_subtracting_large_radii(tmp_path):
    model = slider_model(tmp_path, 1e16)
    result = model.minimum_distance_with_gradient([0.1])
    assert result.distance_result.distance == pytest.approx(-2e16)
    np.testing.assert_allclose(result.distance_result.normal, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(result.gradient, [1.0])


def test_coincident_centers_have_deterministic_zero_gradient(tmp_path):
    model = slider_model(tmp_path, 0.25)
    result = model.minimum_distance_with_gradient([0.0])
    assert result.distance_result.distance == -0.5
    np.testing.assert_array_equal(result.gradient, [0.0])
    assert np.isfinite(result.distance_result.normal).all()


def test_gradient_preserves_small_local_offset_after_large_translation(tmp_path):
    path = tmp_path / "translated.urdf"
    path.write_text(
        '<robot name="translated"><link name="base"/><link name="arm"/>'
        '<joint name="hinge" type="revolute">'
        '<parent link="base"/><child link="arm"/>'
        '<origin xyz="1e16 0 0"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("fixed", "base", [1e16, 1.0, 0.0], 0.1),
            hm.CollisionSphere("arm", "arm", [0.25, 0.0, 0.0], 0.1),
        ],
    )
    result = model.minimum_distance_with_gradient([0.0])
    step = 1e-5
    finite_difference = (
        model.minimum_distance([step]).distance
        - model.minimum_distance([-step]).distance
    ) / (2.0 * step)
    np.testing.assert_allclose(result.gradient, [finite_difference], atol=1e-10)
    np.testing.assert_allclose(result.gradient, [-0.25], atol=1e-10)


def test_finite_clearance_survives_overflowing_center_difference(tmp_path):
    path = tmp_path / "fixed.urdf"
    path.write_text('<robot name="fixed"><link name="base"/></robot>')
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("first", "base", [-1e308, 0, 0], 8e307),
            hm.CollisionSphere("second", "base", [1e308, 0, 0], 8e307),
        ],
        exclude_same_link=False,
    )
    result = model.minimum_distance([])
    assert result.distance / 1e307 == pytest.approx(4.0)
    np.testing.assert_allclose(result.normal, [1.0, 0.0, 0.0])
    assert not model.in_collision([])
    assert model.in_collision([], security_margin=5e307)


def test_unrepresentable_distance_is_reported(tmp_path):
    model = slider_model(tmp_path, 1e308)
    with pytest.raises(OverflowError, match="sphere distance"):
        model.minimum_distance([0.0])
    assert model.in_collision([0.0])


def test_unrepresentable_world_coordinates_are_reported(tmp_path):
    path = tmp_path / "overflow.urdf"
    path.write_text(
        '<robot name="overflow"><link name="base"/><link name="moving"/>'
        '<joint name="slide" type="prismatic">'
        '<parent link="base"/><child link="moving"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    model = hm.SphereCollisionModel(
        str(path),
        [hm.CollisionSphere("moving", "moving", [1e308, 0, 0], 0.1)],
    )
    with pytest.raises(OverflowError, match="world sphere coordinates"):
        model.world_spheres([1e308])
    with pytest.raises(OverflowError, match="world sphere coordinates"):
        model.in_collision([1e308])


@pytest.mark.parametrize(
    "radii", [(0.1, 0.4), (1e-200, 0.4), (1e-200, 1e200), (1e160, 1e160)]
)
@pytest.mark.parametrize("margin", [0.0, 0.5, 1e155, 1e308])
def test_collision_radius_range_dispatch_matches_extended_reference(
    tmp_path, radii, margin
):
    model = slider_model(tmp_path, radii[0], second_radius=radii[1])
    threshold = sum(np.longdouble(value) for value in (*radii, margin))
    for position in (0.0, 1e-200, 0.25, 1.0, 1e200, 1e308):
        expected = bool(np.longdouble(position) <= threshold)
        for stop in (False, True):
            assert (
                model.in_collision(
                    [position], security_margin=margin, stop_at_first=stop
                )
                == expected
            )

    model.clear_collision_pairs()
    assert not model.in_collision([0.0], security_margin=margin)
    model.set_collision_groups(
        {"fixed": ["base"], "slider": ["moving"]}, [("fixed", "slider")]
    )
    assert model.in_collision([0.0], security_margin=margin)
    model.reset_collision_pairs()
    assert model.in_collision([0.0], security_margin=margin)


def test_sphere_property_returns_independent_geometry_snapshots(tmp_path):
    model = slider_model(tmp_path, 0.1)
    revision = model.pair_revision
    snapshot = model.spheres
    snapshot[0].radius = 1e200
    snapshot[0].center = [1e200, 1e200, 1e200]
    snapshot[0].name = "edited"
    snapshot[0].link_name = "missing"

    current = model.spheres
    assert current[0].radius == 0.1
    assert current[0].name == "base"
    assert current[0].link_name == "base"
    np.testing.assert_array_equal(current[0].center, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(model.world_spheres([1.0])[:, 3], [0.1, 0.1])
    assert model.minimum_distance([1.0]).distance == pytest.approx(0.8)
    assert not model.in_collision([1.0])
    assert model.pair_revision == revision
