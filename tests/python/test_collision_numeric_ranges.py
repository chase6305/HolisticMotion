"""Unrepresentable collision geometry must not look like a collision-free pose."""

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "CollisionModel"), reason="native collision support is disabled"
)


@pytest.fixture
def large_translation_model(tmp_path):
    path = tmp_path / "large_translation.urdf"
    geometry = '<collision><geometry><sphere radius="0.1"/></geometry></collision>'
    path.write_text(
        f'<robot name="large_translation"><link name="base">{geometry}</link>'
        f'<link name="tip">{geometry}</link>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tip"/><origin xyz="1e308 0 0"/><axis xyz="1 0 0"/>'
        '<limit lower="-1e308" upper="1e308" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    return hm.CollisionModel(str(path), exclude_adjacent=False)


@pytest.mark.parametrize(
    "method",
    [
        "in_collision",
        "collisions",
        "minimum_distance",
        "evaluate",
        "is_within_distance",
    ],
)
def test_mesh_collision_rejects_overflowed_placements_and_recovers(
    large_translation_model, method
):
    model = large_translation_model
    arguments = (0.1,) if method == "is_within_distance" else ()
    with pytest.raises(OverflowError, match="geometry placement"):
        getattr(model, method)([1e308], *arguments)
    # A finite input can cancel the large joint origin and remain queryable.
    assert model.in_collision([-1e308])
    assert model.minimum_distance([-1e308]).distance == pytest.approx(-0.2)


@pytest.mark.parametrize("method", ["minimum_distance", "evaluate"])
def test_mesh_distance_rejects_an_unrepresentable_result_and_recovers(
    large_translation_model, method
):
    model = large_translation_model
    with pytest.raises(OverflowError, match="distance"):
        getattr(model, method)([0.0])
    assert model.in_collision([-1e308])
    assert model.minimum_distance([-1e308]).distance == pytest.approx(-0.2)


@pytest.mark.parametrize("method", ["minimum_distance", "evaluate", "collisions"])
def test_mesh_distance_rejects_overflowed_nearest_points_and_recovers(
    tmp_path, method
):
    path = tmp_path / "large_nearest_points.urdf"
    path.write_text(
        '<robot name="large_nearest_points"><link name="base"><collision>'
        '<origin xyz="1e308 0 0"/><geometry><sphere radius="1e308"/>'
        '</geometry></collision></link><link name="tip"><collision>'
        '<origin xyz="1e308 0 0"/><geometry><sphere radius="0.1"/>'
        '</geometry></collision></link>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tip"/><axis xyz="0 1 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        '</joint></robot>'
    )
    model = hm.CollisionModel(str(path), exclude_adjacent=False)
    # The coincident-center fallback selects X, where the first surface point
    # overflows even though the signed distance itself remains finite.
    with pytest.raises(OverflowError, match="nearest points"):
        getattr(model, method)([0.0])
    # A small Y displacement gives representable nearest points in this same
    # model, exercising recovery without replacing the instance.
    result = model.minimum_distance([0.5])
    assert result.distance == pytest.approx(-1e308)
    assert np.isfinite(result.nearest_point_first).all()
    assert np.isfinite(result.nearest_point_second).all()
