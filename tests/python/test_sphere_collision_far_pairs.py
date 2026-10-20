"""An irrelevant overflowing positive clearance must not hide a local pair."""

import itertools

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


def _model(tmp_path, spheres):
    path = tmp_path / "far_pairs.urdf"
    path.write_text(
        '<robot name="far_pairs"><link name="base"/><link name="unused"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="unused"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    return hm.SphereCollisionModel(str(path), spheres, exclude_same_link=False)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize(
    "query", ["minimum", "gradient", "combined", "batch", "filtered"]
)
def test_far_positive_clearance_does_not_hide_finite_pair(tmp_path, order, query):
    spheres = [
        hm.CollisionSphere("far", "base", [-1.5e308, 0, 0], 0.1),
        hm.CollisionSphere("near_first", "base", [1.5e308, 0, 0], 0.1),
        hm.CollisionSphere("near_second", "base", [1.5e308, 1, 0], 0.1),
    ]
    model = _model(tmp_path, [spheres[index] for index in order])
    if query == "gradient":
        np.testing.assert_array_equal(model.minimum_distance_gradient([0.0]), [0.0])
        return
    if query == "batch":
        np.testing.assert_allclose(
            model.batch_minimum_distances([[0.0], [0.5], [-0.5]]), [0.8] * 3
        )
        return
    if query == "combined":
        result = model.minimum_distance_with_gradient([0.0])
        np.testing.assert_array_equal(result.gradient, [0.0])
        result = result.distance_result
    elif query == "filtered":
        results = model.distances([0.0], 1.0)
        assert len(results) == 1
        result = results[0]
    else:
        result = model.minimum_distance([0.0])
    assert result.distance == pytest.approx(0.8)
    selected = {model.spheres[result.first].name, model.spheres[result.second].name}
    assert selected == {"near_first", "near_second"}
    assert np.isfinite(result.normal).all()


@pytest.mark.parametrize("maximum", [-1.0, 0.0, 1e308, np.finfo(float).max])
def test_filtered_query_skips_all_unrepresentably_far_pairs(tmp_path, maximum):
    model = _model(
        tmp_path,
        [
            hm.CollisionSphere("first", "base", [-1.5e308, 0, 0], 0.1),
            hm.CollisionSphere("second", "base", [1.5e308, 0, 0], 0.1),
        ],
    )
    assert model.distances([0.0], maximum) == []


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("query", ["minimum", "gradient", "combined", "batch"])
def test_unrepresentable_minimum_still_raises(tmp_path, overlap, query):
    model = _model(
        tmp_path,
        [
            hm.CollisionSphere(
                "first",
                "base",
                [0, 0, 0] if overlap else [-1.5e308, 0, 0],
                1e308 if overlap else 0.1,
            ),
            hm.CollisionSphere(
                "second",
                "base",
                [0, 0, 0] if overlap else [1.5e308, 0, 0],
                1e308 if overlap else 0.1,
            ),
        ],
    )
    with pytest.raises(OverflowError, match="sphere distance"):
        if query == "batch":
            model.batch_minimum_distances([[0.0], [0.5]])
        else:
            method = {
                "minimum": model.minimum_distance,
                "gradient": model.minimum_distance_gradient,
                "combined": model.minimum_distance_with_gradient,
            }[query]
            method([0.0])
