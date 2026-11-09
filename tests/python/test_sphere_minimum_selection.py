"""Minimum scans must agree with an unpruned enumeration, including ties."""

import itertools

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


def model_with_pairs(tmp_path, radii, centers):
    path = tmp_path / "selection.urdf"
    links = ['<link name="base"/>']
    for index in range(len(radii)):
        links.append(
            f'<link name="link{index}"/>'
            f'<joint name="mount{index}" type="fixed">'
            f'<parent link="base"/><child link="link{index}"/></joint>'
        )
    path.write_text('<robot name="selection">' + "".join(links) + "</robot>")
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere(f"sphere{i}", f"link{i}", center, radius)
            for i, (center, radius) in enumerate(zip(centers, radii))
        ],
    )
    groups = {f"g{i}": [f"link{i}"] for i in range(len(radii))}
    model.set_collision_groups(
        groups, [(f"g{i}", f"g{i + 1}") for i in range(0, len(radii), 2)]
    )
    return model


def assert_enumerated_minimum(model, q):
    # distances() evaluates the signed distance of every selected pair. Its
    # filtering threshold is deliberately too large to prune any finite pair.
    distances = model.distances(q, np.finfo(float).max)
    expected = min(distances, key=lambda pair: pair.distance)
    actual = model.minimum_distance(q)
    assert actual.pair_index == expected.pair_index
    assert actual.distance == expected.distance
    np.testing.assert_array_equal(actual.normal, expected.normal)
    combined = model.minimum_distance_with_gradient(q)
    assert combined.distance_result.pair_index == expected.pair_index
    assert combined.distance_result.distance == expected.distance
    np.testing.assert_array_equal(combined.gradient, np.zeros(model.nv))
    np.testing.assert_array_equal(
        model.batch_minimum_distances(np.tile(q, (3, 1))), [expected.distance] * 3
    )
    return distances


def test_bound_rounding_cannot_discard_a_one_ulp_better_pair(tmp_path):
    radius, separation = 0.02089590672068266, 0.06019950810059945
    model = model_with_pairs(
        tmp_path,
        [radius, np.nextafter(radius, 0), radius, radius],
        [[0, 0, 0], [separation, 0, 0]] * 2,
    )
    distances = assert_enumerated_minimum(model, [])
    first, second = (pair.distance for pair in distances)
    assert second == np.nextafter(first, -np.inf)
    # This inward-rounded bound would incorrectly skip the winning pair.
    naive_bound = first + radius + radius
    assert separation**2 > naive_bound**2


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
def test_ties_keep_the_first_active_pair_after_reordering(tmp_path, permutation):
    # Binary-exact distances and radii keep every pair's clearance identical.
    radii = [0.125] * 6
    pairs = [([0, 0, 0], [1, 0, 0]), ([0, 0, 0], [0, 1, 0]), ([0, 0, 0], [0, 0, 1])]
    centers = [point for i in permutation for point in pairs[i]]
    model = model_with_pairs(tmp_path, radii, centers)
    distances = assert_enumerated_minimum(model, [])
    assert len({pair.distance for pair in distances}) == 1
    assert model.minimum_distance([]).pair_index == 0
    model.clear_collision_pairs()
    with pytest.raises(RuntimeError, match="no active"):
        model.minimum_distance([])
    model.set_collision_groups(
        {"all": [f"link{i}" for i in range(6)]}, [("all", "all")]
    )
    assert_enumerated_minimum(model, [])


@pytest.mark.parametrize("scale", [1e-200, 1e-160, 1e-14, 1, 1e150])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_sparse_dense_and_exceptional_ranges_match_full_enumeration(
    tmp_path, scale, heterogeneous
):
    rng = np.random.default_rng(451)
    count = 24
    centers = rng.uniform(-2, 2, (count, 3)) * scale
    radii = np.full(count, 0.1 * scale)
    if heterogeneous:
        radii *= np.exp2(rng.uniform(-10, 10, count))
    model = model_with_pairs(tmp_path, radii, centers)
    assert_enumerated_minimum(model, [])
    model.reset_collision_pairs()
    assert_enumerated_minimum(model, [])


def test_large_inactive_radius_only_weakens_the_bound(tmp_path):
    model = model_with_pairs(
        tmp_path,
        [1e200, 1e200, 0.1, 0.1, 0.1, 0.1],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0.5, 0, 0]],
    )
    model.set_collision_groups(
        {"a": ["link2"], "b": ["link3"], "c": ["link4"], "d": ["link5"]},
        [("a", "b"), ("c", "d")],
    )
    assert_enumerated_minimum(model, [])


@pytest.mark.parametrize("scale", [1e-200, 1e-14, 1.0, 1e150])
def test_scalar_gradient_and_batch_follow_the_changing_minimum(tmp_path, scale):
    path = tmp_path / "moving.urdf"
    path.write_text(
        '<robot name="moving"><link name="base"/><link name="tip"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tip"/><axis xyz="1 0 0"/>'
        '<limit lower="-1e151" upper="1e151" effort="1" velocity="1"/>'
        "</joint></robot>"
    )
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("moving", "tip", [0, 0, 0], 0.1 * scale),
            hm.CollisionSphere("positive", "base", [scale, 0, 0], 0.1 * scale),
            hm.CollisionSphere("negative", "base", [-scale, 0, 0], 0.1 * scale),
        ],
    )
    states = scale * np.array([0.8, -0.8, 0, 0.8, 0.8, -0.8, -0.8])
    expected = []
    for value in states:
        reference = min(
            model.distances([value], np.finfo(float).max),
            key=lambda pair: pair.distance,
        )
        result = model.minimum_distance_with_gradient([value])
        assert result.distance_result.pair_index == reference.pair_index
        assert result.distance_result.distance == reference.distance
        np.testing.assert_allclose(result.gradient, [1 if value < 0 else -1])
        if value != 0:
            step = 1e-6 * scale
            derivative = (
                model.minimum_distance([value + step]).distance
                - model.minimum_distance([value - step]).distance
            ) / (2 * step)
            np.testing.assert_allclose(result.gradient, [derivative], rtol=2e-9)
        expected.append(reference.distance)
    np.testing.assert_array_equal(
        model.batch_minimum_distances(states[:, None]), expected
    )
