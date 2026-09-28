"""Sphere FK and tangent gradients agree with an independent rigid-transform chain."""

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def make_chain(directory, seed, dof=5, sphere_count=13, extra_frames=31):
    random = np.random.default_rng(seed)
    joints = []
    xml = ['<link name="base"/>']
    parent = "base"
    for index in range(2 * dof):
        name = f"link_{index:03}"
        kind = (
            "fixed"
            if index % 2
            else ("revolute", "prismatic", "continuous")[(index // 2) % 3]
        )
        offset = random.uniform(-0.2, 0.2, 3)
        rpy = random.uniform(-1.0, 1.0, 3)
        axis = random.normal(size=3)
        axis /= np.linalg.norm(axis)
        origin_rotation = (
            rotation([0, 0, 1], rpy[2])
            @ rotation([0, 1, 0], rpy[1])
            @ rotation([1, 0, 0], rpy[0])
        )
        joints.append((kind, offset, origin_rotation, axis))
        xml.extend(
            [
                f'<link name="{name}"/>',
                (
                    f'<joint name="joint_{index:03}" type="{kind}">'
                    f'<parent link="{parent}"/><child link="{name}"/>'
                    f'<origin xyz="{" ".join(map(str, offset))}" rpy="{" ".join(map(str, rpy))}"/>'
                    f'<axis xyz="{" ".join(map(str, axis))}"/>'
                    '<limit lower="-2" upper="2" velocity="1" effort="1"/>'
                    "</joint>"
                ),
            ]
        )
        parent = name
    # Unused sensor frames exercise selective frame updates on a larger URDF.
    for index in range(extra_frames):
        xml.extend(
            [
                f'<link name="sensor_{index}"/>',
                (
                    f'<joint name="mount_{index}" type="fixed">'
                    f'<parent link="link_{index % (2 * dof):03}"/>'
                    f'<child link="sensor_{index}"/><origin xyz="1 2 3"/>'
                    "</joint>"
                ),
            ]
        )
    path = directory / "chain.urdf"
    path.write_text('<robot name="chain">' + "".join(xml) + "</robot>")
    owners = random.integers(0, 2 * dof + 1, sphere_count)
    owners[0], owners[-1] = 0, 2 * dof
    offsets = random.uniform(-0.15, 0.15, (sphere_count, 3))
    radii = random.uniform(0.005, 0.08, sphere_count)
    spheres = [
        hm.CollisionSphere(
            f"sphere_{i}",
            "base" if owner == 0 else f"link_{owner - 1:03}",
            offset,
            radius,
        )
        for i, (owner, offset, radius) in enumerate(zip(owners, offsets, radii))
    ]
    model = hm.SphereCollisionModel(str(path), spheres)

    def reference(positions):
        orientations = [np.eye(3)]
        translations = [np.zeros(3)]
        upstream = [[]]
        active = []
        configuration = []
        for kind, offset, origin_rotation, axis in joints:
            translation = translations[-1] + orientations[-1] @ offset
            orientation = orientations[-1] @ origin_rotation
            ancestors = upstream[-1].copy()
            if kind != "fixed":
                index = len(active)
                position = positions[index]
                active.append((kind, translation.copy(), orientation @ axis))
                ancestors.append(index)
                if kind == "prismatic":
                    translation = translation + orientation @ axis * position
                else:
                    orientation = orientation @ rotation(axis, position)
                configuration.extend(
                    [np.cos(position), np.sin(position)]
                    if kind == "continuous"
                    else [position]
                )
            orientations.append(orientation)
            translations.append(translation)
            upstream.append(ancestors)
        world = np.empty((sphere_count, 4))
        jacobians = np.zeros((sphere_count, 3, dof))
        for i, (owner, offset, radius) in enumerate(zip(owners, offsets, radii)):
            world[i, :3] = translations[owner] + orientations[owner] @ offset
            world[i, 3] = radius
            for index in upstream[owner]:
                kind, origin, axis = active[index]
                jacobians[i, :, index] = (
                    axis
                    if kind == "prismatic"
                    else np.cross(axis, world[i, :3] - origin)
                )
        return np.asarray(configuration), world, jacobians

    return model, reference


def check_queries(model, reference, positions):
    configuration, world, jacobians = reference(positions)
    np.testing.assert_allclose(model.world_spheres(configuration), world, atol=2e-14)
    pairs = [(pair.first, pair.second) for pair in model.collision_pairs]
    distances = np.array(
        [
            np.linalg.norm(world[second, :3] - world[first, :3])
            - world[first, 3]
            - world[second, 3]
            for first, second in pairs
        ]
    )
    index = int(np.argmin(distances))
    first, second = pairs[index]
    normal = world[second, :3] - world[first, :3]
    normal /= np.linalg.norm(normal)
    gradient = normal @ (jacobians[second] - jacobians[first])
    result = model.minimum_distance_with_gradient(configuration)
    assert result.distance_result.pair_index == index
    assert result.distance_result.distance == pytest.approx(distances[index], abs=2e-14)
    np.testing.assert_allclose(result.gradient, gradient, atol=2e-13)
    np.testing.assert_allclose(result.distance_result.normal, normal, atol=2e-13)
    assert (
        model.minimum_distance(configuration).distance
        == result.distance_result.distance
    )
    for margin in (0.0, 0.1):
        assert model.in_collision(configuration, margin) == bool(
            np.min(distances) <= margin
        )
    filtered = model.distances(configuration, 0.05)
    assert [item.pair_index for item in filtered] == np.flatnonzero(
        distances <= 0.05
    ).tolist()
    np.testing.assert_allclose(
        model.batch_minimum_distances([configuration, configuration]),
        [distances[index], distances[index]],
        atol=2e-14,
    )


@pytest.mark.parametrize("seed", [5, 27, 103])
def test_mixed_joint_sphere_frames_match_independent_chain(tmp_path, seed):
    model, reference = make_chain(tmp_path, seed)
    random = np.random.default_rng(seed + 1)
    for positions in random.uniform(-0.8, 0.8, (5, model.nv)):
        check_queries(model, reference, positions)
    links = list(dict.fromkeys(sphere.link_name for sphere in model.spheres))
    model.set_collision_groups({"a": links[::2], "b": links[1::2]}, [("a", "b")])
    check_queries(model, reference, np.zeros(model.nv))
    model.reset_collision_pairs()
    check_queries(model, reference, random.uniform(-0.8, 0.8, model.nv))
