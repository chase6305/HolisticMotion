"""Repeated URDF primitives retain independent link-local geometry."""

import numpy as np
import pytest
from holistic_motion.geometry import urdf_sphere_fit as fitting

trimesh = pytest.importorskip("trimesh")

PRIMITIVES = (
    ("box", '<box size="1 2 3"/>', '<box size="2 4 6"/>'),
    (
        "cylinder",
        '<cylinder radius="1" length="3"/>',
        '<cylinder radius="2" length="6"/>',
    ),
    ("icosphere", '<sphere radius="1"/>', '<sphere radius="2"/>'),
)
PRIMITIVE_ARGUMENTS = {
    "box": {"extents": [1, 2, 3]},
    "cylinder": {"radius": 1, "height": 3},
    "icosphere": {"radius": 1, "subdivisions": 2},
}


def _write_urdf(path, declarations, *, single_link=False):
    collisions = [
        f'<collision><origin xyz="{10 * index} 0 0" '
        f'rpy="0 0 {index * np.pi / 2}"/><geometry>{shape}</geometry></collision>'
        for index, shape in enumerate(declarations)
    ]
    links = (
        '<link name="body">' + "".join(collisions) + "</link>"
        if single_link
        else "".join(
            f'<link name="link{index}">{collision}</link>'
            for index, collision in enumerate(collisions)
        )
    )
    path.write_text(f'<robot name="primitives">{links}</robot>')


@pytest.mark.parametrize("constructor,shape,larger", PRIMITIVES)
@pytest.mark.parametrize("single_link", [False, True])
def test_repeated_primitives_generate_once_and_keep_independent_origins(
    tmp_path, monkeypatch, constructor, shape, larger, single_link
):
    path = tmp_path / "robot.urdf"
    _write_urdf(path, [shape, shape, shape, larger], single_link=single_link)
    original = getattr(trimesh.creation, constructor)
    created = []

    def create(*args, **kwargs):
        mesh = original(*args, **kwargs)
        created.append(mesh)
        return mesh

    monkeypatch.setattr(trimesh.creation, constructor, create)
    meshes = fitting.load_urdf_collision_meshes(path)
    assert len(created) == 2
    base = original(**PRIMITIVE_ARGUMENTS[constructor])
    expected = []
    for index in range(4):
        mesh = base.copy()
        if index == 3:
            mesh.apply_scale(2)
        angle = index * np.pi / 2
        transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        transform[:3, 3] = [10 * index, 0, 0]
        mesh.apply_transform(transform)
        expected.append(mesh)
    if single_link:
        expected_mesh = trimesh.util.concatenate(expected)
        np.testing.assert_array_equal(meshes["body"].faces, expected_mesh.faces)
        np.testing.assert_allclose(
            meshes["body"].vertices, expected_mesh.vertices, rtol=0, atol=1e-14
        )
    else:
        for index, mesh in enumerate(expected):
            np.testing.assert_array_equal(meshes[f"link{index}"].faces, mesh.faces)
            np.testing.assert_allclose(
                meshes[f"link{index}"].vertices, mesh.vertices, rtol=0, atol=1e-14
            )
        before = meshes["link1"].vertices.copy()
        meshes["link0"].apply_scale(10)
        np.testing.assert_array_equal(meshes["link1"].vertices, before)


def test_primitive_reuse_ignores_attribute_order_and_expires_after_each_call(
    tmp_path, monkeypatch
):
    path = tmp_path / "robot.urdf"
    _write_urdf(
        path,
        [
            '<cylinder radius="1" length="3"/>',
            '<cylinder length="3" radius="1"/>',
        ],
    )
    original = trimesh.creation.cylinder
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(trimesh.creation, "cylinder", create)
    first = fitting.load_urdf_collision_meshes(path)
    assert len(calls) == 1
    first["link0"].vertices[:] = 0.0
    second = fitting.load_urdf_collision_meshes(path)
    assert len(calls) == 2
    np.testing.assert_allclose(second["link0"].extents, [2, 2, 3])


def test_primitive_reuse_only_copies_requested_nonfinal_instances(
    tmp_path, monkeypatch
):
    path = tmp_path / "robot.urdf"
    _write_urdf(path, ['<box size="1 2 3"/>'] * 4)
    original = trimesh.Trimesh.copy
    copies = []

    def copy(mesh, *args, **kwargs):
        copies.append(mesh)
        return original(mesh, *args, **kwargs)

    monkeypatch.setattr(trimesh.Trimesh, "copy", copy)
    # Ignore the two unrequested declarations when tracking remaining uses.
    meshes = fitting._load_urdf_collision_meshes(
        path, package_dirs=(), selected_links=frozenset(["link0", "link2"])
    )
    assert len(copies) == 1
    assert set(meshes) == {"link0", "link2"}
    np.testing.assert_allclose(meshes["link0"].centroid, [0, 0, 0], atol=1e-14)
    np.testing.assert_allclose(meshes["link2"].centroid, [20, 0, 0], atol=1e-14)


@pytest.mark.parametrize("constructor,shape,larger", PRIMITIVES)
def test_unique_primitives_do_not_copy_meshes(
    tmp_path, monkeypatch, constructor, shape, larger
):
    path = tmp_path / "robot.urdf"
    _write_urdf(path, [shape, larger])
    original = getattr(trimesh.creation, constructor)
    sources = [original(**PRIMITIVE_ARGUMENTS[constructor]) for _ in range(2)]
    sources[1].apply_scale(2)
    pending = iter(sources)

    def create(*_args, **_kwargs):
        return next(pending)

    monkeypatch.setattr(trimesh.creation, constructor, create)
    meshes = fitting.load_urdf_collision_meshes(path)
    assert meshes["link0"] is sources[0]
    assert meshes["link1"] is sources[1]
