"""Selected-link fitting avoids loading unrelated collision geometry."""

import sys
from types import SimpleNamespace

import pytest
from holistic_motion.geometry import urdf_sphere_fit as fitting


@pytest.fixture
def mesh_adapter(monkeypatch):
    created = []

    class Mesh:
        def apply_transform(self, transform):
            assert transform.shape == (4, 4)

    class Scene:
        pass

    def box(*, extents):
        created.append(tuple(extents))
        return Mesh()

    adapter = SimpleNamespace(
        Trimesh=Mesh,
        Scene=Scene,
        creation=SimpleNamespace(box=box),
        util=SimpleNamespace(concatenate=lambda meshes: meshes[0]),
    )
    monkeypatch.setitem(sys.modules, "trimesh", adapter)
    monkeypatch.setattr(fitting, "fit_trimesh", lambda mesh, *_args, **_kwargs: mesh)
    return created


def _urdf(tmp_path, extra):
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="selected"><link name="wanted"><collision><geometry>'
        '<box size="1 2 3"/></geometry></collision></link>' + extra + "</robot>"
    )
    return path


def test_selected_fit_does_not_resolve_unrequested_mesh(tmp_path, mesh_adapter):
    path = _urdf(
        tmp_path,
        '<link name="other"><collision><geometry><mesh filename="missing.stl"/>'
        "</geometry></collision></link>",
    )
    assert list(fitting.fit_urdf_collision_spheres(path, links=["wanted"])) == [
        "wanted"
    ]
    assert mesh_adapter == [(1.0, 2.0, 3.0)]
    # The public all-link loader retains its complete geometry validation.
    with pytest.raises(FileNotFoundError):
        fitting.load_urdf_collision_meshes(path)


def test_selected_fit_constructs_only_requested_meshes_in_requested_order(
    tmp_path, mesh_adapter
):
    path = _urdf(
        tmp_path,
        '<link name="other"><collision><geometry><box size="4 5 6"/>'
        '</geometry></collision></link><link name="last"><collision><geometry>'
        '<box size="7 8 9"/></geometry></collision></link>',
    )
    selected = (name for name in ("last", "wanted"))
    assert list(fitting.fit_urdf_collision_spheres(path, links=selected)) == [
        "last",
        "wanted",
    ]
    assert mesh_adapter == [(1.0, 2.0, 3.0), (7.0, 8.0, 9.0)]


@pytest.mark.parametrize("links", [[], ["wanted", "wanted"], [["wanted"]], [1], [""]])
def test_invalid_selection_is_rejected_before_geometry_loading(
    tmp_path, mesh_adapter, links
):
    path = _urdf(tmp_path, "")
    with pytest.raises(ValueError):
        fitting.fit_urdf_collision_spheres(path, links=links)
    assert mesh_adapter == []


def test_unknown_selection_does_not_load_other_meshes(tmp_path, mesh_adapter):
    path = _urdf(tmp_path, "")
    with pytest.raises(ValueError, match="no collision geometry: missing"):
        fitting.fit_urdf_collision_spheres(path, links=["missing"])
    assert mesh_adapter == []


def test_duplicate_unselected_link_names_are_still_rejected(tmp_path, mesh_adapter):
    path = _urdf(tmp_path, '<link name="other"/><link name="other"/>')
    with pytest.raises(ValueError, match="duplicate link name"):
        fitting.fit_urdf_collision_spheres(path, links=["wanted"])
