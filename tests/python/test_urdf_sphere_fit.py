import numpy as np
import pytest
from holistic_motion.geometry import (
    SphereFitOptions,
    fit_urdf_collision_spheres,
    load_urdf_collision_meshes,
)

trimesh = pytest.importorskip("trimesh")


URDF = """<robot name="fit_test">
  <link name="box_link">
    <collision>
      <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </collision>
  </link>
  <link name="sphere_link">
    <collision>
      <origin xyz="-1 0 0"/>
      <geometry><sphere radius="0.15"/></geometry>
    </collision>
  </link>
</robot>"""


def test_load_urdf_collision_geometry_applies_link_transform(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF)
    meshes = load_urdf_collision_meshes(urdf)

    assert set(meshes) == {"box_link", "sphere_link"}
    np.testing.assert_allclose(meshes["box_link"].centroid, [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(meshes["box_link"].extents, [0.2, 0.4, 0.1], atol=1e-12)
    np.testing.assert_allclose(meshes["sphere_link"].centroid, [-1, 0, 0], atol=1e-12)


def test_fit_selected_urdf_links_is_reproducible(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF)
    options = SphereFitOptions(max_spheres=4, min_radius=0.01, sampled_coverage=True)
    first = fit_urdf_collision_spheres(
        urdf,
        options,
        links=["box_link"],
        pitch=0.05,
        surface_samples=100,
        random_seed=3,
    )
    second = fit_urdf_collision_spheres(
        urdf,
        options,
        links=["box_link"],
        pitch=0.05,
        surface_samples=100,
        random_seed=3,
    )
    assert first == second
    assert first["box_link"].metrics.sampled_coverage == pytest.approx(1.0)

    with pytest.raises(ValueError, match="no collision geometry"):
        fit_urdf_collision_spheres(urdf, links=["missing"])


def test_package_mesh_resolution_and_scale(tmp_path):
    package = tmp_path / "my_robot"
    mesh_dir = package / "meshes"
    mesh_dir.mkdir(parents=True)
    trimesh.creation.box(extents=[1, 1, 1]).export(mesh_dir / "box.stl")
    urdf = tmp_path / "mesh_robot.urdf"
    urdf.write_text(
        """<robot name="mesh"><link name="body"><collision><geometry>
        <mesh filename="package://my_robot/meshes/box.stl" scale="2 3 4"/>
        </geometry></collision></link></robot>"""
    )
    meshes = load_urdf_collision_meshes(urdf, package_dirs=[tmp_path])
    np.testing.assert_allclose(meshes["body"].extents, [2, 3, 4], atol=1e-6)


def test_urdf_vector_attributes_reject_trailing_garbage(tmp_path):
    urdf = tmp_path / "malformed.urdf"
    urdf.write_text(
        """<robot name="malformed"><link name="body"><collision>
        <origin xyz="1 2 3garbage"/><geometry><box size="1 1 1"/></geometry>
        </collision></link></robot>"""
    )

    with pytest.raises(ValueError, match="expected 3 finite values"):
        load_urdf_collision_meshes(urdf)


def test_urdf_loader_rejects_duplicate_links_and_package_traversal(tmp_path):
    duplicate = tmp_path / "duplicate.urdf"
    duplicate.write_text(
        """<robot name="duplicate">
        <link name="body"><collision><geometry><box size="1 1 1"/></geometry>
        </collision></link><link name="body"/></robot>"""
    )
    with pytest.raises(ValueError, match="duplicate link"):
        load_urdf_collision_meshes(duplicate)

    traversal = tmp_path / "traversal.urdf"
    traversal.write_text(
        """<robot name="traversal"><link name="body"><collision><geometry>
        <mesh filename="package://robot/../outside.stl"/>
        </geometry></collision></link></robot>"""
    )
    with pytest.raises(ValueError, match="invalid package mesh URI"):
        load_urdf_collision_meshes(traversal, package_dirs=[tmp_path])


def test_fit_selected_urdf_links_rejects_duplicates(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF)
    with pytest.raises(ValueError, match="unique"):
        fit_urdf_collision_spheres(urdf, links=["box_link", "box_link"])


def test_urdf_sphere_fit_rejects_ambiguous_string_iterables(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF)
    with pytest.raises(TypeError, match="not a string"):
        fit_urdf_collision_spheres(urdf, links="box_link")
    with pytest.raises(TypeError, match="not one path"):
        load_urdf_collision_meshes(urdf, package_dirs=str(tmp_path))
