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
    np.testing.assert_allclose(
        meshes["box_link"].extents, [0.2, 0.4, 0.1], atol=1e-12
    )
    np.testing.assert_allclose(
        meshes["sphere_link"].centroid, [-1, 0, 0], atol=1e-12
    )


def test_fit_selected_urdf_links_is_reproducible(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF)
    options = SphereFitOptions(
        max_spheres=4, min_radius=0.01, sampled_coverage=True
    )
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
