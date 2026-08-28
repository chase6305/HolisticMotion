from pathlib import Path

import numpy as np
import pytest

import holistic_motion as hm


URDF = """<?xml version="1.0"?>
<robot name="sphere_test">
  <link name="base"/>
  <link name="left"/>
  <link name="right"/>
  <joint name="left_slide" type="prismatic">
    <parent link="base"/><child link="left"/>
    <origin xyz="-0.6 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="right_fixed" type="fixed">
    <parent link="base"/><child link="right"/>
    <origin xyz="0.6 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture()
def sphere_model(tmp_path: Path):
    urdf = tmp_path / "sphere.urdf"
    urdf.write_text(URDF)
    spheres = [
        hm.CollisionSphere("left_0", "left", [0.0, 0.0, 0.0], 0.25),
        hm.CollisionSphere("left_1", "left", [0.0, 0.3, 0.0], 0.1),
        hm.CollisionSphere("right_0", "right", [0.0, 0.0, 0.0], 0.25),
    ]
    return hm.SphereCollisionModel(str(urdf), spheres)


def test_sphere_collision_fk_distance_and_batch(sphere_model):
    assert sphere_model.nq == 1
    assert sphere_model.sphere_count == 3
    assert sphere_model.pair_count == 2

    world = sphere_model.world_spheres([0.0])
    np.testing.assert_allclose(world[:, 3], [0.25, 0.1, 0.25])
    np.testing.assert_allclose(world[0, :3], [-0.6, 0.0, 0.0])
    assert sphere_model.minimum_distance([0.0]).distance == pytest.approx(0.7)
    assert not sphere_model.in_collision([0.0])
    assert sphere_model.in_collision([0.71])

    distances = sphere_model.batch_minimum_distances([[0.0], [0.5], [0.8]])
    np.testing.assert_allclose(distances, [0.7, 0.2, -0.1], atol=1e-12)


def test_sphere_collision_groups_and_validation(sphere_model):
    count = sphere_model.set_collision_groups(
        {"left_arm": ["left"], "right_arm": ["right"]},
        [("left_arm", "right_arm")],
    )
    assert count == 2
    assert {pair.first_link for pair in sphere_model.collision_pairs} == {"left"}

    sphere_model.clear_collision_pairs()
    assert sphere_model.pair_count == 0
    with pytest.raises(RuntimeError):
        sphere_model.minimum_distance([0.0])
    with pytest.raises(ValueError):
        sphere_model.in_collision([0.0], security_margin=float("nan"))


def test_sphere_collision_integrates_with_sampling_planner(sphere_model):
    planner = hm.SamplingPlanner.from_sphere_collision_model(
        [-1.0], [1.0], sphere_model, security_margin=0.01
    )
    options = hm.PlanningOptions()
    options.timeout_seconds = 0.1
    result = planner.plan([0.4], [0.6], options)
    assert result.success
    assert result.message == "direct path found"
    np.testing.assert_allclose(result.path, [[0.4], [0.6]])


def test_sphere_collision_rejects_invalid_spheres(tmp_path: Path):
    urdf = tmp_path / "sphere.urdf"
    urdf.write_text(URDF)
    with pytest.raises(ValueError, match="positive"):
        hm.SphereCollisionModel(
            str(urdf), [hm.CollisionSphere("bad", "left", [0, 0, 0], 0.0)]
        )
