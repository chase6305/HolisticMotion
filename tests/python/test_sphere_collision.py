from pathlib import Path

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


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

REVOLUTE_URDF = """<?xml version="1.0"?>
<robot name="sphere_gradient_test">
  <link name="base"/>
  <link name="arm"/>
  <link name="target"/>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="target_fixed" type="fixed">
    <parent link="base"/><child link="target"/><origin xyz="1.5 0 0"/>
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


def test_sphere_minimum_distance_gradient_matches_finite_difference(sphere_model):
    configuration = np.array([0.1])
    step = 1e-6
    analytic = sphere_model.minimum_distance_gradient(configuration)
    combined = sphere_model.minimum_distance_with_gradient(configuration)
    positive = sphere_model.minimum_distance(configuration + step).distance
    negative = sphere_model.minimum_distance(configuration - step).distance
    finite_difference = np.array([(positive - negative) / (2.0 * step)])

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(combined.gradient, analytic)
    assert combined.distance_result.distance == pytest.approx(
        sphere_model.minimum_distance(configuration).distance
    )


def test_sphere_offset_point_gradient_matches_finite_difference(tmp_path):
    urdf = tmp_path / "revolute.urdf"
    urdf.write_text(REVOLUTE_URDF)
    model = hm.SphereCollisionModel(
        str(urdf),
        [
            hm.CollisionSphere("arm", "arm", [1.0, 0.0, 0.0], 0.1),
            hm.CollisionSphere("target", "target", [0.0, 0.0, 0.0], 0.1),
        ],
    )
    configuration = np.array([0.4])
    step = 1e-6
    analytic = model.minimum_distance_gradient(configuration)
    positive = model.minimum_distance(configuration + step).distance
    negative = model.minimum_distance(configuration - step).distance
    finite_difference = np.array([(positive - negative) / (2.0 * step)])

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-8)


def test_sphere_gradient_supports_continuous_joint_tangent(tmp_path):
    urdf = tmp_path / "continuous.urdf"
    continuous_urdf = REVOLUTE_URDF.replace(
        'type="revolute"', 'type="continuous"'
    ).replace(' lower="-2" upper="2"', "")
    urdf.write_text(continuous_urdf)
    model = hm.SphereCollisionModel(
        str(urdf),
        [
            hm.CollisionSphere("arm", "arm", [1.0, 0.0, 0.0], 0.1),
            hm.CollisionSphere("target", "target", [0.0, 0.0, 0.0], 0.1),
        ],
    )
    angle = 0.4
    step = 1e-6

    def configuration(value):
        return np.array([np.cos(value), np.sin(value)])

    analytic = model.minimum_distance_gradient(configuration(angle))
    positive = model.minimum_distance(configuration(angle + step)).distance
    negative = model.minimum_distance(configuration(angle - step)).distance

    assert model.nq == 2
    assert model.nv == 1
    assert analytic.shape == (model.nv,)
    np.testing.assert_allclose(
        analytic, [(positive - negative) / (2.0 * step)], rtol=1e-6, atol=1e-8
    )


def test_sphere_collision_groups_and_validation(sphere_model):
    initial_revision = sphere_model.pair_revision
    count = sphere_model.set_collision_groups(
        {"left_arm": ["left"], "right_arm": ["right"]},
        [("left_arm", "right_arm")],
    )
    assert count == 2
    assert sphere_model.pair_revision > initial_revision
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


def test_sphere_collision_integrates_with_path_optimizer(sphere_model):
    optimizer = hm.PathOptimizer.from_sphere_collision_model(
        [-1.0], [1.0], sphere_model, security_margin=0.01, clearance=0.2
    )
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 1.0
    result = optimizer.optimize([[-0.5], [0.0], [0.4]], options)

    assert result.success
    np.testing.assert_allclose(result.path[0], [-0.5])
    np.testing.assert_allclose(result.path[-1], [0.4])
    assert all(
        not sphere_model.in_collision(q, security_margin=0.01) for q in result.path
    )
    assert result.statistics.state_cost_evaluations > 0
    assert result.statistics.state_cost_gradient_evaluations > 0


def test_sphere_path_optimizer_handles_empty_pair_set(sphere_model):
    clearance_optimizer = hm.PathOptimizer.from_sphere_collision_model(
        [-1.0], [1.0], sphere_model, clearance=0.1
    )
    sphere_model.clear_collision_pairs()
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 1.0
    assert clearance_optimizer.optimize([[-0.5], [0.0], [0.5]], options).success
    sphere_model.reset_collision_pairs()
    assert clearance_optimizer.optimize([[-0.5], [0.0], [0.5]], options).success
    sphere_model.clear_collision_pairs()

    optimizer = hm.PathOptimizer.from_sphere_collision_model(
        [-1.0], [1.0], sphere_model
    )
    assert optimizer.optimize([[-0.5], [0.0], [0.5]]).success

    with pytest.raises(ValueError, match="active sphere collision pairs"):
        hm.PathOptimizer.from_sphere_collision_model(
            [-1.0], [1.0], sphere_model, clearance=0.1
        )


def test_sphere_collision_rejects_invalid_spheres(tmp_path: Path):
    urdf = tmp_path / "sphere.urdf"
    urdf.write_text(URDF)
    with pytest.raises(ValueError, match="positive"):
        hm.SphereCollisionModel(
            str(urdf), [hm.CollisionSphere("bad", "left", [0, 0, 0], 0.0)]
        )
