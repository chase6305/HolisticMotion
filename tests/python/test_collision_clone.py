"""Collision workers snapshot geometry and own independent mutable workspaces."""

import gc
from concurrent.futures import ThreadPoolExecutor

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "CollisionModel"), reason="native collision support is disabled"
)


def _model(path, geometry):
    path.write_text(
        '<robot name="clone">'
        f'<link name="base"><collision>{geometry}</collision></link>'
        f'<link name="slider"><collision>{geometry}</collision></link>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="slider"/><axis xyz="1 0 0"/>'
        '<limit lower="0" upper="2" effort="1" velocity="1"/>'
        "</joint></robot>"
    )
    return hm.CollisionModel(str(path), exclude_adjacent=False)


def _sphere_model(path):
    return _model(path, '<geometry><sphere radius="0.2"/></geometry>')


def test_clone_preserves_obstacles_pairs_and_independent_edits(tmp_path):
    model = _sphere_model(tmp_path / "source.urdf")
    pose = np.eye(4)
    pose[0, 3] = 0.5
    model.add_box_obstacle("wall", [0.2, 0.5, 0.5], pose)
    model.set_collision_groups(
        {"robot": ["slider"], "world": ["wall"]}, [("robot", "world")]
    )
    # Warm mutable query caches before copying: a clone starts fresh and must
    # still preserve signed-distance and nearest-point request settings.
    expected = model.minimum_distance([1.0])
    assert model.is_within_distance([1.0], 0.3)
    clone = model.clone()
    assert clone.nq == model.nq == 1 and clone.nv == model.nv == 1
    assert clone.geometry_names == model.geometry_names
    assert clone.geometry_count == model.geometry_count == 3
    assert clone.pair_count == model.pair_count == 1
    actual = clone.minimum_distance([1.0])
    assert actual.distance == pytest.approx(expected.distance)
    np.testing.assert_allclose(actual.nearest_point_first, expected.nearest_point_first)
    np.testing.assert_allclose(
        actual.nearest_point_second, expected.nearest_point_second
    )
    assert not clone.is_within_distance([1.0], 0.1)
    assert clone.in_collision([0.5])
    assert clone.remove_obstacle("wall")
    assert clone.geometry_count == 2 and clone.pair_count == 0
    assert model.geometry_count == 3 and model.pair_count == 1
    model.clear_collision_pairs()
    assert clone.reset_collision_pairs(False) == 1
    assert model.pair_count == 0


@pytest.mark.parametrize("mesh", [False, True])
def test_clone_uses_memory_snapshot_after_sources_and_owner_disappear(tmp_path, mesh):
    path = tmp_path / "source.urdf"
    mesh_path = tmp_path / "tetra.obj"
    if mesh:
        mesh_path.write_text(
            "v 0 0 0\nv 0.2 0 0\nv 0 0.2 0\nv 0 0 0.2\n"
            "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        )
        model = _model(path, f'<geometry><mesh filename="{mesh_path}"/></geometry>')
    else:
        model = _sphere_model(path)
    expected = model.minimum_distance([1.0]).distance
    path.unlink()
    if mesh:
        mesh_path.unlink()
    clone = model.clone()
    del model
    gc.collect()
    assert clone.minimum_distance([1.0]).distance == pytest.approx(expected)
    next_clone = clone.clone()
    clone.clear_collision_pairs()
    assert next_clone.pair_count == 1
    assert next_clone.minimum_distance([1.0]).distance == pytest.approx(expected)


def test_cloned_collision_planners_run_on_separate_workers(tmp_path):
    original = _sphere_model(tmp_path / "source.urdf")
    models = [original.clone() for _ in range(4)]
    original.clear_collision_pairs()
    del original
    planners = [
        hm.SamplingPlanner.from_collision_joints(model, ["slide"], [0.8])
        for model in models
    ]
    options = hm.PlanningOptions()
    options.edge_resolution = 0.01

    def query(worker):
        for sample in range(16):
            start = 0.6 + sample * 0.001
            result = planners[worker].plan([start], [1.2], options, release_gil=True)
            assert result.success
            np.testing.assert_allclose(result.path[0], [start])
            np.testing.assert_allclose(result.path[-1], [1.2])
        models[worker].clear_collision_pairs()
        return result

    with ThreadPoolExecutor(4) as workers:
        results = list(workers.map(query, range(4)))
    assert len(results) == 4 and all(model.pair_count == 0 for model in models)


def test_clone_preserves_floating_joint_manifold_and_warmed_workspaces(tmp_path):
    source = tmp_path / "floating.urdf"
    geometry = '<collision><geometry><sphere radius="0.1"/></geometry></collision>'
    source.write_text(
        f'<robot name="floating"><link name="base">{geometry}</link>'
        '<link name="body"/>'
        f'<link name="tool">{geometry}</link>'
        '<joint name="floating" type="floating"><parent link="base"/>'
        '<child link="body"/></joint>'
        '<joint name="slide" type="prismatic"><parent link="body"/>'
        '<child link="tool"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/>'
        '<limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    model = hm.CollisionModel(str(source), exclude_adjacent=False)
    assert model.nq == 8 and model.nv == 7
    states = []
    random = np.random.default_rng(873)
    for _ in range(24):
        state = model.neutral_configuration()
        state[:3] = random.uniform(-0.2, 0.2, 3)
        quaternion = random.normal(size=4)
        state[3:7] = quaternion / np.linalg.norm(quaternion)
        state[7] = random.uniform(0.1, 0.5)
        states.append(state)
    expected = [model.minimum_distance(state).distance for state in states]
    source.unlink()
    clone = model.clone()
    for state, distance in zip(reversed(states), reversed(expected)):
        assert clone.minimum_distance(state).distance == pytest.approx(
            distance, abs=1e-8
        )
        model.minimum_distance(states[0])
    mapping = clone.create_joint_position_map(["slide"])
    full = mapping.configuration_with_joint_positions(states[0], [0.3])
    np.testing.assert_array_equal(full[:7], states[0][:7])
    assert mapping.joint_positions(full) == pytest.approx([0.3])
