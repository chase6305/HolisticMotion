"""The named planning pipeline preserves topology and full-model context."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import holistic_motion as hm
import numpy as np
import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/python/planning/named_plan_and_retime.py"
)
BENCHMARK = (
    Path(__file__).resolve().parents[2] / "benchmarks/motion_pipeline_benchmark.py"
)


@pytest.fixture
def branched_model(tmp_path):
    if not hasattr(hm, "CollisionModel"):
        pytest.skip("requires native collision support")
    urdf = tmp_path / "branches.urdf"
    urdf.write_text("""<robot name="named_motion">
      <link name="world"/><link name="base"/><link name="turn_link"/>
      <link name="tool"><collision><origin xyz="0.4 0 0"/>
        <geometry><sphere radius="0.1"/></geometry></collision></link>
      <link name="other"><collision><origin xyz="4 0 0"/>
        <geometry><sphere radius="0.1"/></geometry></collision></link>
      <joint name="floating_base" type="floating">
        <parent link="world"/><child link="base"/></joint>
      <joint name="turn" type="continuous">
        <parent link="base"/><child link="turn_link"/><axis xyz="0 0 1"/>
        <limit effort="1" velocity="1"/></joint>
      <joint name="slide" type="prismatic">
        <parent link="turn_link"/><child link="tool"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="other_slide" type="prismatic">
        <parent link="world"/><child link="other"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    </robot>""")
    model = hm.CollisionModel(str(urdf))
    reference = model.configuration_from_joint_positions({"other_slide": 0.37})
    base = next(j for j in model.joint_coordinates if j.name == "floating_base")
    reference[base.idx_q : base.idx_q + 3] = [0.2, -0.3, 0.4]
    reference[base.idx_q + 5 : base.idx_q + 7] = [np.sin(0.2), np.cos(0.2)]
    return urdf, model, reference


@pytest.mark.parametrize("names", [("slide", "turn"), ("turn", "slide")])
@pytest.mark.parametrize("direction", [-1, 1])
def test_named_pipeline_keeps_floating_base_branch_and_short_periodic_arc(
    branched_model, tmp_path, names, direction
):
    urdf, model, reference = branched_model
    start = {"slide": 0.1, "turn": direction * 3.0}
    goal = {"slide": 0.3, "turn": -direction * 3.0}
    output = tmp_path / "trajectory.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--urdf",
            str(urdf),
            "--joint-names",
            *names,
            "--reference",
            *map(str, reference),
            "--start",
            *(str(start[name]) for name in names),
            "--goal",
            *(str(goal[name]) for name in names),
            "--max-velocity",
            "0.8",
            "0.8",
            "--max-acceleration",
            "1.5",
            "1.5",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["success"]
    assert summary["joint_names"] == list(names)
    assert summary["collision_pairs"] > 0
    assert summary["nq"] > summary["nv"] > summary["active_dof"]
    mapping = model.create_joint_position_map(list(names))
    changed = {i for j in mapping.joints for i in range(j.idx_q, j.idx_q + j.nq)}
    preserved = sorted(set(range(model.nq)) - changed)
    with np.load(output) as arrays:
        np.testing.assert_array_equal(arrays["joint_names"], names)
        np.testing.assert_array_equal(arrays["reference"], reference)
        configuration = arrays["configuration"]
        np.testing.assert_array_equal(
            configuration[:, preserved],
            np.broadcast_to(reference[preserved], (len(configuration), len(preserved))),
        )
        np.testing.assert_allclose(arrays["velocity"][[0, -1]], 0, atol=1e-12)
        assert np.all(np.diff(arrays["time"]) > 0)
        assert np.max(np.abs(arrays["velocity"])) <= 0.8 * (1 + 1e-8)
        assert np.max(np.abs(arrays["acceleration"])) <= 1.5 * (1 + 1e-8)
        turn = names.index("turn")
        assert abs(arrays["position"][-1, turn] - arrays["position"][0, turn]) < 0.3
        for row in (0, -1):
            expected = start if row == 0 else goal
            np.testing.assert_allclose(
                mapping.joint_positions(configuration[row]),
                [expected[name] for name in names],
                atol=1e-12,
            )
        for q in configuration:
            model.validate_configuration(q)
            assert not model.in_collision(q)


def test_polyline_timing_preserves_corners_and_stops_without_duplicate_times():
    sample = runpy.run_path(str(EXAMPLE))["sample_polyline"]
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    time, position, velocity, acceleration = sample(path, [0.8, 0.8], [1.5, 1.5], 33)
    assert len(time) == 65
    assert np.all(np.diff(time) > 0)
    np.testing.assert_allclose(position[[0, 32, -1]], path[[0, 1, -1]], atol=1e-12)
    np.testing.assert_allclose(velocity[[0, 32, -1]], 0, atol=1e-12)
    np.testing.assert_array_equal(position[:33, 1], 0)
    np.testing.assert_array_equal(position[33:, 0], 1)
    assert np.max(np.abs(acceleration)) <= 1.5 * (1 + 1e-8)


def test_stationary_polyline_returns_one_hold_sample():
    sample = runpy.run_path(str(EXAMPLE))["sample_polyline"]
    path = np.array([[0.2, -0.3], [0.2, -0.3]])
    time, position, velocity, acceleration = sample(path, [1, 1], [2, 2], 2)
    np.testing.assert_array_equal(time, [0])
    np.testing.assert_array_equal(position, path[:1])
    np.testing.assert_array_equal(velocity, [[0, 0]])
    np.testing.assert_array_equal(acceleration, [[0, 0]])


def test_polyline_rejects_unrepresentable_timestamps_between_different_edge_scales():
    sample = runpy.run_path(str(EXAMPLE))["sample_polyline"]
    # Both edges are representable, but the second duration is much smaller
    # than the floating-point spacing at the first edge's arrival time.
    path = np.array([[0.0, 0.0], [1e12, 0.0], [1e12, 1e-11]])
    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        sample(path, [1.0, 1.0], [1e24, 1e24], 2)


def test_invalid_named_request_does_not_write_trajectory(branched_model, tmp_path):
    urdf, _, _ = branched_model
    output = tmp_path / "invalid.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--urdf",
            str(urdf),
            "--joint-names",
            "slide",
            "--start",
            "2",
            "--goal",
            "0",
            "--max-velocity",
            "1",
            "--max-acceleration",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["stage"] == "planning"
    assert not output.exists()


@pytest.mark.parametrize(
    "start,goal,clear",
    [
        ([-0.8, 0], [0.8, 0], False),
        ([-0.8, 0.75], [0.8, 0.75], False),
        ([-0.8, 0.8], [0.8, 0.8], True),
        ([0, -0.9], [0, 0.9], False),
        ([0.3, -0.9], [0.3, 0.9], True),
        ([0, 0], [0, 0], False),
        ([0.8, 0], [0.8, 0], True),
    ],
)
def test_pipeline_benchmark_independently_checks_whole_edges(start, goal, clear):
    check = runpy.run_path(str(BENCHMARK))["edge_clears_wall"]
    assert check(np.asarray(start), np.asarray(goal)) is clear


def test_pipeline_benchmark_retains_failures_in_success_rate(monkeypatch):
    if not hasattr(hm, "SamplingPlanner"):
        pytest.skip("requires native planning")
    namespace = runpy.run_path(str(BENCHMARK))
    request = namespace["request"]
    monkeypatch.setitem(request.__globals__, "GOAL", np.zeros(2))
    row = request("RRT_CONNECT", "toppra", 7, 1.0)
    assert not row["success"]
    assert row["failed_stage"] == "planning"
    summary = namespace["summarize"]([row])
    assert summary["requests"] == 1
    assert summary["successes"] == 0
    assert summary["success_rate"] == 0
    assert summary["end_to_end_ms"] is None
