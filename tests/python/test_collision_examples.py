"""Collision CLI examples preserve manifold coordinates at their boundary."""

import os
import subprocess
import sys
from pathlib import Path

import holistic_motion as hm
import numpy as np
import pytest


@pytest.fixture
def collision_example(tmp_path):
    if not hasattr(hm, "CollisionModel"):
        pytest.skip("native collision support is disabled")
    urdf = tmp_path / "cli.urdf"
    urdf.write_text("""<robot name="cli">
      <link name="world"/>
      <link name="base"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
      <link name="arm"/>
      <link name="tool"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
      <joint name="floating" type="floating"><parent link="world"/><child link="base"/></joint>
      <joint name="turn" type="continuous"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 1"/><limit effort="1" velocity="1"/></joint>
      <joint name="slide" type="prismatic"><parent link="arm"/><child link="tool"/>
        <origin xyz="1 0 0"/><axis xyz="1 0 0"/>
        <limit lower="-0.5" upper="2" effort="1" velocity="1"/></joint>
    </robot>""")
    native = (
        sys.modules.get("holistic_motion._holistic_motion")
        or sys.modules["_holistic_motion"]
    )
    native_path = Path(native.__file__).parent
    if native_path.name == "holistic_motion":
        native_path = native_path.parent
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "0"
    environment["HOLISTICMOTION_PYTHON_PATH"] = str(native_path)
    examples = Path(__file__).resolve().parents[2] / "examples" / "python" / "collision"

    def run(name, *arguments):
        return subprocess.run(
            [sys.executable, str(examples / name), "--urdf", str(urdf), *arguments],
            env=environment,
            capture_output=True,
            text=True,
        )

    return run, hm.CollisionModel(str(urdf))


@pytest.mark.parametrize("kind", ["basic", "groups"])
def test_query_examples_default_to_neutral_continuous_and_floating_blocks(
    collision_example, kind
):
    run, _ = collision_example
    if kind == "basic":
        result = run("basic_query.py")
        assert "in collision: False" in result.stdout
    else:
        result = run(
            "dual_arm_groups.py", "--left-links", "base", "--right-links", "tool"
        )
        assert "arms in collision: False" in result.stdout
    assert result.returncode == 0, result.stderr


def test_named_path_scan_interpolates_scalar_positions(collision_example):
    run, model = collision_example
    context = model.neutral_configuration()
    context[:3] = [0.3, 0.4, 0.5]
    context_arg = ",".join(map(str, context))
    result = run(
        "path_scan.py",
        "--joint-names",
        "slide",
        "turn",
        "--q-start=0.2,3.0",
        "--q-end=0.4,3.3",
        "--context=" + context_arg,
        "--samples",
        "5",
    )
    assert result.returncode == 0, result.stderr
    assert "minimum clearance: 1.000000 m at alpha=0.0000" in result.stdout
    assert "path collision-free at sampled configurations" in result.stdout
    # Full q endpoints are valid, but their straight component interpolation
    # would leave the continuous unit circle and must not be accepted.
    endpoints = [
        model.configuration_with_joint_positions(context, ["slide", "turn"], position)
        for position in ([0.2, 3.0], [0.4, 3.3])
    ]
    result = run(
        "path_scan.py",
        "--q-start=" + ",".join(map(str, endpoints[0])),
        "--q-end=" + ",".join(map(str, endpoints[1])),
    )
    assert result.returncode == 2
    assert "non-Euclidean configurations require --joint-names" in result.stderr
    invalid = np.zeros(model.nq)
    result = run("basic_query.py", "-q", *map(str, invalid))
    assert result.returncode == 2
    assert "normalized continuous pairs/quaternions" in result.stderr
