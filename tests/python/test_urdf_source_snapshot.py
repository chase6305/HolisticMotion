"""Kinematic and collision geometry construction consume one source snapshot."""

import os
import subprocess
import sys

import holistic_motion as hm
import pytest


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(hm, "CollisionModel"),
    reason="requires a POSIX pipe and native collision support",
)
def test_collision_model_consumes_a_single_urdf_source(tmp_path):
    source = tmp_path / "source.urdf"
    os.mkfifo(source)
    geometry = '<collision><geometry><sphere radius="0.1"/></geometry></collision>'
    xml = (
        f'<robot name="single_source"><link name="base">{geometry}</link>'
        f'<link name="tip">{geometry}</link>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tip"/><origin xyz="0.5 0 0"/><axis xyz="1.5 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>'
    )
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            str(source),
            xml,
        ]
    )
    try:
        reader = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys,holistic_motion as hm; "
                "m=hm.CollisionModel(sys.argv[1], exclude_adjacent=False); "
                "assert abs(m.minimum_distance([0.2]).distance-0.5)<1e-12",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert reader.returncode == 0, reader.stdout + reader.stderr
    finally:
        if writer.poll() is None:
            writer.terminate()
        writer.wait(timeout=5)
