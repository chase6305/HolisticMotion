"""The documented torso-first example runs on a caller-provided URDF."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["left_arm", "right_arm", "dual_arm"])
def test_classic_torso_first_example(tmp_path, mode):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "upper_body.urdf"
    urdf.write_text("""<robot name="upper_body">
      <link name="base"/><link name="torso_link"/>
      <link name="left_tool"/><link name="right_tool"/>
      <joint name="waist" type="prismatic">
        <parent link="base"/><child link="torso_link"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="2" effort="1"/>
      </joint>
      <joint name="left" type="prismatic">
        <parent link="torso_link"/><child link="left_tool"/>
        <origin xyz="0 0.3 0"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" velocity="2" effort="1"/>
      </joint>
      <joint name="right" type="prismatic">
        <parent link="torso_link"/><child link="right_tool"/>
        <origin xyz="0 -0.3 0"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" velocity="2" effort="1"/>
      </joint>
    </robot>""")
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(repository / "examples/python/retargeting/classic_torso_first.py"),
        "--urdf",
        str(urdf),
        "--torso-frame",
        "torso_link",
        "--torso-joint",
        "waist",
        "--arm-mode",
        mode,
    ]
    if mode in ("left_arm", "dual_arm"):
        command.extend(("--left-frame", "left_tool", "--left-joint", "left"))
    if mode in ("right_arm", "dual_arm"):
        command.extend(("--right-frame", "right_tool", "--right-joint", "right"))
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment
    )
    assert "torso: success=True reason=converged" in result.stdout
    assert "arms: success=True reason=converged" in result.stdout
    assert "solve_ms=" in result.stdout
    assert "aligned joints: waist=" in result.stdout
    assert "manipulation joints: waist=" in result.stdout
    if mode in ("left_arm", "dual_arm"):
        assert "left=" in result.stdout
    if mode in ("right_arm", "dual_arm"):
        assert "right=" in result.stdout
    assert "manipulation configuration:" in result.stdout

    if mode == "dual_arm":
        customized = subprocess.run(
            [
                *command,
                "--joint-delta",
                "waist=-0.2",
                "--joint-delta",
                "left=0.15",
                "--joint-delta",
                "right=-0.1",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        summary = next(
            line
            for line in customized.stdout.splitlines()
            if line.startswith("manipulation joints:")
        )
        values = dict(
            item.split("=")
            for item in summary.removeprefix("manipulation joints:").strip().split(", ")
        )
        assert float(values["waist"]) == pytest.approx(-0.2, abs=2e-5)
        assert float(values["left"]) == pytest.approx(0.15, abs=2e-5)
        assert float(values["right"]) == pytest.approx(-0.1, abs=2e-5)

    if mode == "left_arm":
        invalid = subprocess.run(
            [*command, "--torso-delta", "2"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert invalid.returncode == 2
        assert "outside its URDF limits [-1, 1]" in invalid.stderr
        assert "Traceback" not in invalid.stderr

        unknown_command = command.copy()
        unknown_command[unknown_command.index("waist")] = "missing_joint"
        unknown = subprocess.run(
            unknown_command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert unknown.returncode == 2
        assert "unknown joints: ['missing_joint']" in unknown.stderr
        assert "Traceback" not in unknown.stderr

        bad_override = subprocess.run(
            [*command, "--joint-delta", "right=0.1"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert bad_override.returncode == 2
        assert "unconfigured joint 'right'" in bad_override.stderr
        assert "Traceback" not in bad_override.stderr


def test_classic_torso_first_help_describes_repeatable_groups():
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/python/retargeting/classic_torso_first.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "repeat to define the group" in result.stdout
    assert "arm group solved after torso alignment" in result.stdout
    assert "--joint-delta JOINT=DELTA" in result.stdout
