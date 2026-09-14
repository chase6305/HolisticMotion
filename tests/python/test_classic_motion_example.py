"""The documented asset-free planning pipeline must remain executable."""

import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("profile", ["double_s", "trapezoidal"])
def test_classic_plan_and_retime_example(profile):
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/python/planning/classic_plan_and_retime.py"),
            "--profile",
            profile,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = re.search(r"path: (\d+) points/.+ -> (\d+) points/", result.stdout)
    assert counts is not None
    assert int(counts.group(2)) < int(counts.group(1))
    assert f"timing: {profile}" in result.stdout
