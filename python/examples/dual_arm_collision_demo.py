#!/usr/bin/env python3
"""Compatibility launcher for the organized dual-arm collision example."""

import runpy
from pathlib import Path

runpy.run_path(
    Path(__file__).resolve().parents[2]
    / "examples/python/collision/dual_arm_groups.py",
    run_name="__main__",
)
