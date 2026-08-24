#!/usr/bin/env python3
"""Compatibility launcher for the organized dual-arm collision Gizmo."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "examples/python/visualization"),
)
from dual_arm_collision_gizmo import main  # noqa: E402


if __name__ == "__main__":
    main()
