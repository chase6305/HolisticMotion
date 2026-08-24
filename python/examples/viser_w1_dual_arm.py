#!/usr/bin/env python3
"""Launch the dual-arm SRS gizmo demo with the complete W1 asset."""

from __future__ import annotations

import sys
from pathlib import Path

from viser_dual_arm_gizmo import main


DEFAULT_URDF = Path(
    "/home/ubuntu/workspace/chase/HumanoidAssets/"
    "Dexforce_W1/robot_with_ee.urdf"
)


if __name__ == "__main__":
    if "--urdf" not in sys.argv:
        sys.argv.extend(("--urdf", str(DEFAULT_URDF)))
    main()
