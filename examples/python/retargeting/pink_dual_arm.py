#!/usr/bin/env python3
"""Solve a dual-arm pose task with the built-in Pink-style solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--profile", type=Path, default=repository / "examples/configs/marvin.json"
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/home/ubuntu/workspace/chase/HumanoidAssets"),
    )
    parser.add_argument("--urdf")
    parser.add_argument("--left-frame")
    parser.add_argument("--right-frame")
    parser.add_argument("--head-frame")
    parser.add_argument("--left-joints", help="comma-separated names")
    parser.add_argument("--right-joints", help="comma-separated names")
    parser.add_argument("--offset", type=float, default=0.02)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    retargeting = profile["retargeting"]
    frames = retargeting["frames"].copy()
    groups = {
        name: list(joints) for name, joints in retargeting["joint_groups"].items()
    }
    if args.left_frame:
        frames["left_hand"] = args.left_frame
    if args.right_frame:
        frames["right_hand"] = args.right_frame
    if args.head_frame:
        frames["head"] = args.head_frame
    if args.left_joints:
        groups["left_arm"] = _names(args.left_joints)
    if args.right_joints:
        groups["right_arm"] = _names(args.right_joints)
    urdf = Path(args.urdf) if args.urdf else args.asset_root / profile["urdf"]

    solver = PinkRetargetingSolver(
        urdf,
        frames=frames,
        joint_groups=groups,
        frame_tasks={
            "left_hand": FrameTask(position_cost=1.0, orientation_cost=0.2),
            "right_hand": FrameTask(position_cost=1.0, orientation_cost=0.2),
        },
        posture_task=PostureTask(cost=1e-3),
        tolerance=2e-3,
        max_iterations=150,
    )
    q0 = np.asarray(solver.pin.neutral(solver.model))
    solver.pin.forwardKinematics(solver.model, solver.data, q0)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    targets = {
        name: np.asarray(solver.data.oMf[frame_id].homogeneous).copy()
        for name, frame_id in solver._frame_ids.items()
    }
    targets["left_hand"][2, 3] += args.offset
    targets["right_hand"][2, 3] += args.offset
    solver.set_mode("dual_arm")
    result = solver.solve(targets)
    print(
        f"success={result.success} iterations={result.iterations} "
        f"residual={result.residual:.6g} solve_ms={result.solve_ms:.3f}"
    )
    print("configuration:", result.configuration)


if __name__ == "__main__":
    main()
