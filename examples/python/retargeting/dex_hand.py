#!/usr/bin/env python3
"""Retarget one keypoint frame using an explicitly supplied hand URDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from holistic_motion.kit.retargeting import DexHandRetargetingSolver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--joint", action="append", required=True, help="repeat per optimized joint"
    )
    parser.add_argument(
        "--vector",
        action="append",
        nargs=4,
        required=True,
        metavar=("ORIGIN_LINK", "TASK_LINK", "ORIGIN_INDEX", "TASK_INDEX"),
        help="repeat per robot/keypoint vector correspondence",
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        required=True,
        help="(N, 3) .npy array in hand base axes",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        help="JSON object with all independent hand joint positions",
    )
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=4e-3)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--objective-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-evaluations", type=int, default=100)
    args = parser.parse_args()
    try:
        pairs = [[int(row[2]), int(row[3])] for row in args.vector]
    except ValueError:
        parser.error("vector keypoint indices must be integers")
    solver = DexHandRetargetingSolver(
        args.urdf,
        joint_names=args.joint,
        origin_links=[row[0] for row in args.vector],
        task_links=[row[1] for row in args.vector],
        keypoint_pairs=pairs,
        scaling=args.scaling,
        temporal_weight=args.temporal_weight,
        vector_tolerance=args.tolerance,
        objective_tolerance=args.objective_tolerance,
        max_evaluations=args.max_evaluations,
    )
    seed = json.loads(args.seed.read_text(encoding="utf-8")) if args.seed else None
    result = solver.solve(np.load(args.keypoints, allow_pickle=False), seed=seed)
    print(
        json.dumps(
            {
                "success": result.success,
                "termination_reason": result.termination_reason,
                "joint_positions": dict(result.joint_positions),
                "residual": result.residual,
                "objective": result.objective,
                "evaluations": result.evaluations,
                "optimizer_status": result.optimizer_status,
                "solve_ms": result.solve_ms,
                "message": result.message,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
