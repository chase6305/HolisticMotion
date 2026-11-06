"""Plan with NVIDIA cuRobo v2 using caller-provided local robot assets."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from holistic_motion.kit.curobo_planner import CuroboPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", required=True, type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--joint-names", nargs="+")
    parser.add_argument("--start", required=True, nargs="+", type=float)
    parser.add_argument("--goal", required=True, nargs="+", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, help="Save owned trajectory arrays to NPZ"
    )
    args = parser.parse_args()

    with CuroboPlanner(
        args.robot_config,
        scene=args.scene,
        joint_names=args.joint_names,
        device=args.device,
    ) as planner:
        planner.warmup(args.warmup_iterations)
        started = perf_counter()
        result = planner.plan(args.start, args.goal, max_attempts=args.attempts)
        summary = {
            "success": result.success,
            "message": result.message,
            "wall_time_ms": (perf_counter() - started) * 1000,
        }
        if result.success:
            trajectory = result.trajectory
            summary.update(
                joint_names=trajectory.joint_names,
                samples=trajectory.time.size,
                duration=trajectory.duration,
            )
            if args.output is not None:
                arrays = {
                    name: getattr(trajectory, name)
                    for name in ("time", "position", "velocity", "acceleration", "jerk")
                    if getattr(trajectory, name) is not None
                }
                np.savez(args.output, joint_names=trajectory.joint_names, **arrays)
        print(json.dumps(summary))
        return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
