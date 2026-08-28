#!/usr/bin/env python3
"""Compare HolisticMotion's dependency-free RRT variants in a 2-D scene."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _import_holistic_motion():
    try:
        import holistic_motion

        return holistic_motion
    except ModuleNotFoundError as error:
        repository = Path(__file__).resolve().parents[3]
        runner = repository / "scripts/run.sh"
        if error.name != "holistic_motion" or os.environ.get("HMOTION_REEXEC"):
            raise
        os.environ["HMOTION_REEXEC"] = "1"
        os.execv(str(runner), [str(runner), sys.executable, __file__, *sys.argv[1:]])
        raise RuntimeError from error


hm = _import_holistic_motion()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    def valid(q):
        return not (-0.2 < q[0] < 0.2 and -0.75 < q[1] < 0.75)

    planner = hm.SamplingPlanner([-1.0, -1.0], [1.0, 1.0], valid)
    for name in ("RRT_CONNECT", "RRT_STAR", "INFORMED_RRT_STAR"):
        options = hm.PlanningOptions()
        options.algorithm = getattr(hm.SamplingAlgorithm, name)
        options.timeout_seconds = args.timeout
        options.extension_range = 0.15
        options.edge_resolution = 0.02
        options.random_seed = args.seed
        result = planner.plan([-0.8, 0.0], [0.8, 0.0], options)
        stats = result.statistics
        print(
            f"{name:18s} success={result.success!s:5s} "
            f"time={stats.planning_time_ms:7.2f} ms nodes={stats.tree_nodes:5d} "
            f"checks={stats.collision_checks:6d} waypoints={len(result.path):3d} "
            f"length={stats.final_path_length:.4f}"
        )


if __name__ == "__main__":
    main()
