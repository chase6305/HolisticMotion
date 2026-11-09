"""Measure validated plan/optimize/retime/sample requests around a fixed wall.

Model-free setup and warmup are outside timing. Each measured request includes
owned output conversion and independent verification. Raw rows retain failures;
successful latency percentiles never hide the separately reported success rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import holistic_motion as hm
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples/python/planning")
)
from classic_plan_and_retime import (
    EDGE_RESOLUTION,
    GOAL,
    PLANNING_MARGIN,
    START,
    WALL,
    collision_free,
)
from named_plan_and_retime import sample_polyline


def edge_clears_wall(start, goal):
    """Exact slab intersection, independent of the planner's sampled validator."""
    lower = np.array([WALL[0], WALL[2]])
    upper = np.array([WALL[1], WALL[3]])
    enter, leave = 0.0, 1.0
    for axis in range(2):
        delta = goal[axis] - start[axis]
        if delta == 0:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return True
            continue
        first = (lower[axis] - start[axis]) / delta
        last = (upper[axis] - start[axis]) / delta
        enter = max(enter, min(first, last))
        leave = min(leave, max(first, last))
        if enter > leave:
            return True
    return False


def request(algorithm, profile, seed, timeout):
    validator = lambda state: collision_free(state, PLANNING_MARGIN)
    space = hm.JointSpace([-1.0, -1.0], [1.0, 1.0], ["x", "y"])
    space = space.with_joint_weights([1.0, 1.0])
    planner = hm.SamplingPlanner(space, validator)
    optimizer = hm.PathOptimizer(space, validator)
    options = hm.PlanningOptions()
    options.algorithm = getattr(hm.SamplingAlgorithm, algorithm)
    options.timeout_seconds = timeout
    options.max_iterations = 5000
    options.extension_range = 0.15
    options.edge_resolution = EDGE_RESOLUTION
    options.random_seed = seed
    optimization = hm.PathOptimizationOptions()
    optimization.timeout_seconds = timeout
    optimization.edge_resolution = EDGE_RESOLUTION
    row = {"algorithm": algorithm, "profile": profile, "seed": seed, "success": False}
    started = perf_counter()
    stage = "planning"
    stage_started = started
    try:
        planned = planner.plan(START, GOAL, options)
        row["planning_ms"] = (perf_counter() - stage_started) * 1000
        row["planning_checks"] = planned.statistics.collision_checks
        if not planned.success:
            raise RuntimeError(planned.message)
        stage = "optimization"
        stage_started = perf_counter()
        optimized = optimizer.optimize(planned.path, optimization)
        row["optimization_ms"] = (perf_counter() - stage_started) * 1000
        row["optimization_checks"] = optimized.statistics.collision_checks
        if not optimized.success:
            raise RuntimeError(optimized.message)
        path = np.asarray(optimized.joint_space.unwrap_path(optimized.path))
        row["planned_points"] = len(planned.path)
        row["optimized_points"] = len(path)
        row["planned_length"] = planned.statistics.final_path_length
        row["optimized_length"] = optimized.statistics.final_path_length
        stage = "retiming_and_sampling"
        stage_started = perf_counter()
        jerk = None
        if profile == "toppra":
            time, position, velocity, acceleration = sample_polyline(
                path, [0.8, 0.8], [1.5, 1.5], 65
            )
        else:
            trajectory = hm.RnTrajectory(
                path,
                max_velocity=[0.8, 0.8],
                max_acceleration=[1.5, 1.5],
                max_jerk=[4.0, 4.0],
                blend_tolerance=0.0,
                profile=profile,
            )
            time, position, velocity, acceleration, jerk = trajectory.sample_uniform(
                401
            )
        row["retiming_and_sampling_ms"] = (perf_counter() - stage_started) * 1000
        stage = "verification"
        stage_started = perf_counter()
        if not all(edge_clears_wall(a, b) for a, b in zip(path[:-1], path[1:])):
            raise RuntimeError("optimized polyline intersects the physical wall")
        for array in (time, position, velocity, acceleration):
            if not np.isfinite(array).all():
                raise RuntimeError("non-finite trajectory")
        if profile == "double_s" and (
            not np.isfinite(jerk).all() or np.any(np.abs(jerk) > 4.0 * (1 + 1e-8))
        ):
            raise RuntimeError("Double-S samples exceed the configured jerk limit")
        if (
            np.any(np.diff(time) <= 0)
            or not np.allclose(position[[0, -1]], [START, GOAL], rtol=0, atol=1e-10)
            or not np.allclose(velocity[[0, -1]], 0, rtol=0, atol=1e-10)
            or np.any(np.abs(position) > 1 + 1e-10)
            or np.any(np.abs(velocity) > 0.8 * (1 + 1e-8))
            or np.any(np.abs(acceleration) > 1.5 * (1 + 1e-8))
            or not all(collision_free(q) for q in position)
        ):
            raise RuntimeError("trajectory failed endpoint, collision or limit checks")
        row["verification_ms"] = (perf_counter() - stage_started) * 1000
        row.update(success=True, duration=float(time[-1]), samples=len(time))
    except (RuntimeError, ValueError) as error:
        row.setdefault(f"{stage}_ms", (perf_counter() - stage_started) * 1000)
        row.update(failed_stage=stage, message=str(error))
    row["end_to_end_ms"] = (perf_counter() - started) * 1000
    return row


def summarize(rows):
    successful = [row for row in rows if row["success"]]
    summary = {
        "requests": len(rows),
        "successes": len(successful),
        "success_rate": len(successful) / len(rows),
    }
    for metric in (
        "planning_ms",
        "optimization_ms",
        "retiming_and_sampling_ms",
        "verification_ms",
        "end_to_end_ms",
        "duration",
        "planned_length",
        "optimized_length",
    ):
        values = [row[metric] for row in successful]
        summary[metric] = (
            {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
            if values
            else None
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("RRT_CONNECT", "RRT_STAR", "INFORMED_RRT_STAR"),
        default=["RRT_CONNECT", "RRT_STAR", "INFORMED_RRT_STAR"],
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("double_s", "trapezoidal", "toppra"),
        default=["double_s", "toppra"],
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    if any(seed < 0 or seed >= 2**64 for seed in args.seeds):
        parser.error("seeds must be unsigned 64-bit integers")
    rows = []
    for algorithm in args.algorithms:
        for profile in args.profiles:
            request(algorithm, profile, args.seeds[0], args.timeout)
    # Alternate order to reduce persistent algorithm/profile ordering effects.
    cases = [
        (a, p, s) for a in args.algorithms for p in args.profiles for s in args.seeds
    ]
    for repeat in range(args.repeats):
        for algorithm, profile, seed in cases if repeat % 2 == 0 else reversed(cases):
            row = request(algorithm, profile, seed, args.timeout)
            rows.append(dict(row, repeat=repeat))
    groups = []
    for algorithm in args.algorithms:
        for profile in args.profiles:
            selected = [
                r
                for r in rows
                if r["algorithm"] == algorithm and r["profile"] == profile
            ]
            groups.append(
                dict(algorithm=algorithm, profile=profile, **summarize(selected))
            )
    report = {
        "configuration": {
            "timeout_seconds": args.timeout,
            "repeats": args.repeats,
            "seeds": args.seeds,
            "edge_resolution": EDGE_RESOLUTION,
            "planning_margin": PLANNING_MARGIN,
            "algorithms": args.algorithms,
            "profiles": args.profiles,
        },
        "summary": groups,
        "requests": rows,
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(output, end="")
    if args.json is not None:
        args.json.write_text(output)
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
