"""Plan named scalar joints while preserving a caller's full configuration.

Each checked polyline edge is timed separately with TOPPRA, stopping at its
corners. Two-point splines preserve the edge geometry; a single spline through
all waypoints could leave the collision-checked path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import holistic_motion as hm
import numpy as np
from holistic_motion.trajectory import ToppraTrajectory


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def sample_polyline(path, max_velocity, max_acceleration, samples_per_edge):
    """Return scalar positions and derivatives without rounding path corners."""
    blocks = []
    elapsed = 0.0
    for start, goal in zip(path[:-1], path[1:]):
        if np.array_equal(start, goal):
            continue
        trajectory = ToppraTrajectory([start, goal], max_velocity, max_acceleration)
        time, position, velocity, acceleration = trajectory.sample_uniform(
            samples_per_edge
        )
        # Keep each corner once. Both adjoining profiles have zero velocity.
        keep = slice(None) if not blocks else slice(1, None)
        blocks.append(
            (time[keep] + elapsed, position[keep], velocity[keep], acceleration[keep])
        )
        elapsed += trajectory.duration
    if not blocks:
        position = np.asarray(path[:1], dtype=float)
        return np.zeros(1), position, np.zeros_like(position), np.zeros_like(position)
    samples = tuple(np.concatenate(values) for values in zip(*blocks))
    if not np.isfinite(samples[0]).all() or np.any(np.diff(samples[0]) <= 0):
        raise ValueError(
            "polyline timing cannot represent strictly increasing timestamps"
        )
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, action="append", default=[])
    parser.add_argument("--joint-names", nargs="+", required=True)
    parser.add_argument("--reference", nargs="+", type=float)
    parser.add_argument("--start", nargs="+", type=float, required=True)
    parser.add_argument("--goal", nargs="+", type=float, required=True)
    parser.add_argument("--max-velocity", nargs="+", type=positive, required=True)
    parser.add_argument("--max-acceleration", nargs="+", type=positive, required=True)
    parser.add_argument(
        "--timeout",
        type=positive,
        default=2.0,
        help="Time budget in seconds for each planning/optimization stage",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples-per-edge", type=int, default=101)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.samples_per_edge < 2:
        parser.error("--samples-per-edge must be at least 2")
    if args.seed < 0 or args.seed >= 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")
    for name in ("start", "goal", "max_velocity", "max_acceleration"):
        if len(getattr(args, name)) != len(args.joint_names):
            parser.error(f"--{name.replace('_', '-')} must follow --joint-names")

    model = hm.CollisionModel(
        str(args.urdf.expanduser().resolve(strict=True)),
        [str(path.expanduser().resolve(strict=True)) for path in args.package_dir],
    )
    reference = (
        model.neutral_configuration()
        if args.reference is None
        else np.asarray(args.reference, dtype=float)
    )
    model.validate_configuration(reference)
    mapping = model.create_joint_position_map(args.joint_names)
    planner = hm.SamplingPlanner.from_collision_joints(
        model, args.joint_names, reference
    )
    optimizer = hm.PathOptimizer.from_collision_joints(
        model, args.joint_names, reference
    )
    options = hm.PlanningOptions()
    options.timeout_seconds = args.timeout
    options.random_seed = args.seed
    started = perf_counter()
    result = planner.plan(args.start, args.goal, options)
    planning_ms = (perf_counter() - started) * 1000
    if not result.success:
        print(
            json.dumps(
                {"success": False, "stage": "planning", "message": result.message}
            )
        )
        return 1
    started = perf_counter()
    optimization = hm.PathOptimizationOptions()
    optimization.timeout_seconds = args.timeout
    optimized = optimizer.optimize(result.path, optimization)
    optimization_ms = (perf_counter() - started) * 1000
    if not optimized.success:
        print(
            json.dumps(
                {
                    "success": False,
                    "stage": "optimization",
                    "message": optimized.message,
                }
            )
        )
        return 1

    # Use the result's topology snapshot, including continuous joint flags.
    space = optimized.joint_space
    path = np.asarray(space.unwrap_path(optimized.path))
    started = perf_counter()
    time, position, velocity, acceleration = sample_polyline(
        path, args.max_velocity, args.max_acceleration, args.samples_per_edge
    )
    retiming_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    bounded = np.ones(len(args.joint_names), dtype=bool)
    bounded[space.continuous_joint_indices] = False
    if (
        np.any(position[:, bounded] < space.lower_limits[bounded] - 1e-10)
        or np.any(position[:, bounded] > space.upper_limits[bounded] + 1e-10)
        or np.any(np.abs(velocity) > np.asarray(args.max_velocity) * (1 + 1e-8))
        or np.any(np.abs(acceleration) > np.asarray(args.max_acceleration) * (1 + 1e-8))
    ):
        raise RuntimeError("retimed samples violate joint limits")
    # Fill one output buffer instead of retaining a list of full configurations
    # and then copying the entire list into another sampled array.
    configuration = np.empty((len(position), model.nq))
    for index, point in enumerate(position):
        full_q = mapping.configuration_with_joint_positions(reference, point)
        model.validate_configuration(full_q)
        if model.in_collision(full_q):
            raise RuntimeError("retimed sample is in collision")
        configuration[index] = full_q
    verification_ms = (perf_counter() - started) * 1000
    if args.output is not None:
        np.savez(
            args.output,
            joint_names=args.joint_names,
            time=time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            configuration=configuration,
            reference=reference,
        )
    print(
        json.dumps(
            {
                "success": True,
                "joint_names": args.joint_names,
                "nq": model.nq,
                "nv": model.nv,
                "active_dof": len(args.joint_names),
                "collision_pairs": model.pair_count,
                "planned_points": len(result.path),
                "optimized_points": len(optimized.path),
                "samples": len(time),
                "duration": float(time[-1]),
                "planning_ms": planning_ms,
                "optimization_ms": optimization_ms,
                "retiming_ms": retiming_ms,
                "verification_ms": verification_ms,
                "planning_checks": result.statistics.collision_checks,
                "optimization_checks": optimized.statistics.collision_checks,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
