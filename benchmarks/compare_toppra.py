"""Compare two standalone NumPy TOPPRA modules on identical path grids.

Example (no compiled extension required)::

    git show 6e56cc7:python/holistic_motion/trajectory/toppra.py > /tmp/toppra-old.py
    python benchmarks/compare_toppra.py --baseline /tmp/toppra-old.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    # Supply package context for relative validation imports without running
    # holistic_motion.__init__ or loading its optional native extension.
    root = Path(__file__).resolve().parents[1] / "python/holistic_motion"
    package_name = f"_holistic_motion_benchmark_{name}"
    for package, directory in (
        (package_name, root),
        (f"{package_name}.trajectory", root / "trajectory"),
    ):
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(directory)]
        namespace.__package__ = package
        sys.modules[package] = namespace
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.trajectory.{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    module.source_sha256 = hashlib.sha256(source).hexdigest()
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "python/holistic_motion/trajectory/toppra.py",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--waypoint-counts", type=int, nargs="+", default=[12])
    parser.add_argument("--dofs", type=int, nargs="+", default=[1, 7, 32])
    parser.add_argument(
        "--active-dof-counts",
        type=int,
        nargs="+",
        help="hold the remaining joints fixed; default: all joints move",
    )
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least two")
    if any(count < 2 for count in args.waypoint_counts):
        parser.error("--waypoint-counts must be at least two")
    if any(dof < 1 for dof in args.dofs):
        parser.error("--dofs must be positive")
    if args.active_dof_counts is not None and any(
        count < 1 for count in args.active_dof_counts
    ):
        parser.error("--active-dof-counts must be positive")
    workloads = [
        (waypoint_count, dof, active_count)
        for waypoint_count in args.waypoint_counts
        for dof in args.dofs
        for active_count in (args.active_dof_counts or [dof])
        if active_count <= dof
    ]
    if not workloads:
        parser.error("no active joint count fits the requested DOFs")
    modules = [
        load_module(args.baseline, "toppra_baseline"),
        load_module(args.candidate, "toppra_candidate"),
    ]
    results = []
    for waypoint_count, dof, active_count in workloads:
        points = np.random.default_rng(53).normal(size=(waypoint_count, dof))
        if active_count < dof:
            points[:, active_count:] = points[0, active_count:]
        velocity = np.ones(dof)
        acceleration = 2.0 * velocity
        for grid_size in (50, 200, 800):
            for end_speed in (0.0, 0.01):
                options = {
                    "gridpoints": np.linspace(0.0, 1.0, grid_size),
                    "end_path_velocity": end_speed,
                }

                def construct(
                    module,
                    points=points,
                    velocity=velocity,
                    acceleration=acceleration,
                    options=options,
                ):
                    return module.ToppraTrajectory(
                        points, velocity, acceleration, **options
                    )

                trajectories = [construct(module) for module in modules]
                # Fixed caller grids separate solver performance from changes
                # in the default discretization. Check actual grids as well.
                for name in (
                    "gridpoints",
                    "path_speeds",
                    "path_accelerations",
                    "times",
                ):
                    np.testing.assert_allclose(
                        getattr(trajectories[0].result, name),
                        getattr(trajectories[1].result, name),
                        rtol=1e-9,
                        atol=1e-10,
                    )
                # Avoid exact acceleration jumps in cross-solver samples.
                samples = [
                    item.sample(np.linspace(0.0037, 0.9961, 501) * item.duration)
                    for item in trajectories
                ]
                for before, after in zip(*samples):
                    np.testing.assert_allclose(before, after, rtol=1e-9, atol=1e-9)

                elapsed = [[], []]
                for repetition in range(args.rounds):
                    for index in (0, 1) if repetition % 2 == 0 else (1, 0):
                        start = time.perf_counter_ns()
                        trajectory = construct(modules[index])
                        elapsed[index].append((time.perf_counter_ns() - start) / 1e6)
                        # Release results outside timing, including the previous
                        # repetition's result, identically for both versions.
                        del trajectory
                medians = [statistics.median(values) for values in elapsed]
                results.append(
                    {
                        "waypoint_count": waypoint_count,
                        "dof": dof,
                        "active_dof_count": active_count,
                        "grid_size": grid_size,
                        "actual_grid_size": len(trajectories[0].result.gridpoints),
                        "end_speed": end_speed,
                        "baseline_ms": medians[0],
                        "candidate_ms": medians[1],
                        "speedup": medians[0] / medians[1],
                        "paired_speedup": statistics.median(
                            before / after for before, after in zip(*elapsed)
                        ),
                        "raw_ms": elapsed,
                    }
                )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "rounds": args.rounds,
                "baseline": str(args.baseline.resolve()),
                "candidate": str(args.candidate.resolve()),
                "baseline_sha256": modules[0].source_sha256,
                "candidate_sha256": modules[1].source_sha256,
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
