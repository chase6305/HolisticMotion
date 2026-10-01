"""Compare two standalone NumPy TOPPRA modules on identical path grids.

Example (no compiled extension required)::

    git show 6e56cc7:python/holistic_motion/trajectory/toppra.py > /tmp/toppra-old.py
    python benchmarks/compare_toppra.py --baseline /tmp/toppra-old.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least two")
    modules = [
        load_module(args.baseline, "toppra_baseline"),
        load_module(args.candidate, "toppra_candidate"),
    ]
    results = []
    for dof in (1, 7, 32):
        points = np.random.default_rng(53).normal(size=(12, dof))
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
                samples = [
                    item.sample(np.linspace(0.0, item.duration, 501))
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
                        "dof": dof,
                        "grid_size": grid_size,
                        "actual_grid_size": len(trajectories[0].result.gridpoints),
                        "end_speed": end_speed,
                        "baseline_ms": medians[0],
                        "candidate_ms": medians[1],
                        "speedup": medians[0] / medians[1],
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
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
