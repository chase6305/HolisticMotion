"""Measure complete short collision-planning requests with prepared mappings.

Synthetic models use long joint names to expose metadata-copy costs. Model,
factory and input creation run outside timing. Queries include binding copies,
request snapshots, collision validation and owned output conversion.
"""

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import holistic_motion as hm
import numpy as np
from collision_reconfiguration_benchmark import model_source


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def benchmark(dof, calls, repeats):
    with tempfile.TemporaryDirectory(prefix="hm_planning_snapshot_") as folder:
        path = Path(folder) / "model.urdf"
        source = model_source(dof)
        names = [f"robot_model_joint_number_{index}" for index in range(dof)]
        for index, name in enumerate(names):
            source = source.replace(f'name="joint{index}"', f'name="{name}"')
        path.write_text(source)
        model = hm.CollisionModel(str(path), exclude_adjacent=False)
        start = np.full(dof, 0.05)
        goal = np.full(dof, 0.06)
        planner = hm.SamplingPlanner.from_collision_joints(model, names, start)
        optimizer = hm.PathOptimizer.from_collision_joints(model, names, start)
        options = hm.PlanningOptions()
        options.edge_resolution = 0.5
        optimize_options = hm.PathOptimizationOptions()
        optimize_options.edge_resolution = 0.5
        optimize_options.max_iterations = 1
        path_input = np.vstack((start, (start + goal) / 2, goal))
        queries = {
            "plan": lambda: planner.plan(start, goal, options),
            "optimize": lambda: optimizer.optimize(path_input, optimize_options),
        }
        rows = []
        for name, query in queries.items():
            reference = query()
            assert reference.success
            for _ in range(64):
                assert query().success
            samples = []
            for _ in range(repeats):
                started = perf_counter()
                for _ in range(calls):
                    result = query()
                samples.append((perf_counter() - started) * 1e6 / calls)
                assert result.success
                np.testing.assert_array_equal(result.path, reference.path)
            rows.append(
                {
                    "operation": name,
                    "dof": dof,
                    "calls": calls,
                    "median_us": statistics.median(samples),
                    "samples_us": samples,
                    "checksum": float(np.asarray(reference.path).sum()),
                }
            )
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dof", type=positive_int, nargs="+", default=[2, 7, 64])
    parser.add_argument("--calls", type=positive_int, default=4096)
    parser.add_argument("--repeats", type=positive_int, default=9)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = [row for dof in args.dof for row in benchmark(dof, args.calls, args.repeats)]
    output = json.dumps(rows, indent=2) + "\n"
    print(output, end="")
    if args.json:
        args.json.write_text(output)


if __name__ == "__main__":
    main()
