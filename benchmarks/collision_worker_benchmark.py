"""Compare complete worker construction and in-memory collision snapshots.

Uses synthetic URDF primitives and the same configured obstacle/pair set. Clone
includes a deep geometry copy and fresh Pinocchio/Coal workspaces. Query checks
and destruction run outside timing; no external robot asset is needed.
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


def benchmark(count, calls, repeats):
    with tempfile.TemporaryDirectory(prefix="hm_collision_workers_") as folder:
        path = Path(folder) / "model.urdf"
        path.write_text(model_source(count))
        pose = np.eye(4)
        pose[0, 3] = 3.0
        groups = {"base": ["base"], "tip": [f"link{count - 1}"], "world": ["wall"]}
        pairs = [("base", "tip"), ("tip", "world")]

        def construct():
            result = hm.CollisionModel(str(path), exclude_adjacent=False)
            result.add_box_obstacle("wall", [0.1, 0.1, 0.1], pose)
            result.set_collision_groups(groups, pairs)
            return result

        source = construct()
        state = np.full(count, 0.05)
        reference = source.minimum_distance(state).distance
        samples = {"construct": [], "clone": []}
        for repeat in range(repeats):
            order = (
                ("construct", "clone") if repeat % 2 == 0 else ("clone", "construct")
            )
            for kind in order:
                operation = construct if kind == "construct" else source.clone
                started = perf_counter()
                models = [operation() for _ in range(calls)]
                samples[kind].append((perf_counter() - started) * 1e6 / calls)
                for result in models:
                    assert result.pair_count == source.pair_count
                    assert result.geometry_names == source.geometry_names
                    assert np.isclose(
                        result.minimum_distance(state).distance, reference
                    )
                del result, models
        return {
            "dof": count,
            "calls": calls,
            "repeats": repeats,
            "samples_us": samples,
            "median_us": {
                name: statistics.median(values) for name, values in samples.items()
            },
            "distance": reference,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dof", type=positive_int, nargs="+", default=[1, 7, 64])
    parser.add_argument("--calls", type=positive_int, default=32)
    parser.add_argument("--repeats", type=positive_int, default=9)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = [benchmark(count, args.calls, args.repeats) for count in args.dof]
    output = json.dumps(rows, indent=2) + "\n"
    print(output, end="")
    if args.json:
        args.json.write_text(output)


if __name__ == "__main__":
    main()
