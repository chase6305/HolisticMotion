"""Measure complete geometry/pair edits on synthetic scalar-chain models."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import holistic_motion as hm
import numpy as np


def model_source(count):
    geometry = '<geometry><sphere radius="0.1"/></geometry>'
    links = [f'<link name="base"><collision>{geometry}</collision></link>']
    joints = []
    for index in range(count):
        collision = f"<collision>{geometry}</collision>" if index == count - 1 else ""
        links.append(f'<link name="link{index}">{collision}</link>')
        parent = "base" if index == 0 else f"link{index - 1}"
        joints.append(
            f'<joint name="joint{index}" type="prismatic"><parent link="{parent}"/>'
            f'<child link="link{index}"/><origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>'
            '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint>'
        )
    return '<robot name="reconfiguration">' + "".join(links + joints) + "</robot>"


def benchmark(count, calls, repeats):
    with TemporaryDirectory(prefix="holistic-motion-collision-benchmark-") as folder:
        path = Path(folder) / "chain.urdf"
        path.write_text(model_source(count))
        model = hm.CollisionModel(str(path), exclude_adjacent=False)
        groups = {"base": ["base"], "tip": [f"link{count - 1}"]}
        pairs = [("base", "tip")]
        obstacle_pose = np.eye(4)
        obstacle_pose[0, 3] = 3.0

        def obstacle_cycle():
            model.add_box_obstacle("world_box", [0.1, 0.1, 0.1], obstacle_pose)
            assert model.remove_obstacle("world_box")
            return model.pair_count

        operations = {
            "reset_pairs": lambda: model.reset_collision_pairs(False),
            "set_groups": lambda: model.set_collision_groups(groups, pairs),
            "obstacle_add_remove": obstacle_cycle,
        }
        q = np.full(count, 0.05)
        expected = np.hypot(0.05 * count, 0.1 * count) - 0.2

        def verify():
            assert model.pair_count == 1
            distance = model.minimum_distance(q).distance
            assert np.isclose(distance, expected, rtol=1e-12, atol=1e-12)
            return distance

        verify()
        rows = []
        for name, operation in operations.items():
            for _ in range(3):
                assert operation() == 1
            timings = []
            for _ in range(repeats):
                started = perf_counter()
                checksum = sum(operation() for _ in range(calls))
                timings.append((perf_counter() - started) * 1e6 / calls)
                assert checksum == calls
            rows.append(
                dict(
                    joints=count,
                    operation=name,
                    calls=calls,
                    median_us=float(np.median(timings)),
                    samples_us=timings,
                    minimum_distance=verify(),
                )
            )
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", type=int, nargs="+", default=[7, 32, 128])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()
    if min(arguments.joints + [arguments.calls, arguments.repeats]) < 1:
        parser.error("joint counts, calls and repeats must be positive")
    if not hasattr(hm, "CollisionModel"):
        parser.error("select a native build with collision support")
    results = [
        row
        for count in arguments.joints
        for row in benchmark(count, arguments.calls, arguments.repeats)
    ]
    print("joints,operation,calls,median_us,minimum_distance")
    for row in results:
        print(
            f"{row['joints']},{row['operation']},{row['calls']},"
            f"{row['median_us']:.6f},{row['minimum_distance']:.12g}"
        )
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
