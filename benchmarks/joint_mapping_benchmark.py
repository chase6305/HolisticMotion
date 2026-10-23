"""Measure repeated name resolution versus a reusable joint-position mapping.

URDF parsing, mapping construction and equivalence checks are outside timing.
Complete calls include each API's validation, binding conversion and packing,
without FK or collision queries. Model convenience calls also validate the full
configuration manifold; backend-independent maps validate selected blocks.
"""

import argparse
import csv
import json
import statistics
import sys
import tempfile
from functools import partial
from pathlib import Path
from time import perf_counter

import holistic_motion as hm
import numpy as np
from holistic_motion.configuration import JointPositionMap


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def write_model(path, count, continuous):
    links = [
        '<link name="link0"><collision><geometry>'
        '<sphere radius="0.1"/></geometry></collision></link>'
    ]
    joints = []
    for index in range(count):
        links.append(f'<link name="link{index + 1}"/>')
        kind = "continuous" if continuous and index % 3 == 0 else "revolute"
        joints.append(
            f'<joint name="joint{index}" type="{kind}">'
            f'<parent link="link{index}"/><child link="link{index + 1}"/>'
            '<axis xyz="0 0 1"/>'
            '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        )
    path.write_text('<robot name="mapping">' + "".join(links + joints) + "</robot>")


def benchmark(path, count, selected, continuous, iterations, repeats):
    write_model(path, count, continuous)
    model = hm.CollisionModel(str(path))
    selected = min(selected, count)
    names = [f"joint{i}" for i in range(count - selected, count)]
    mapping = model.create_joint_position_map(names)
    numpy_mapping = JointPositionMap.from_model(model, names)
    context = model.neutral_configuration()
    positions = np.full(selected, 0.2)
    queries = {
        "named_convenience": partial(
            model.configuration_with_joint_positions, context, names, positions
        ),
        "reusable_mapping": partial(
            mapping.configuration_with_joint_positions, context, positions
        ),
        "numpy_mapping": partial(
            numpy_mapping.configuration_with_joint_positions, context, positions
        ),
    }
    expected = queries["named_convenience"]()
    for name in ("reusable_mapping", "numpy_mapping"):
        np.testing.assert_allclose(queries[name](), expected, rtol=0, atol=1e-14)
    for query in queries.values():
        for _ in range(1000):
            query()
    samples = {name: [] for name in queries}
    for repeat in range(repeats):
        order = tuple(queries) if repeat % 2 == 0 else tuple(reversed(queries))
        for name in order:
            query = queries[name]
            started = perf_counter()
            for _ in range(iterations):
                result = query()
            elapsed = perf_counter() - started
            samples[name].append(elapsed * 1e6 / iterations)
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
    return [
        {
            "model_joints": count,
            "nq": model.nq,
            "nv": model.nv,
            "selected_joints": selected,
            "mode": name,
            "median_us": statistics.median(values),
            "configuration_checksum": float(np.sum(expected)),
            "samples_us": values,
        }
        for name, values in samples.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", type=positive_int, nargs="+", default=[7, 64, 256])
    parser.add_argument("--selected", type=positive_int, default=7)
    parser.add_argument("--iterations", type=positive_int, default=10000)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not hasattr(hm, "CollisionModel"):
        parser.error("requires a native build with collision enabled")
    rows = []
    with tempfile.TemporaryDirectory(prefix="hm_mapping_benchmark_") as directory:
        path = Path(directory) / "model.urdf"
        for count in args.joints:
            rows.extend(
                benchmark(
                    path,
                    count,
                    args.selected,
                    args.continuous,
                    args.iterations,
                    args.repeats,
                )
            )
    columns = [key for key in rows[0] if key != "samples_us"]
    writer = csv.DictWriter(sys.stdout, columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
