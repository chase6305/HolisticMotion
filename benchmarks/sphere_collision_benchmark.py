"""Measure native sphere queries using a generated seven-joint robot.

Run with scripts/run.sh and the Python interpreter used to build the extension.
Times include the Python binding and use median wall time after warmup.
Use --extra-fixed-frames to measure robots whose sensor/tool frames are not
part of their collision-sphere representation.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def make_model(directory, count, extra_fixed_frames=0, radius_ratio=1.0):
    import holistic_motion as hm

    links = ['<link name="base"/>']
    parent = "base"
    for index in range(7):
        name = f"link_{index}"
        links.append(f'<link name="{name}"/>')
        links.append(
            f'<joint name="joint_{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="{name}"/>'
            '<origin xyz="0.25 0 0"/><axis xyz="0 0 1"/>'
            '<limit lower="-3" upper="3" effort="1" velocity="1"/>'
            "</joint>"
        )
        parent = name
    for index in range(extra_fixed_frames):
        name = f"sensor_{index}"
        links.append(f'<link name="{name}"/>')
        links.append(
            f'<joint name="sensor_mount_{index}" type="fixed">'
            f'<parent link="link_{index % 7}"/><child link="{name}"/>'
            '<origin xyz="0.01 0.02 0.03" rpy="0.1 0.2 0.3"/>'
            "</joint>"
        )
    path = directory / "benchmark.urdf"
    path.write_text('<robot name="benchmark">' + "".join(links) + "</robot>")
    spheres = [
        hm.CollisionSphere(
            f"sphere_{link}_{point}",
            f"link_{link}",
            [0.2 * point / count, 0.015 * (point % 3 - 1), 0.0],
            0.01,
        )
        for link in range(7)
        for point in range(count)
    ]
    spheres[0].radius *= radius_ratio
    return hm.SphereCollisionModel(str(path), spheres)


def measure(args):
    configurations = np.random.default_rng(20260922).uniform(
        -0.3, 0.3, (args.queries, 7)
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="hm-sphere-benchmark-") as temporary:
        for count in (2, 8, 32):
            model = make_model(
                Path(temporary), count, args.extra_fixed_frames, args.radius_ratio
            )
            expected_distances = np.array(
                [model.minimum_distance(q).distance for q in configurations]
            )
            operations = {
                "world_spheres": lambda q, model=model: float(
                    np.sum(model.world_spheres(q))
                ),
                "minimum_distance": lambda q, model=model: (
                    model.minimum_distance(q).distance
                ),
                "distance_gradient": lambda q, model=model: float(
                    np.sum(model.minimum_distance_with_gradient(q).gradient)
                ),
                "filtered_distances": lambda q, model=model: float(
                    len(model.distances(q, 0.05))
                ),
                "in_collision": lambda q, model=model: float(model.in_collision(q)),
                "batch_minimum_distances": None,
            }
            for name, query in operations.items():
                elapsed = []
                checksum = None
                for repeat in range(args.repeats + 1):
                    begin = time.perf_counter()
                    if query is None:
                        distances = model.batch_minimum_distances(configurations)
                    else:
                        value = sum(query(q) for q in configurations)
                    duration = time.perf_counter() - begin
                    if query is None:
                        np.testing.assert_array_equal(distances, expected_distances)
                        value = float(distances.sum())
                    if checksum is not None and value != checksum:
                        raise RuntimeError("query results changed between repeats")
                    checksum = value
                    if repeat:
                        elapsed.append(duration * 1e6 / args.queries)
                results.append(
                    {
                        "spheres": model.sphere_count,
                        "pairs": model.pair_count,
                        "extra_fixed_frames": args.extra_fixed_frames,
                        "radius_ratio": args.radius_ratio,
                        "query": name,
                        "median_us": statistics.median(elapsed),
                        "samples_us": elapsed,
                        "checksum": checksum,
                    }
                )
    return results


def package(value):
    label, separator, paths = value.partition("=")
    if not separator or not label or not paths:
        raise argparse.ArgumentTypeError("expected LABEL=PYTHONPATH")
    resolved = [Path(item).resolve() for item in paths.split(os.pathsep)]
    if any(not path.is_dir() for path in resolved):
        raise argparse.ArgumentTypeError("package paths must be directories")
    return label, os.pathsep.join(map(str, resolved))


def compare_packages(args):
    samples = {label: [] for label, _ in args.package}
    expected = None
    for repeat in range(args.repeats):
        order = args.package if repeat % 2 == 0 else reversed(args.package)
        for label, paths in order:
            environment = dict(
                os.environ, PYTHONPATH=paths, HOLISTICMOTION_PURE_PYTHON="0"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--queries",
                    str(args.queries),
                    "--repeats",
                    "1",
                    "--extra-fixed-frames",
                    str(args.extra_fixed_frames),
                    "--radius-ratio",
                    str(args.radius_ratio),
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=True,
                timeout=120,
            )
            rows = json.loads(completed.stdout)
            identity = [
                (r["spheres"], r["pairs"], r["query"], r["checksum"]) for r in rows
            ]
            if expected is not None and identity != expected:
                raise RuntimeError(
                    f"{label}: query results differ across packages or rounds"
                )
            expected = identity
            samples[label].append(rows)
    results = []
    for label, paths in args.package:
        for index, row in enumerate(samples[label][0]):
            values = [sample[index]["median_us"] for sample in samples[label]]
            results.append(
                dict(
                    row,
                    label=label,
                    pythonpath=paths,
                    median_us=statistics.median(values),
                    samples_us=values,
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--extra-fixed-frames", type=int, default=0)
    parser.add_argument(
        "--radius-ratio",
        type=float,
        default=1.0,
        help="Scale the first radius to exercise bounds with a large outlier",
    )
    parser.add_argument(
        "--package",
        type=package,
        action="append",
        help="Compare LABEL=PYTHONPATH SDKs in alternating isolated processes",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.queries, args.repeats) < 1:
        parser.error("queries and repeats must be positive")
    if args.extra_fixed_frames < 0:
        parser.error("extra-fixed-frames must be non-negative")
    if not np.isfinite(args.radius_ratio) or args.radius_ratio <= 0:
        parser.error("radius-ratio must be finite and positive")
    if args.package and len({label for label, _ in args.package}) != len(args.package):
        parser.error("package labels must be unique")
    rows = compare_packages(args) if args.package else measure(args)
    output = json.dumps(rows, indent=2) + "\n"
    print(output, end="")
    if args.json is not None:
        args.json.write_text(output)


if __name__ == "__main__":
    main()
