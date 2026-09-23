"""Measure native sphere queries using a generated seven-joint robot.

Run with scripts/run.sh and the Python interpreter used to build the extension.
Times include the Python binding and use median wall time after warmup.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import holistic_motion as hm
import numpy as np


def make_model(directory, count):
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
    return hm.SphereCollisionModel(str(path), spheres)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.queries, args.repeats) < 1:
        parser.error("queries and repeats must be positive")
    configurations = np.random.default_rng(20260922).uniform(
        -0.3, 0.3, (args.queries, 7)
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="hm-sphere-benchmark-") as temporary:
        for count in (2, 8, 32):
            model = make_model(Path(temporary), count)
            operations = {
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
            }
            for name, query in operations.items():
                elapsed = []
                checksum = None
                for repeat in range(args.repeats + 1):
                    begin = time.perf_counter()
                    value = sum(query(q) for q in configurations)
                    duration = time.perf_counter() - begin
                    if checksum is not None and value != checksum:
                        raise RuntimeError("query results changed between repeats")
                    checksum = value
                    if repeat:
                        elapsed.append(duration * 1e6 / args.queries)
                results.append(
                    {
                        "spheres": model.sphere_count,
                        "pairs": model.pair_count,
                        "query": name,
                        "median_us": statistics.median(elapsed),
                        "checksum": checksum,
                    }
                )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
