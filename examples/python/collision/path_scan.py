#!/usr/bin/env python3
"""Sample a joint-space segment and report collisions and clearance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/examples"))
from _bootstrap import import_holistic_motion  # noqa: E402


hm = import_holistic_motion()


def configuration(value: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result.size or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError("configuration must contain finite values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a linear joint-space segment for collision."
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--q-start", type=configuration, required=True)
    parser.add_argument("--q-end", type=configuration, required=True)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--package-dir", action="append", default=[], type=Path)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    model = hm.CollisionModel(
        str(args.urdf.expanduser().resolve()),
        [str(path.expanduser().resolve()) for path in args.package_dir],
    )
    if args.q_start.shape != (model.nq,) or args.q_end.shape != (model.nq,):
        parser.error(f"--q-start and --q-end each need {model.nq} values")
    if not model.pair_count:
        parser.error("model has no active collision pairs")

    first_collision = None
    minimum = None
    for index, alpha in enumerate(np.linspace(0.0, 1.0, args.samples)):
        q = (1.0 - alpha) * args.q_start + alpha * args.q_end
        nearest = model.minimum_distance(q)
        if minimum is None or nearest.distance < minimum[0]:
            minimum = (nearest.distance, alpha, nearest)
        if first_collision is None and model.in_collision(q):
            first_collision = (index, alpha, q.copy())

    assert minimum is not None
    distance, alpha, result = minimum
    print(
        f"minimum clearance: {distance:.6f} m at alpha={alpha:.4f} "
        f"({result.first_geometry} <-> {result.second_geometry})"
    )
    if first_collision is None:
        print("path collision-free at sampled configurations")
    else:
        index, alpha, q = first_collision
        print(f"first collision: sample={index}, alpha={alpha:.4f}, q={q.tolist()}")


if __name__ == "__main__":
    main()
