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
    parser.add_argument(
        "--joint-names",
        nargs="+",
        help="Interpret endpoints as scalar joint positions in this name order.",
    )
    parser.add_argument(
        "--context",
        type=configuration,
        help="Full nq reference for named joints; defaults to model neutral.",
    )
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--package-dir", action="append", default=[], type=Path)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    model = hm.CollisionModel(
        str(args.urdf.expanduser().resolve()),
        [str(path.expanduser().resolve()) for path in args.package_dir],
    )
    mapping = None
    if args.joint_names:
        try:
            mapping = model.create_joint_position_map(args.joint_names)
        except ValueError as error:
            parser.error(str(error))
        context = (
            model.neutral_configuration() if args.context is None else args.context
        )
        size = len(args.joint_names)
    else:
        if args.context is not None:
            parser.error("--context requires --joint-names")
        if model.nq != model.nv:
            parser.error(
                "non-Euclidean configurations require --joint-names; "
                "interpolating cosine/sine or quaternion components is invalid"
            )
        size = model.nq
    if args.q_start.shape != (size,) or args.q_end.shape != (size,):
        parser.error(f"--q-start and --q-end each need {size} values")

    def full_configuration(positions):
        if mapping is None:
            return positions
        return mapping.configuration_with_joint_positions(context, positions)

    try:
        model.validate_configuration(full_configuration(args.q_start))
        model.validate_configuration(full_configuration(args.q_end))
    except ValueError as error:
        parser.error(str(error))
    if not model.pair_count:
        parser.error("model has no active collision pairs")

    first_collision = None
    minimum = None
    for index, alpha in enumerate(np.linspace(0.0, 1.0, args.samples)):
        q = full_configuration((1.0 - alpha) * args.q_start + alpha * args.q_end)
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
