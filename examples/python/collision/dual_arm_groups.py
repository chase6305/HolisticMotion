#!/usr/bin/env python3
"""Treat two link sets as complete arms and check selected group pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/examples"))
from _bootstrap import import_holistic_motion  # noqa: E402


hm = import_holistic_motion()


def csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check collision between two complete robot arms."
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--left-links", type=csv, required=True)
    parser.add_argument("--right-links", type=csv, required=True)
    parser.add_argument("--configuration", "-q", type=float, nargs="*")
    parser.add_argument("--package-dir", action="append", default=[], type=Path)
    parser.add_argument("--left-self", action="store_true")
    parser.add_argument("--right-self", action="store_true")
    parser.add_argument(
        "--list-pairs", action="store_true",
        help="Print the generated geometry-level collision pairs.",
    )
    args = parser.parse_args()

    model = hm.CollisionModel(
        str(args.urdf.expanduser().resolve()),
        [str(path.expanduser().resolve()) for path in args.package_dir],
    )
    enabled = [("left_arm", "right_arm")]
    if args.left_self:
        enabled.append(("left_arm", "left_arm"))
    if args.right_self:
        enabled.append(("right_arm", "right_arm"))
    model.set_collision_groups(
        {"left_arm": args.left_links, "right_arm": args.right_links}, enabled
    )

    q = np.asarray(
        args.configuration if args.configuration is not None else [0.0] * model.nq,
        dtype=float,
    )
    if q.shape != (model.nq,):
        parser.error(f"configuration needs {model.nq} values, received {q.size}")

    print(f"generated arm collision pairs: {model.pair_count}")
    if args.list_pairs:
        for pair in model.collision_pairs:
            print(
                f"  {pair.first_link}/{pair.first_geometry} <-> "
                f"{pair.second_link}/{pair.second_geometry}"
            )
    collisions = model.collisions(q)
    print(f"arms in collision: {bool(collisions)}")
    for result in collisions:
        print(
            f"  {result.first_geometry} <-> {result.second_geometry}: "
            f"{result.distance:.6f} m"
        )
    if model.pair_count:
        nearest = model.minimum_distance(q)
        print(f"minimum arm distance: {nearest.distance:.6f} m")


if __name__ == "__main__":
    main()
