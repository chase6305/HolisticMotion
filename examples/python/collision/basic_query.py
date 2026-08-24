#!/usr/bin/env python3
"""Run self-collision and minimum-distance queries for a URDF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/examples"))
from _bootstrap import import_holistic_motion  # noqa: E402


hm = import_holistic_motion()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check one robot configuration for self-collision."
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--configuration", "-q", type=float, nargs="*",
        help="Pinocchio configuration values; defaults to zero for fixed-base robots.",
    )
    parser.add_argument(
        "--package-dir", action="append", default=[], type=Path,
        help="Package root used to resolve package:// mesh paths (repeatable).",
    )
    parser.add_argument(
        "--include-adjacent", action="store_true",
        help="Keep collision pairs belonging to adjacent links.",
    )
    parser.add_argument(
        "--group", action="append", default=[], metavar="NAME=LINK1,LINK2",
        help="Define a collision group by link names (repeatable).",
    )
    parser.add_argument(
        "--check-groups", action="append", default=[], metavar="GROUP1:GROUP2",
        help="Enable a group pair, including NAME:NAME for group self-collision.",
    )
    args = parser.parse_args()

    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF does not exist: {urdf_path}")
    if not hasattr(hm, "CollisionModel"):
        parser.error("collision support is disabled; rebuild without --no-collision")

    model = hm.CollisionModel(
        str(urdf_path),
        [str(path.expanduser().resolve()) for path in args.package_dir],
        not args.include_adjacent,
    )
    if bool(args.group) != bool(args.check_groups):
        parser.error("--group and --check-groups must be used together")
    if args.group:
        groups = {}
        for value in args.group:
            name, separator, links = value.partition("=")
            members = [link for link in links.split(",") if link]
            if not separator or not name or not members or name in groups:
                parser.error(f"invalid or duplicate group definition: {value}")
            groups[name] = members
        group_pairs = []
        for value in args.check_groups:
            first, separator, second = value.partition(":")
            if not separator or not first or not second:
                parser.error(f"invalid group pair: {value}")
            group_pairs.append((first, second))
        model.set_collision_groups(groups, group_pairs)
    q = np.asarray(
        args.configuration if args.configuration is not None else [0.0] * model.nq,
        dtype=float,
    )
    if q.shape != (model.nq,):
        parser.error(f"configuration needs {model.nq} values, received {q.size}")

    collisions = model.collisions(q)
    print(
        f"model: nq={model.nq}, nv={model.nv}, "
        f"geometries={model.geometry_count}, pairs={model.pair_count}"
    )
    print(f"in collision: {bool(collisions)}")
    for result in collisions:
        print(
            f"  pair[{result.pair_index}] {result.first_geometry} <-> "
            f"{result.second_geometry}, distance={result.distance:.6f} m"
        )

    if model.pair_count:
        nearest = model.minimum_distance(q)
        print(
            f"minimum distance: {nearest.distance:.6f} m "
            f"({nearest.first_geometry} <-> {nearest.second_geometry})"
        )


if __name__ == "__main__":
    main()
