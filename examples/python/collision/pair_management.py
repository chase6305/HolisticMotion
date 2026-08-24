#!/usr/bin/env python3
"""Inspect, remove, and restore active collision pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/examples"))
from _bootstrap import import_holistic_motion  # noqa: E402


hm = import_holistic_motion()


def print_pairs(model, limit: int) -> None:
    pairs = model.collision_pairs
    print(f"active collision pairs: {len(pairs)}")
    for index, pair in enumerate(pairs[:limit]):
        print(
            f"  [{index}] {pair.first_link}/{pair.first_geometry} <-> "
            f"{pair.second_link}/{pair.second_geometry}"
        )
    if len(pairs) > limit:
        print(f"  ... {len(pairs) - limit} more pair(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and edit the active collision-pair set."
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--package-dir", action="append", default=[], type=Path)
    parser.add_argument(
        "--remove", action="append", default=[], metavar="GEOMETRY1:GEOMETRY2",
        help="Remove one geometry pair (repeatable).",
    )
    parser.add_argument(
        "--include-adjacent", action="store_true",
        help="Start with adjacent-link pairs enabled.",
    )
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    model = hm.CollisionModel(
        str(args.urdf.expanduser().resolve()),
        [str(path.expanduser().resolve()) for path in args.package_dir],
        not args.include_adjacent,
    )
    print_pairs(model, args.limit)
    for value in args.remove:
        first, separator, second = value.partition(":")
        if not separator or not first or not second:
            parser.error(f"invalid geometry pair: {value}")
        removed = model.remove_collision_pair(first, second)
        print(f"remove {first} <-> {second}: {'removed' if removed else 'not active'}")
    if args.remove:
        print_pairs(model, args.limit)

    restored = model.reset_collision_pairs(not args.include_adjacent)
    print(f"reset restores {restored} active pair(s)")


if __name__ == "__main__":
    main()
