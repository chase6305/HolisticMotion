"""Generate a complete link-local collision-sphere model from a URDF."""

import argparse
from pathlib import Path

from holistic_motion.geometry import (
    SphereFitOptions,
    fit_urdf_collision_spheres,
    save_sphere_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, action="append", default=[])
    parser.add_argument("--link", action="append", dest="links")
    parser.add_argument("--max-spheres", type=int, default=32)
    parser.add_argument("--min-radius", type=float, default=0.002)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--surface-samples", type=int, default=5000)
    parser.add_argument("--sampled-coverage", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    options = SphereFitOptions(
        max_spheres=args.max_spheres,
        min_radius=args.min_radius,
        padding=args.padding,
        sampled_coverage=args.sampled_coverage,
    )
    results = fit_urdf_collision_spheres(
        args.urdf,
        options,
        links=args.links,
        package_dirs=args.package_dir,
        pitch=args.pitch,
        surface_samples=args.surface_samples,
        random_seed=args.random_seed,
    )
    save_sphere_model(
        args.output,
        {link: result.spheres for link, result in results.items()},
        metadata={
            "source_urdf": str(args.urdf.resolve()),
            "fit_mode": (
                "sampled_coverage" if args.sampled_coverage else "inscribed"
            ),
            "random_seed": args.random_seed,
        },
    )
    for link, result in results.items():
        metrics = result.metrics
        print(
            f"{link}: spheres={metrics.sphere_count} "
            f"coverage={metrics.sampled_coverage:.2%} "
            f"mean_gap={metrics.mean_uncovered_distance:.6f} m "
            f"max_gap={metrics.maximum_uncovered_distance:.6f} m"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
