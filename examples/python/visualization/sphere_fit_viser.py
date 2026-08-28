"""Fit collision spheres to one mesh and inspect them in Viser."""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from holistic_motion.geometry import (
    SphereFitOptions,
    fit_trimesh,
    save_sphere_model,
)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene is empty: {path}")
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")
    return loaded


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--link", required=True, help="Owning URDF link name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-spheres", type=int, default=32)
    parser.add_argument("--min-radius", type=float, default=0.002)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--surface-samples", type=int, default=5000)
    parser.add_argument("--sampled-coverage", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = load_mesh(args.mesh)
    options = SphereFitOptions(
        max_spheres=args.max_spheres,
        min_radius=args.min_radius,
        padding=args.padding,
        sampled_coverage=args.sampled_coverage,
    )
    result = fit_trimesh(
        mesh,
        options,
        pitch=args.pitch,
        surface_samples=args.surface_samples,
    )
    metrics = result.metrics
    print(
        f"mode={result.mode} spheres={metrics.sphere_count} "
        f"sampled_coverage={metrics.sampled_coverage:.3%} "
        f"mean_gap={metrics.mean_uncovered_distance:.6f} m "
        f"max_gap={metrics.maximum_uncovered_distance:.6f} m"
    )
    if args.output:
        save_sphere_model(
            args.output,
            {args.link: result.spheres},
            metadata={
                "source_mesh": str(args.mesh.resolve()),
                "fit_mode": result.mode,
                "sampled_coverage": metrics.sampled_coverage,
            },
        )
        print(f"wrote {args.output}")
    if args.fit_only:
        return

    server = viser.ViserServer(port=args.port)
    server.scene.add_mesh_trimesh("/mesh", mesh, opacity=0.35)
    colors = [(45, 170, 255), (255, 155, 40)]
    for index, sphere in enumerate(result.spheres):
        server.scene.add_icosphere(
            f"/spheres/{index:03d}",
            radius=sphere.radius,
            position=np.asarray(sphere.center),
            color=colors[index % len(colors)],
            opacity=0.45,
        )
    print(f"Viser: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
