"""Measure complete URDF primitive loading, excluding input file creation.

Run with HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python. Optionally pass
--baseline /path/to/previous/urdf_sphere_fit.py for paired comparisons.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh
from holistic_motion.geometry import urdf_sphere_fit


def _write_urdf(path, shape, count, repeated):
    links = []
    for index in range(count):
        scale = 1.0 if repeated else 1.0 + index / count
        if shape == "box":
            geometry = f'<box size="{scale} {2 * scale} {3 * scale}"/>'
        elif shape == "cylinder":
            geometry = f'<cylinder radius="{scale}" length="{3 * scale}"/>'
        else:
            geometry = f'<sphere radius="{scale}"/>'
        links.append(
            f'<link name="link{index}"><collision>'
            f'<origin xyz="{index} 0 0" rpy="0 0 {index * 0.1}"/>'
            f"<geometry>{geometry}</geometry></collision></link>"
        )
    path.write_text('<robot name="benchmark">' + "".join(links) + "</robot>")


def _measure(path, baseline, repeats):
    modules = {"current": urdf_sphere_fit}
    if baseline is not None:
        modules["baseline"] = baseline
    results = {
        name: module.load_urdf_collision_meshes(path)
        for name, module in modules.items()
    }
    if baseline is not None:
        for link, mesh in results["current"].items():
            previous = results["baseline"][link]
            np.testing.assert_array_equal(mesh.vertices, previous.vertices)
            np.testing.assert_array_equal(mesh.faces, previous.faces)
    timings = {name: [] for name in modules}
    for repeat in range(repeats):
        # Alternate execution order to reduce drift in paired comparisons.
        order = list(modules) if repeat % 2 == 0 else list(reversed(modules))
        for name in order:
            started = time.perf_counter()
            meshes = modules[name].load_urdf_collision_meshes(path)
            elapsed = time.perf_counter() - started
            if len(meshes) != len(results[name]):
                raise RuntimeError("loaded link count changed")
            timings[name].append(elapsed * 1000)
            del meshes
    record = {
        name: {"median_ms": statistics.median(samples)}
        for name, samples in timings.items()
    }
    if baseline is not None:
        record["speedup"] = statistics.median(
            previous / current
            for previous, current in zip(timings["baseline"], timings["current"])
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--links", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if min([args.repeats] + args.links) < 1:
        parser.error("links and repeats must be positive")
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location(
            "holistic_motion.geometry._urdf_geometry_baseline", args.baseline
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
    records = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "robot.urdf"
        for shape in ("box", "cylinder", "sphere"):
            for count in args.links:
                for repeated in (False, True) if count > 1 else (False,):
                    _write_urdf(path, shape, count, repeated)
                    records.append(
                        {
                            "shape": shape,
                            "links": count,
                            "repeated": repeated,
                            **_measure(path, baseline, args.repeats),
                        }
                    )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "trimesh": trimesh.__version__,
                "repeats": args.repeats,
                "sources": {
                    name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                    for name, module in (
                        {"current": urdf_sphere_fit, "baseline": baseline}
                        if baseline is not None
                        else {"current": urdf_sphere_fit}
                    ).items()
                },
                "cases": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
