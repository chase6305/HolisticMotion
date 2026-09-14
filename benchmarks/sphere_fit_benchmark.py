"""Measure complete NumPy sphere fits, excluding mesh I/O and voxelization.

Run with OPENBLAS_NUM_THREADS=1 HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python.
Optionally pass --baseline /path/to/old/sphere_fit.py to compare implementations.
Timing and tracemalloc peak allocation measurements are performed separately.
"""

import argparse
import importlib.util
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict

import numpy as np
from holistic_motion.geometry import sphere_fit


def _measure(module, interior, surface, coverage, repeats):
    options = module.SphereFitOptions(max_spheres=32, sampled_coverage=coverage)

    def run():
        return module.fit_spheres(interior, surface, options)

    result = run()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    tracemalloc.start()
    try:
        run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "median_ms": float(np.median(timings) * 1000),
        "peak_mib": peak / 2**20,
        "metrics": asdict(result.metrics),
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="Previous sphere_fit.py source file")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location(
            "sphere_fit_baseline", args.baseline
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)

    rng = np.random.default_rng(77)
    cases = []
    for interior_count, surface_count in ((256, 1000), (2000, 5000), (2000, 20000)):
        interior = rng.uniform(-0.5, 0.5, (interior_count, 3))
        surface = rng.normal(size=(surface_count, 3))
        surface /= np.linalg.norm(surface, axis=1)[:, None]
        for coverage in (False, True):
            record = {
                "interior_count": interior_count,
                "surface_count": surface_count,
                "sampled_coverage": coverage,
            }
            record["current"], result = _measure(
                sphere_fit, interior, surface, coverage, args.repeats
            )
            if baseline is not None:
                record["baseline"], previous = _measure(
                    baseline, interior, surface, coverage, args.repeats
                )
                np.testing.assert_allclose(
                    [sphere.center for sphere in result.spheres],
                    [sphere.center for sphere in previous.spheres],
                    rtol=0,
                    atol=0,
                )
                np.testing.assert_allclose(
                    [sphere.radius for sphere in result.spheres],
                    [sphere.radius for sphere in previous.spheres],
                    rtol=1e-12,
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    list(asdict(result.metrics).values()),
                    list(asdict(previous.metrics).values()),
                    rtol=1e-12,
                    atol=1e-14,
                )
                record["speedup"] = (
                    record["baseline"]["median_ms"] / record["current"]["median_ms"]
                )
            cases.append(record)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "repeats": args.repeats,
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
