"""Measure Python trajectory result conversion and sampling without robot assets.

Run against an installed package selected with PYTHONPATH. CSV checksums include
array values, shape, dtype, strides and ownership so independent baseline and
candidate processes can verify results as well as compare timings.
"""

import argparse
import csv
import gc
import hashlib
import statistics
import sys
import time

import holistic_motion as hm
import numpy as np


def checksum(result):
    digest = hashlib.sha256()
    if isinstance(result, dict):
        items = sorted(result.items())
    else:
        items = [(None, value) for value in result]
    for key, value in items:
        if key is not None:
            digest.update(key.encode())
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise RuntimeError("benchmark produced non-finite values")
        metadata = (array.shape, array.dtype.str, array.strides,
                    array.flags.owndata, array.flags.writeable)
        digest.update(repr(metadata).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fixtures(dofs=(1, 3, 7, 14, 32)):
    for profile in ("double_s", "trapezoidal"):
        for dof in dofs:
            for count in (4, 64):
                points = 0.3 * np.sin(
                    0.4 * np.arange(count)[:, None]
                    + 0.3 * np.arange(dof)[None, :]
                )
                limits = np.ones(dof)
                trajectory = hm.RnTrajectory(
                    points, limits, limits, limits,
                    blend_tolerance=0.005, profile=profile,
                )
                yield "Rn", dof, count, profile, trajectory
        start, end = np.eye(4), np.eye(4)
        angle = 0.4
        end[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]
        end[:3, 3] = [0.3, 0.4, 0.5]
        limits = np.ones(6)
        yield "SE3", 6, 2, profile, hm.CartesianLineTrajectory(
            start, end, limits, limits, limits, profile=profile,
        )


def measure(call, batch, repeats):
    call()
    elapsed = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        for _ in range(batch):
            result = call()
        elapsed.append((time.perf_counter_ns() - begin) / batch / 1000)
    return statistics.median(elapsed), checksum(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--operations", nargs="+", choices=("state", "uniform", "report"),
        default=("state", "uniform"),
        help="operations to time; reports include all returned diagnostics",
    )
    parser.add_argument(
        "--dofs", nargs="+", type=int, choices=range(1, 33),
        default=(1, 3, 7, 14, 32),
        help="joint dimensions to measure; Cartesian fixtures are always included",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    hm.setup_logging("ERROR", handlers=[])
    writer = csv.writer(sys.stdout)
    writer.writerow(("group", "dof", "waypoints", "profile", "operation",
                     "samples", "batch", "median_us", "duration", "checksum"))
    gc.disable()
    for group, dof, count, profile, trajectory in fixtures(args.dofs):
        query_time = trajectory.duration * 0.37
        cases = []
        if "state" in args.operations:
            cases.append((
                "state", 1, 256,
                lambda trajectory=trajectory, query_time=query_time:
                    trajectory.state(query_time),
            ))
        for samples, batch in ((2, 128), (65, 16), (2001, 2), (20001, 1)):
            if "uniform" in args.operations:
                cases.append((
                    "uniform", samples, batch,
                    lambda samples=samples, trajectory=trajectory:
                        trajectory.sample_uniform(samples),
                ))
            if "report" in args.operations:
                cases.append((
                    "report", samples, batch,
                    lambda samples=samples, trajectory=trajectory:
                        trajectory.constraint_report(samples),
                ))
        for operation, samples, batch, call in cases:
            elapsed, digest = measure(call, batch, args.repeats)
            writer.writerow((group, dof, count, profile, operation, samples,
                             batch, elapsed, trajectory.duration, digest))


if __name__ == "__main__":
    main()
