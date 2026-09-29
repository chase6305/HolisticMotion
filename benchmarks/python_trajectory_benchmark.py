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

import numpy as np

import holistic_motion as hm


def checksum(result):
    digest = hashlib.sha256()
    for value in result:
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise RuntimeError("benchmark produced non-finite values")
        metadata = (array.shape, array.dtype.str, array.strides,
                    array.flags.owndata, array.flags.writeable)
        digest.update(repr(metadata).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fixtures():
    for profile in ("double_s", "trapezoidal"):
        for dof in (1, 3, 7, 14, 32):
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
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    hm.setup_logging("ERROR", handlers=[])
    writer = csv.writer(sys.stdout)
    writer.writerow(("group", "dof", "waypoints", "profile", "operation",
                     "samples", "batch", "median_us", "duration", "checksum"))
    gc.disable()
    for group, dof, count, profile, trajectory in fixtures():
        query_time = trajectory.duration * 0.37
        cases = [("state", 1, 256, lambda: trajectory.state(query_time))]
        for samples, batch in ((2, 128), (65, 16), (2001, 2), (20001, 1)):
            cases.append((
                "uniform", samples, batch,
                lambda samples=samples: trajectory.sample_uniform(samples),
            ))
        for operation, samples, batch, call in cases:
            elapsed, digest = measure(call, batch, args.repeats)
            writer.writerow((group, dof, count, profile, operation, samples,
                             batch, elapsed, trajectory.duration, digest))


if __name__ == "__main__":
    main()
