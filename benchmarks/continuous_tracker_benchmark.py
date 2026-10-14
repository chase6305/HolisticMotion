"""Measure Python tracker overhead independently of native FK/IK work.

Run with HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python. Optionally pass
--baseline /path/to/previous/srs_tracker.py to compare complete tracking steps.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
from holistic_motion.kinematics import srs_tracker


class FixtureSolver:
    compatible = True
    joint_limits = (np.full(7, -3.0), np.full(7, 3.0))

    def __init__(self, count):
        self.candidates = [np.full(7, index * 0.001) for index in range(count)]
        self.pose = np.eye(4)
        self.matrix = np.eye(6, 7)

    def configuration(self, _joints):
        return SimpleNamespace(shoulder=1, elbow=1, wrist=1, redundancy=0.0)

    def forward(self, _joints):
        return self.pose

    def jacobian(self, _joints):
        return self.matrix

    def solve(self, _target, _seed, _method):
        return self.candidates


def measure(module, count, queries, repeats):
    # These labels are fixture-local, allowing a pure-Python process without
    # importing the native solver enums or mixing Pinocchio runtimes.
    class Tracker(module.SRSContinuousTracker):
        def _seeded_method(self):
            return "seeded"

        def _all_configurations_method(self):
            return "all"

    timings = []
    checksum = None
    for _ in range(repeats):
        tracker = Tracker(FixtureSolver(count), np.zeros(7))
        target = np.eye(4)
        for _ in range(32):
            tracker.solve(target, 0.02)
        total = 0
        started = time.perf_counter()
        for _ in range(queries):
            result = tracker.solve(target, 0.02)
            if not np.array_equal(result.joints, np.zeros(7)):
                raise RuntimeError("tracking result changed")
            total += result.candidate_count
        elapsed = time.perf_counter() - started
        if checksum is not None and total != checksum:
            raise RuntimeError("candidate selection changed between repeats")
        checksum = total
        timings.append(elapsed * 1e6 / queries)
    return {"median_us": statistics.median(timings), "checksum": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.queries, args.repeats) < 1:
        parser.error("queries and repeats must be positive")
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location(
            "holistic_motion.kinematics._baseline_srs_tracker", args.baseline
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
    records = []
    for count in (1, 8):
        record = {"candidates": count}
        if baseline is not None:
            record["baseline"] = measure(baseline, count, args.queries, args.repeats)
        record["current"] = measure(srs_tracker, count, args.queries, args.repeats)
        if baseline is not None:
            if record["baseline"]["checksum"] != record["current"]["checksum"]:
                raise RuntimeError("baseline tracking result changed")
            record["speedup"] = (
                record["baseline"]["median_us"] / record["current"]["median_us"]
            )
        records.append(record)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
