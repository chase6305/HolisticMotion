"""Audit native continuity reports on deterministic, rescaled stress paths.

This checks construction and report flags, not dense trajectory invariants.
Use trajectory_audit.py for independent sampled limit and continuity checks.
Every trial scales coordinates, blend tolerance, and all derivative limits by
the same random factor from 1e-2 to 1e4. NumPy version is part of reproduction.
"""

import argparse
import json
import time

import holistic_motion as hm
import numpy as np
from trajectory_audit import _rescaled_stress_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    if args.seed < 0 or args.cases < 1:
        parser.error("seed must be non-negative and cases positive")
    if args.case_index is not None and args.case_index < 0:
        parser.error("case-index must be non-negative")
    hm.setup_logging("ERROR", handlers=[])
    rng = np.random.default_rng(args.seed)
    counts, failures = {}, []
    started = time.perf_counter()
    trials = args.cases if args.case_index is None else args.case_index + 1
    for trial in range(trials):
        inputs, scale = _rescaled_stress_inputs(rng, trial)
        if args.case_index is not None and trial != args.case_index:
            continue
        kind, details = "passed", {}
        try:
            trajectory = hm.RnTrajectory(**inputs)
        except (ValueError, RuntimeError) as error:
            kind, details = "rejected", {"error": str(error)}
        else:
            try:
                report = trajectory.constraint_report(3)
            except (ValueError, RuntimeError) as error:
                kind, details = "report_error", {"error": str(error)}
            else:
                if not report["velocity_continuous"] or (
                    inputs["profile"] == "double_s"
                    and not report["acceleration_continuous"]
                ):
                    kind = "report_continuity"
                    details = {
                        "report": {
                            key: value.tolist() if hasattr(value, "tolist") else value
                            for key, value in report.items()
                        }
                    }
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "passed" and counts[kind] <= 10:
            failures.append(
                {
                    "case_index": trial,
                    "kind": kind,
                    "scale": scale,
                    "input": inputs,
                    **details,
                }
            )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "numpy_version": np.__version__,
                "report_samples": 3,
                "coordinate_scale_exponents": [-2, 4],
                "elapsed_seconds": time.perf_counter() - started,
                "counts": counts,
                "first_failures": failures,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return int(any(kind != "passed" for kind in counts))


if __name__ == "__main__":
    raise SystemExit(main())
