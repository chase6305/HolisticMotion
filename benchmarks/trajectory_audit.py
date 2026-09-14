"""Audit deterministic native trajectories and report reproducible failures.

Run against a built package with PYTHONPATH=build/install. This is a diagnostic,
not a timing benchmark or a proof of continuous constraint satisfaction. Any
rejected input or sampled invariant failure produces exit status 1. Reproduce
one trial with the same --seed and --case-index reported in the JSON output.
"""

import argparse
import json
import time

import holistic_motion as hm
import numpy as np


def _inputs(rng, trial):
    dof = int(rng.choice([1, 2, 3, 6, 7, 14, 20]))
    count = int(rng.integers(2, 10))
    scale = 10.0 ** rng.uniform(-3, 1)
    points = np.cumsum(rng.normal(size=(count, dof)), axis=0) * scale
    velocity, acceleration, jerk = 10.0 ** rng.uniform(-1, 1, (3, dof))
    return {
        "waypoints": points.tolist(),
        "max_velocity": velocity.tolist(),
        "max_acceleration": acceleration.tolist(),
        "max_jerk": jerk.tolist(),
        "blend_tolerance": 0.0 if trial % 3 == 0 else scale * 0.03,
        "profile": "double_s" if trial % 2 == 0 else "trapezoidal",
    }, scale


def _check(inputs, scale, sample_count):
    try:
        trajectory = hm.RnTrajectory(**inputs)
    except (ValueError, RuntimeError) as error:
        return "rejected", {"error": str(error)}
    if not np.isfinite(trajectory.duration) or trajectory.duration <= 0.0:
        return "invalid_duration", {}
    times = np.unique(
        np.concatenate(
            (
                np.linspace(0, trajectory.duration, sample_count),
                trajectory.breakpoints,
            )
        )
    )
    try:
        q, dq, ddq, dddq = trajectory.sample(times)
    except (ValueError, RuntimeError) as error:
        return "evaluation_error", {"error": str(error)}
    if not all(np.isfinite(value).all() for value in (q, dq, ddq, dddq)):
        return "nonfinite", {}
    endpoints = np.asarray(inputs["waypoints"])[[0, -1]]
    if np.max(np.abs(q[[0, -1]] - endpoints)) > 1e-7 * max(1.0, scale):
        return "endpoint", {}
    ratios = [
        float(np.max(np.abs(dq) / inputs["max_velocity"])),
        float(np.max(np.abs(ddq) / inputs["max_acceleration"])),
    ]
    if inputs["profile"] == "double_s":
        ratios.append(float(np.max(np.abs(dddq) / inputs["max_jerk"])))
    if max(ratios) > 1.0001:
        return "limits", {"ratios": ratios}
    return "passed", {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    if args.seed < 0 or args.cases < 1 or args.samples < 2:
        parser.error("seed must be non-negative, cases positive, and samples >= 2")
    if args.case_index is not None and args.case_index < 0:
        parser.error("case-index must be non-negative")
    hm.setup_logging("ERROR", handlers=[])
    rng = np.random.default_rng(args.seed)
    counts, failures = {}, []
    started = time.perf_counter()
    count = args.cases if args.case_index is None else args.case_index + 1
    for trial in range(count):
        inputs, scale = _inputs(rng, trial)
        if args.case_index is not None and trial != args.case_index:
            continue
        kind, details = _check(inputs, scale, args.samples)
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "passed" and len(failures) < 10:
            failures.append(
                {"case_index": trial, "kind": kind, "input": inputs, **details}
            )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "numpy_version": np.__version__,
                "samples": args.samples,
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
