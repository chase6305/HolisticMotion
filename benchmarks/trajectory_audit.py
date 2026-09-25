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


def _stress_inputs(rng, trial):
    """Unequal legs, near-collinearity, reversals, and anisotropic limits."""
    dof = int(rng.choice([1, 2, 3, 7, 14, 20, 32]))
    count = int(rng.integers(2, 33))
    scale = 10.0 ** rng.uniform(-3, 1)
    steps = rng.normal(size=(count, dof))
    shape = trial % 4
    if shape == 0:
        steps *= 10.0 ** rng.uniform(-3, 1, (count, 1))
    elif shape in (1, 2):
        direction = rng.normal(size=dof)
        direction /= np.linalg.norm(direction)
        if shape == 1:
            steps = direction * rng.uniform(0.1, 2, (count, 1)) + (
                10.0 ** rng.uniform(-10, -3)
            ) * steps
        else:
            steps = (
                direction
                * rng.choice([-1.0, 1.0], (count, 1))
                * rng.uniform(0.1, 2, (count, 1))
                + 1e-4 * steps
            )
    points = scale * np.cumsum(steps, axis=0)
    velocity, acceleration, jerk = 10.0 ** rng.uniform(-3, 3, (3, dof))
    return {
        "waypoints": points.tolist(),
        "max_velocity": velocity.tolist(),
        "max_acceleration": acceleration.tolist(),
        "max_jerk": jerk.tolist(),
        "blend_tolerance": scale * 0.03,
        # Exercise every shape with both profiles over each block of eight.
        "profile": "double_s" if (trial // 4) % 2 == 0 else "trapezoidal",
    }, scale


def _check(inputs, scale, sample_count, phase_samples=0, check_continuity=False):
    try:
        trajectory = hm.RnTrajectory(**inputs)
    except (ValueError, RuntimeError) as error:
        return "rejected", {"error": str(error)}
    if not np.isfinite(trajectory.duration) or trajectory.duration <= 0.0:
        return "invalid_duration", {}
    breakpoints = trajectory.breakpoints
    grids = [np.linspace(0, trajectory.duration, sample_count), breakpoints]
    if phase_samples:
        grids.extend(
            np.linspace(start, end, phase_samples)
            for start, end in zip(breakpoints[:-1], breakpoints[1:])
        )
    times = np.unique(np.concatenate(grids))
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
    if check_continuity and len(breakpoints) > 2:
        knots = np.asarray(breakpoints)
        # Stay outside knot snapping while approaching from the left, even
        # when phase lengths differ substantially. A half-span cap stays beyond
        # the spline's quarter-span snapping tolerance on very short phases.
        # The next derivative bounds the genuine state change over this
        # interval; the residual allowance
        # accounts for evaluation roundoff rather than physical motion.
        spans = np.diff(knots)[:-1]
        offset = np.minimum(
            0.5 * spans,
            np.maximum(
                1e-8 * spans,
                256 * np.finfo(float).eps * np.abs(knots[1:-1]),
            ),
        )
        try:
            left_times = knots[1:-1] - offset
            left = trajectory.sample(left_times)
            right = trajectory.sample(knots[1:-1])
            # Subtracting from a large clock rounds the requested offset.
            # Bound state changes over the elapsed time actually sampled.
            elapsed = knots[1:-1] - left_times
        except (ValueError, RuntimeError) as error:
            return "evaluation_error", {"stage": "phase_join", "error": str(error)}
        if not all(np.isfinite(value).all() for value in (*left, *right)):
            return "nonfinite", {"stage": "phase_join"}
        allowances = [
            ("max_velocity", 1e-9 * max(1.0, scale)),
            ("max_acceleration", 1e-7 * np.maximum(1.0, inputs["max_velocity"])),
        ]
        if inputs["profile"] == "double_s":
            allowances.append(
                ("max_jerk", 1e-7 * np.maximum(1.0, inputs["max_acceleration"]))
            )
        jumps = [
            float(
                np.max(
                    np.abs(right[order] - left[order])
                    / (np.outer(elapsed, inputs[limit]) + roundoff)
                )
            )
            for order, (limit, roundoff) in enumerate(allowances)
        ]
        if not np.isfinite(jumps).all() or max(jumps) > 1.0:
            return "continuity", {"jump_ratios": jumps}
        try:
            report = trajectory.constraint_report(2)
        except (ValueError, RuntimeError) as error:
            return "evaluation_error", {"stage": "constraint_report", "error": str(error)}
        if not report["velocity_continuous"] or (
            inputs["profile"] == "double_s" and not report["acceleration_continuous"]
        ):
            return "report_continuity", {
                "maximum_velocity_jump": report["maximum_velocity_jump"].tolist(),
                "maximum_acceleration_jump": report["maximum_acceleration_jump"].tolist(),
            }
    return "passed", {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--case-index", type=int)
    parser.add_argument(
        "--distribution",
        choices=("standard", "stress"),
        default="standard",
        help="stress also covers unequal legs, near-collinear paths, and wide limits",
    )
    parser.add_argument(
        "--phase-samples",
        type=int,
        default=0,
        help="also sample each time phase independently (0 disables)",
    )
    parser.add_argument(
        "--check-continuity",
        action="store_true",
        help="check phase-join state changes and native one-sided continuity diagnostics",
    )
    args = parser.parse_args()
    if args.seed < 0 or args.cases < 1 or args.samples < 2:
        parser.error("seed must be non-negative, cases positive, and samples >= 2")
    if args.phase_samples < 0 or args.phase_samples == 1:
        parser.error("phase-samples must be zero or at least 2")
    if args.case_index is not None and args.case_index < 0:
        parser.error("case-index must be non-negative")
    hm.setup_logging("ERROR", handlers=[])
    rng = np.random.default_rng(args.seed)
    counts, failures = {}, []
    started = time.perf_counter()
    count = args.cases if args.case_index is None else args.case_index + 1
    generate = _stress_inputs if args.distribution == "stress" else _inputs
    for trial in range(count):
        inputs, scale = generate(rng, trial)
        if args.case_index is not None and trial != args.case_index:
            continue
        kind, details = _check(
            inputs, scale, args.samples, args.phase_samples, args.check_continuity
        )
        counts[kind] = counts.get(kind, 0) + 1
        # Preserve examples of each failure kind so early construction
        # rejections cannot hide invalid trajectories that were accepted.
        if kind != "passed" and counts[kind] <= 10:
            failures.append(
                {"case_index": trial, "kind": kind, "input": inputs, **details}
            )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "distribution": args.distribution,
                "numpy_version": np.__version__,
                "samples": args.samples,
                "phase_samples": args.phase_samples,
                "check_continuity": args.check_continuity,
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
