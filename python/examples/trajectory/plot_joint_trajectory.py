#!/usr/bin/env python3
"""Generate, validate, and plot a joint trajectory."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))
from _bootstrap import import_holistic_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dof", type=int, choices=range(1, 33), default=7)
    parser.add_argument("--waypoints", type=Path,
                        help="optional headerless CSV with one waypoint per row")
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--blend", type=float, default=0.02,
                        help="joint-space waypoint blend tolerance [rad]")
    parser.add_argument("--profile", choices=("double_s", "trapezoidal"),
                        default="double_s")
    parser.add_argument("--max-velocity", type=float, default=1.2,
                        help="symmetric per-joint velocity limit [rad/s]")
    parser.add_argument("--max-acceleration", type=float, default=2.5,
                        help="symmetric per-joint acceleration limit [rad/s²]")
    parser.add_argument("--max-jerk", type=float, default=8.0,
                        help="symmetric per-joint jerk limit [rad/s³]")
    parser.add_argument("--output", type=Path,
                        default=Path("trajectory_profiles.png"))
    parser.add_argument("--data-output", type=Path,
                        help="optionally save sampled states as a CSV file")
    parser.add_argument("--show", action="store_true",
                        help="open an interactive matplotlib window")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate construction and limits without plotting")
    return parser.parse_args()


def example_waypoints(dof: int) -> np.ndarray:
    waypoints = np.zeros((6, dof))
    if dof == 1:
        waypoints[:, 0] = np.deg2rad([0, 12, 28, 43, 61, 75])
        return waypoints
    primary = np.deg2rad([
        [0, 0, 0], [18, -12, 8], [35, 5, -15],
        [12, 24, 10], [-8, 8, 20], [0, 0, 0],
    ])
    waypoints[:, :min(3, dof)] = primary[:, :min(3, dof)]
    for joint in range(3, dof):
        scale = 1.0 / (1.0 + 0.08 * (joint - 3))
        phase = 0.35 * joint
        waypoints[:, joint] = scale * np.deg2rad(15.0) * np.sin(
            np.linspace(0.0, 2.0 * np.pi, len(waypoints)) + phase
        )
        waypoints[-1, joint] = waypoints[0, joint]
    return waypoints


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")

    hm = import_holistic_motion()
    if args.waypoints:
        try:
            waypoints = np.loadtxt(args.waypoints, delimiter=",", ndmin=2)
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"failed to read waypoint CSV {args.waypoints}: {error}"
            ) from error
        if waypoints.shape[0] < 2 or waypoints.shape[1] != args.dof:
            raise SystemExit(
                f"--waypoints must have shape (M, {args.dof}), M >= 2; "
                f"received {waypoints.shape}"
            )
    else:
        waypoints = example_waypoints(args.dof)
    for name in ("max_velocity", "max_acceleration", "max_jerk"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    velocity_limit = np.full(args.dof, args.max_velocity)
    acceleration_limit = np.full(args.dof, args.max_acceleration)
    jerk_limit = np.full(args.dof, args.max_jerk)
    construction_start = time.perf_counter()
    trajectory = hm.RnTrajectory(
        waypoints, velocity_limit, acceleration_limit, jerk_limit,
        blend_tolerance=args.blend, profile=args.profile,
    )
    construction_seconds = time.perf_counter() - construction_start

    breakpoints = np.asarray(trajectory.breakpoints)
    segment_lengths = np.diff(breakpoints)
    floating_offset = (
        256.0 * np.finfo(float).eps * max(1.0, trajectory.duration)
    )
    left_limits = breakpoints[1:-1] - np.minimum(
        0.5 * segment_lengths[:-1],
        np.maximum(floating_offset, 1e-9 * segment_lengths[:-1]),
    )
    times = np.unique(np.concatenate((
        np.linspace(0.0, trajectory.duration, args.samples),
        breakpoints,
        left_limits,
    )))
    sampling_start = time.perf_counter()
    position, velocity, acceleration, jerk = trajectory.sample(times)
    sampling_seconds = time.perf_counter() - sampling_start
    report = trajectory.constraint_report(samples=args.samples)

    if args.validate_only:
        print(f"Rn trajectory: dof={trajectory.dof}, "
              f"duration={trajectory.duration:.3f} s, "
              f"states={len(times)}, limits="
              f"{'OK' if report['within_limits'] else 'VIOLATION'}")
        return

    try:
        if not args.show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required; install the example dependencies with "
            "`python3 -m pip install -e '.[examples]'`"
        ) from error

    colors = plt.colormaps["tab10"](np.linspace(0, 1, args.dof))
    figure, axes = plt.subplots(5, 1, figsize=(11, 15), constrained_layout=True)
    if args.dof == 1:
        axes[0].plot(times, np.rad2deg(position[:, 0]), color="tab:blue",
                     linewidth=1.8, label="smoothed path")
        axes[0].plot(np.linspace(0.0, trajectory.duration,
                                len(trajectory.waypoints)),
                     np.rad2deg(trajectory.waypoints[:, 0]), color="black",
                     marker="o", linestyle="--", linewidth=0.8,
                     label="input waypoints")
    else:
        axes[0].plot(np.rad2deg(position[:, 0]), np.rad2deg(position[:, 1]),
                     color="tab:blue", linewidth=1.8, label="smoothed path")
        axes[0].plot(np.rad2deg(trajectory.waypoints[:, 0]),
                     np.rad2deg(trajectory.waypoints[:, 1]), color="black",
                     marker="o", linestyle="--", linewidth=0.8,
                     label="input waypoints")
    for joint, color in enumerate(colors):
        label = f"q{joint + 1}"
        axes[1].plot(times, np.rad2deg(position[:, joint]), color=color, label=label)
        axes[2].plot(times, velocity[:, joint], color=color)
        axes[3].plot(times, acceleration[:, joint], color=color)
        axes[4].plot(times, jerk[:, joint], color=color)

    waypoint_times = breakpoints
    axes[0].grid(alpha=0.25)
    for axis in axes[1:]:
        for breakpoint_time in waypoint_times[1:-1]:
            axis.axvline(breakpoint_time, color="0.75", linewidth=0.7,
                         linestyle=":")
        axis.grid(alpha=0.25)
    axes[0].set_xlabel("time [s]" if args.dof == 1 else "q1 [deg]")
    axes[0].set_ylabel("q1 [deg]" if args.dof == 1 else "q2 [deg]")
    axes[0].set_title("Joint-space path")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("position [deg]")
    axes[1].set_title(f"{trajectory.profile} time parameterization")
    axes[1].legend(ncol=min(args.dof, 7), loc="upper right")
    for axis, limit, ylabel in (
        (axes[2], velocity_limit, "velocity [rad/s]"),
        (axes[3], acceleration_limit, "acceleration [rad/s²]"),
        (axes[4], jerk_limit, "jerk [rad/s³]"),
    ):
        axis.axhline(np.max(limit), color="black", linestyle="--", linewidth=0.8)
        axis.axhline(-np.max(limit), color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel("time [s]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"duration: {trajectory.duration:.3f} s")
    print(f"path length: {trajectory.path_length:.3f} rad")
    print(f"compute: construction={construction_seconds * 1e3:.3f} ms, "
          f"sampling={sampling_seconds * 1e3:.3f} ms "
          f"({len(times) / sampling_seconds:.0f} states/s)")
    print("sampled peaks: "
          f"velocity={np.max(report['peak_velocity']):.3f} rad/s, "
          f"acceleration={np.max(report['peak_acceleration']):.3f} rad/s², "
          f"jerk={np.max(report['peak_jerk']):.3f} rad/s³")
    print(f"maximum constraint utilization: "
          f"{100.0 * report['maximum_utilization']:.1f}% "
          f"({'OK' if report['within_limits'] else 'VIOLATION'})")
    print("breakpoint continuity: "
          f"velocity={'OK' if report['velocity_continuous'] else 'JUMP'}, "
          f"acceleration="
          f"{'OK' if report['acceleration_continuous'] else 'JUMP'}")
    print(f"plot: {args.output.resolve()}")
    if args.data_output:
        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        columns = [times[:, None], position, velocity, acceleration, jerk]
        names = ["time"] + [
            f"{quantity}_q{joint + 1}"
            for quantity in ("position", "velocity", "acceleration", "jerk")
            for joint in range(args.dof)
        ]
        np.savetxt(args.data_output, np.hstack(columns), delimiter=",",
                   header=",".join(names), comments="")
        print(f"data: {args.data_output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
