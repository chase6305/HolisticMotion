#!/usr/bin/env python3
"""Plan a 2-D point robot around a wall, then retime the collision-free path."""

from __future__ import annotations

import argparse
import math

import holistic_motion as hm
import numpy as np

WALL = (-0.2, 0.2, -0.75, 0.75)  # xmin, xmax, ymin, ymax
START = np.array([-0.8, 0.0])
GOAL = np.array([0.8, 0.0])
EDGE_RESOLUTION = 0.02
PLANNING_MARGIN = 0.03


def collision_free(state, margin=0.0) -> bool:
    """Treat the robot as a point in a square configuration space."""

    x, y = state
    xmin, xmax, ymin, ymax = WALL
    return not (
        xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin
    )


def positive_int(value: str) -> int:
    result = int(value)
    if result < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return result


def plan(seed: int):
    # Edge checking is discrete. A margin larger than the edge resolution keeps
    # the final polyline outside the physical obstacle between samples as well.
    planner = hm.SamplingPlanner(
        [-1.0, -1.0],
        [1.0, 1.0],
        lambda state: collision_free(state, PLANNING_MARGIN),
    )
    # Unit weights make the reported path length equal the plot's Euclidean length.
    planner.set_joint_weights([1.0, 1.0])
    options = hm.PlanningOptions()
    options.algorithm = hm.SamplingAlgorithm.RRT_CONNECT
    options.timeout_seconds = 1.0
    options.max_iterations = 5000
    options.extension_range = 0.15
    options.edge_resolution = EDGE_RESOLUTION
    options.shortcut_attempts = 100
    options.random_seed = seed
    return planner.plan(START, GOAL, options)


def edge_collision_free(start, goal) -> bool:
    """Recheck a complete edge using the planner's margin and resolution."""

    delta = goal - start
    intervals = max(1, math.ceil(np.linalg.norm(delta) / EDGE_RESOLUTION))
    return all(
        collision_free(start + delta * (index / intervals), PLANNING_MARGIN)
        for index in range(intervals + 1)
    )


def prune_line_of_sight(path) -> np.ndarray:
    """Keep the farthest visible future waypoint at each retained waypoint."""

    path = np.asarray(path, dtype=float)
    retained = [path[0]]
    current = 0
    while current + 1 < len(path):
        for candidate in range(len(path) - 1, current, -1):
            if edge_collision_free(path[current], path[candidate]):
                break
        retained.append(path[candidate])
        current = candidate
    return np.asarray(retained)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=positive_int, default=401)
    parser.add_argument(
        "--profile", choices=("double_s", "trapezoidal"), default="double_s"
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    result = plan(args.seed)
    if not result.success:
        print(f"planning failed: {result.status} ({result.message})")
        return 1

    planned_path = np.asarray(result.path, dtype=float)
    waypoints = prune_line_of_sight(planned_path)
    # A zero blend tolerance preserves the collision-checked polyline exactly.
    trajectory = hm.RnTrajectory(
        waypoints,
        max_velocity=[0.8, 0.8],
        max_acceleration=[1.5, 1.5],
        max_jerk=[4.0, 4.0],
        blend_tolerance=0.0,
        profile=args.profile,
    )
    _times, positions, velocities, accelerations, jerks = trajectory.sample_uniform(
        args.samples
    )
    if not all(collision_free(position) for position in positions):
        raise RuntimeError("retimed samples left the collision-free planned polyline")
    report = trajectory.constraint_report(args.samples)
    if not report["within_limits"]:
        raise RuntimeError("retimed path violates its configured derivative limits")

    stats = result.statistics
    execution_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    print("classic point-robot motion planning")
    print(f"  planner: RRT_CONNECT, seed={args.seed}")
    print(
        f"  obstacle margin: {PLANNING_MARGIN:.3f} "
        f"(edge resolution {EDGE_RESOLUTION:.3f})"
    )
    print(
        f"  path: {len(planned_path)} points/{stats.final_path_length:.4f} units planned "
        f"-> {len(waypoints)} points/{execution_length:.4f} units executed"
    )
    print(f"  planning validity checks: {stats.collision_checks}")
    print(f"  timing: {args.profile}, duration={trajectory.duration:.4f} s")
    print(
        "  sampled peaks: "
        f"|v|={np.max(np.abs(velocities), axis=0)}, "
        f"|a|={np.max(np.abs(accelerations), axis=0)}, "
        f"|j|={np.max(np.abs(jerks), axis=0)}"
    )

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        figure, axis = plt.subplots()
        xmin, xmax, ymin, ymax = WALL
        axis.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                color="tab:red",
                alpha=0.3,
                label="obstacle",
            )
        )
        axis.plot(planned_path[:, 0], planned_path[:, 1], ".--", label="RRT waypoints")
        axis.plot(waypoints[:, 0], waypoints[:, 1], "o--", label="execution waypoints")
        axis.plot(positions[:, 0], positions[:, 1], "-", label="timed samples")
        axis.scatter(*START, marker="s", color="tab:green", label="start")
        axis.scatter(*GOAL, marker="*", color="tab:blue", s=120, label="goal")
        axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xlabel="q1", ylabel="q2")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
