#!/usr/bin/env python3
"""Visualize mobile-base, R^n joint, and Cartesian line trajectories."""

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
    parser.add_argument("--profile", choices=("double_s", "trapezoidal"),
                        default="double_s")
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0],
                     [-sine, 0.0, cosine]])


def make_trajectories(hm, dof: int, profile: str):
    base_points = np.array([[0.0, 0.0, 0.0], [0.45, 0.15, 0.35],
                            [0.85, -0.1, -0.25], [1.2, 0.1, 0.0]])
    base = hm.BaseTrajectory(
        base_points, np.array([0.6, 0.6, 0.8]),
        np.array([1.2, 1.2, 1.5]), np.array([4.0, 4.0, 5.0]),
        blend_tolerance=0.015, profile=profile,
    )

    joint_points = np.zeros((4, dof))
    phases = np.arange(dof) * 0.31
    joint_points[1] = 0.35 * np.sin(phases + 0.4)
    joint_points[2] = 0.45 * np.sin(phases - 0.7)
    joint_points[3] = 0.15 * np.sin(phases + 1.1)
    joint_limits = np.ones(dof)
    joints = hm.RnTrajectory(
        joint_points, joint_limits, 2.0 * joint_limits,
        6.0 * joint_limits, blend_tolerance=0.005, profile=profile,
    )

    start, end = np.eye(4), np.eye(4)
    start[:3, 3] = [-0.2, 0.8, 0.35]
    end[:3, :3] = rotation_y(np.deg2rad(55.0))
    end[:3, 3] = [0.35, 0.8, 0.65]
    cartesian = hm.CartesianLineTrajectory(
        start, end, np.array([0.5] * 3 + [1.0] * 3),
        np.array([1.2] * 3 + [2.0] * 3),
        np.array([4.0] * 3 + [6.0] * 3), profile=profile,
    )
    return base, joints, cartesian


def planar_chain(configuration: np.ndarray) -> np.ndarray:
    """Map arbitrary joint state to a compact planar serial-chain display."""
    segment = min(0.16, 1.25 / max(1, len(configuration)))
    angles = np.cumsum(configuration)
    points = np.zeros((len(configuration) + 1, 3))
    points[0] = [0.0, -0.8, 0.25]
    for index, angle in enumerate(angles, start=1):
        points[index] = points[index - 1] + [
            segment * np.cos(angle), 0.0, segment * np.sin(angle)
        ]
    return points


def validate(base, joints, cartesian) -> None:
    for name, trajectory in (("base", base), ("Rn", joints),
                             ("Cartesian", cartesian)):
        report = trajectory.constraint_report(samples=501)
        if not report["within_limits"]:
            raise RuntimeError(f"{name} trajectory violates its limits")
        print(f"{name}: duration={trajectory.duration:.3f} s, limits=OK")


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0:
        raise SystemExit("--rate must be positive")
    hm = import_holistic_motion()
    base, joints, cartesian = make_trajectories(hm, args.dof, args.profile)
    validate(base, joints, cartesian)
    if args.validate_only:
        return

    try:
        import viser
        from _viser_utils import (ViserPerformanceMonitor, add_line_segments,
                                  pose_components)
    except ImportError as error:
        raise SystemExit("install examples with `pip install -e '.[examples]'`") \
            from error

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    base_frame = server.scene.add_frame("/base", axes_length=0.25,
                                        axes_radius=0.012)
    tcp_frame = server.scene.add_frame("/tcp", axes_length=0.2,
                                       axes_radius=0.01)
    initial_chain = planar_chain(joints.position(0.0))
    chain_handle = add_line_segments(
        server.scene, "/joints", np.stack((initial_chain[:-1],
                                            initial_chain[1:]), axis=1),
        (70, 150, 240), line_width=4.0,
    )
    cart_samples = cartesian.sample_uniform(80)[1][:, :3, 3]
    add_line_segments(
        server.scene, "/tcp/path",
        np.stack((cart_samples[:-1], cart_samples[1:]), axis=1),
        (245, 150, 55), line_width=2.0,
    )

    duration = max(base.duration, joints.duration, cartesian.duration)
    playing = server.gui.add_checkbox("Play", initial_value=args.autoplay)
    looping = server.gui.add_checkbox("Loop", initial_value=args.loop)
    timeline = server.gui.add_slider("Time [s]", min=0.0, max=duration,
                                     step=duration / 1000.0,
                                     initial_value=0.0)
    status = server.gui.add_markdown("")
    monitor = ViserPerformanceMonitor(server.gui, target_fps=args.rate)
    current_time = 0.0
    previous = time.perf_counter()

    while True:
        frame_started = time.perf_counter()
        elapsed = frame_started - previous
        previous = frame_started
        if playing.value:
            current_time += elapsed
            if current_time > duration:
                if looping.value:
                    current_time %= duration
                else:
                    current_time = duration
                    playing.value = False
            timeline.value = current_time
        else:
            current_time = float(timeline.value)

        base_state = base.position(min(current_time, base.duration))
        yaw = float(base_state[2])
        base_frame.position = np.array([base_state[0], base_state[1], 0.05])
        base_frame.wxyz = np.array([np.cos(yaw / 2), 0.0, 0.0,
                                    np.sin(yaw / 2)])
        chain = planar_chain(joints.position(min(current_time,
                                                 joints.duration)))
        chain_handle.points = np.stack((chain[:-1], chain[1:]), axis=1)
        transform = cartesian.position(min(current_time, cartesian.duration))
        tcp_frame.wxyz, tcp_frame.position = pose_components(transform)
        status.content = (
            "### Trajectory state\n"
            f"- Space: base R3 + joints R{args.dof} + TCP SE(3)\n"
            f"- Profile: **{args.profile}**\n"
            f"- Time: **{current_time:.3f} / {duration:.3f} s**"
        )
        monitor.record(frame_started)
        time.sleep(max(0.0, 1.0 / args.rate -
                       (time.perf_counter() - frame_started)))


if __name__ == "__main__":
    main()
