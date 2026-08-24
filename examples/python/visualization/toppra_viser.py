#!/usr/bin/env python3
"""Interactively visualize TOPPRA path timing and constraint utilization."""

from __future__ import annotations

import argparse
import time

import numpy as np
from holistic_motion.trajectory import ToppraTrajectory

COLORS = ((70, 150, 240), (245, 150, 55), (90, 200, 120))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def make_trajectory(grid_size: int) -> ToppraTrajectory:
    return ToppraTrajectory(
        [
            [0.0, 0.0, 0.0],
            [0.35, -0.30, 0.25],
            [0.75, 0.15, -0.35],
            [1.0, 0.50, 0.10],
        ],
        max_velocity=[1.0, 0.8, 0.9],
        max_acceleration=[2.0, 1.5, 1.8],
        grid_size=grid_size,
    )


def segments(points: np.ndarray) -> np.ndarray:
    return np.stack((points[:-1], points[1:]), axis=1)


def chart_points(times: np.ndarray, values: np.ndarray, z: float) -> np.ndarray:
    """Map a time series to an x/y chart embedded in the 3D scene."""
    x = 1.8 * times / times[-1] - 0.9
    return np.column_stack((x, values, np.full_like(x, z)))


def validate(
    trajectory: ToppraTrajectory,
    velocity: np.ndarray,
    acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    peak_velocity = np.max(np.abs(velocity), axis=0)
    peak_acceleration = np.max(np.abs(acceleration), axis=0)
    if np.any(peak_velocity > trajectory.max_velocity + 1e-6):
        raise RuntimeError("sampled trajectory violates velocity limits")
    if np.any(peak_acceleration > trajectory.max_acceleration + 1e-6):
        raise RuntimeError("sampled trajectory violates acceleration limits")
    print(f"duration={trajectory.duration:.6f} s")
    print(f"peak velocity={peak_velocity}")
    print(f"peak acceleration={peak_acceleration}")
    return peak_velocity, peak_acceleration


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0 or args.samples < 2 or args.grid_size < 2:
        raise SystemExit("--rate must be positive; sample and grid sizes must be >= 2")
    trajectory = make_trajectory(args.grid_size)
    times, position, velocity, acceleration = trajectory.sample_uniform(args.samples)
    peak_velocity, peak_acceleration = validate(trajectory, velocity, acceleration)
    if args.validate_only:
        return

    try:
        import viser
        from holistic_motion.visualization.viser import (
            ViserPerformanceMonitor,
            add_line_segments,
        )
    except ImportError as error:
        raise SystemExit(
            "install examples with `pip install -e '.[examples]'`"
        ) from error

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    add_line_segments(
        server.scene,
        "/toppra/joint_path",
        segments(position),
        (235, 235, 235),
        line_width=3.0,
    )
    path_cursor = server.scene.add_frame(
        "/toppra/current", axes_length=0.09, axes_radius=0.006
    )

    velocity_cursors = []
    acceleration_cursors = []
    for joint, color in enumerate(COLORS):
        velocity_points = chart_points(times, velocity[:, joint], 1.1 + 0.08 * joint)
        acceleration_points = chart_points(
            times, acceleration[:, joint] * 0.35, 1.8 + 0.08 * joint
        )
        add_line_segments(
            server.scene,
            f"/charts/velocity/joint_{joint + 1}",
            segments(velocity_points),
            color,
            line_width=2.0,
        )
        add_line_segments(
            server.scene,
            f"/charts/acceleration/joint_{joint + 1}",
            segments(acceleration_points),
            color,
            line_width=2.0,
        )
        velocity_cursors.append(
            server.scene.add_frame(
                f"/charts/velocity/cursor_{joint + 1}",
                axes_length=0.045,
                axes_radius=0.003,
            )
        )
        acceleration_cursors.append(
            server.scene.add_frame(
                f"/charts/acceleration/cursor_{joint + 1}",
                axes_length=0.045,
                axes_radius=0.003,
            )
        )

    playing = server.gui.add_checkbox("Play", initial_value=args.autoplay)
    looping = server.gui.add_checkbox("Loop", initial_value=args.loop)
    speed = server.gui.add_slider(
        "Playback speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
    )
    timeline = server.gui.add_slider(
        "Time [s]",
        min=0.0,
        max=trajectory.duration,
        step=trajectory.duration / 1000.0,
        initial_value=0.0,
    )
    status = server.gui.add_markdown("")
    monitor = ViserPerformanceMonitor(server.gui, target_fps=args.rate)
    current_time = 0.0
    previous = time.perf_counter()

    while True:
        frame_started = time.perf_counter()
        elapsed = frame_started - previous
        previous = frame_started
        if playing.value:
            current_time += elapsed * float(speed.value)
            if current_time > trajectory.duration:
                if looping.value:
                    current_time %= trajectory.duration
                else:
                    current_time = trajectory.duration
                    playing.value = False
            timeline.value = current_time
        else:
            current_time = float(timeline.value)

        state_position, state_velocity, state_acceleration = trajectory.sample(
            [current_time]
        )
        path_cursor.position = state_position[0]
        chart_x = 1.8 * current_time / trajectory.duration - 0.9
        for joint in range(trajectory.dof):
            velocity_cursors[joint].position = np.array(
                [chart_x, state_velocity[0, joint], 1.1 + 0.08 * joint]
            )
            acceleration_cursors[joint].position = np.array(
                [chart_x, state_acceleration[0, joint] * 0.35, 1.8 + 0.08 * joint]
            )
        velocity_use = np.max(np.abs(state_velocity[0]) / trajectory.max_velocity)
        acceleration_use = np.max(
            np.abs(state_acceleration[0]) / trajectory.max_acceleration
        )
        status.content = (
            "### TOPPRA state\n"
            f"- Time: **{current_time:.3f} / {trajectory.duration:.3f} s**\n"
            f"- Velocity utilization: **{100.0 * velocity_use:.1f}%**\n"
            f"- Acceleration utilization: **{100.0 * acceleration_use:.1f}%**\n"
            f"- Global peaks: **v={peak_velocity.round(3)}**, "
            f"**a={peak_acceleration.round(3)}**"
        )
        monitor.record(frame_started)
        time.sleep(max(0.0, 1.0 / args.rate - (time.perf_counter() - frame_started)))


if __name__ == "__main__":
    main()
