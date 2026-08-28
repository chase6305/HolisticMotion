#!/usr/bin/env python3
"""Animate a TOPPRA-retimed dual-arm trajectory on a URDF robot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _import_holistic_motion():
    """Use the installed package or re-exec inside the local build environment."""
    try:
        import holistic_motion

        return holistic_motion
    except ModuleNotFoundError as error:
        if error.name != "holistic_motion":
            raise
        repository = Path(__file__).resolve().parents[3]
        runner = repository / "scripts/run.sh"
        if not runner.is_file() or os.environ.get("HOLISTICMOTION_DEMO_REEXEC"):
            raise ModuleNotFoundError(
                "HolisticMotion is not importable. Run ./scripts/build.sh first, "
                "then source scripts/activate.sh."
            ) from error
        os.environ["HOLISTICMOTION_DEMO_REEXEC"] = "1"
        os.execv(
            str(runner),
            [str(runner), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        )
        raise RuntimeError("failed to enter the HolisticMotion build environment")


hm = _import_holistic_motion()

from holistic_motion.trajectory import ToppraTrajectory
from holistic_motion.visualization.viser import (
    ViserPerformanceMonitor,
    add_line_segments,
    pose_components,
    tree_topology,
    tree_transforms,
    visual_mesh,
)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=repository / "examples/configs/marvin.json",
    )
    assets = parser.add_mutually_exclusive_group(required=True)
    assets.add_argument("--asset-root", type=Path)
    assets.add_argument(
        "--urdf", "--urdf-path", "--urdf_path", dest="urdf", type=Path
    )
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--amplitude", type=float, default=0.16)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def make_trajectory(robot, profile: dict, args: argparse.Namespace):
    retargeting = profile["retargeting"]
    groups = retargeting["joint_groups"]
    joint_names = groups["left_arm"] + groups["right_arm"]
    seeds = np.asarray(
        profile["arms"]["left"]["seed"] + profile["arms"]["right"]["seed"],
        dtype=float,
    )
    joints = [robot.get_joint(name) for name in joint_names]
    lower = np.asarray([joint.limit.lower for joint in joints])
    upper = np.asarray([joint.limit.upper for joint in joints])
    phase = np.arange(len(joint_names), dtype=float) * 0.61
    delta = args.amplitude * np.sin(phase + 0.35)
    margin = 1e-4

    def bounded(configuration: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(configuration, lower + margin), upper - margin)

    waypoints = np.vstack(
        (
            bounded(seeds),
            bounded(seeds + delta),
            bounded(seeds - 0.75 * delta[::-1]),
            bounded(seeds),
        )
    )
    limits = retargeting["trajectory_limits"]
    max_velocity = np.asarray(limits["max_velocity"] * 2, dtype=float)
    max_acceleration = np.asarray(limits["max_acceleration"] * 2, dtype=float)
    trajectory = ToppraTrajectory(
        waypoints,
        max_velocity,
        max_acceleration,
        grid_size=args.grid_size,
    )
    return trajectory, joint_names, max_velocity, max_acceleration


def link_paths(robot, topology, joint_names, configurations, link_names):
    paths = {name: [] for name in link_names}
    positions = {joint.name: 0.0 for joint in robot.actuated_joints}
    for configuration in configurations:
        positions.update(zip(joint_names, configuration))
        transforms = tree_transforms(robot, positions, topology)
        for name in link_names:
            paths[name].append(transforms[name][:3, 3].copy())
    return {name: np.asarray(points) for name, points in paths.items()}


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0 or args.samples < 2 or args.grid_size < 2:
        raise SystemExit("--rate must be positive; sample and grid sizes must be >= 2")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    urdf_path = (args.urdf or (args.asset_root / profile["urdf"])).resolve()
    robot = hm.Robot(str(urdf_path), True)
    topology = tree_topology(robot)
    trajectory, joint_names, max_velocity, max_acceleration = make_trajectory(
        robot, profile, args
    )
    times, configurations, velocity, acceleration = trajectory.sample_uniform(
        args.samples
    )
    peak_velocity = np.max(np.abs(velocity), axis=0)
    peak_acceleration = np.max(np.abs(acceleration), axis=0)
    if np.any(peak_velocity > max_velocity + 1e-6):
        raise RuntimeError("TOPPRA trajectory violates velocity limits")
    if np.any(peak_acceleration > max_acceleration + 1e-6):
        raise RuntimeError("TOPPRA trajectory violates acceleration limits")
    ee_links = (profile["arms"]["left"]["ee"], profile["arms"]["right"]["ee"])
    paths = link_paths(robot, topology, joint_names, configurations, ee_links)
    print(
        f"validated {robot.name}: {len(joint_names)} timed arm joints, "
        f"duration={trajectory.duration:.3f}s, visuals={sum(len(x.visuals) for x in robot.links)}"
    )
    if args.validate_only:
        return

    try:
        import viser
    except ImportError as error:
        raise SystemExit(
            "install examples with `pip install -e '.[examples]'`"
        ) from error

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    positions = {joint.name: 0.0 for joint in robot.actuated_joints}
    positions.update(zip(joint_names, configurations[0]))
    initial_tree = tree_transforms(robot, positions, topology)
    handles = {}
    for link in robot.links:
        handles[link.name] = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf_path.parent)
            origin = np.asarray(visual.origin)
            pose = initial_tree[link.name] @ origin
            wxyz, position = pose_components(pose)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/visual_{index}",
                mesh,
                scale=tuple(visual.scale),
                wxyz=wxyz,
                position=position,
            )
            handles[link.name].append((handle, origin))

    colors = ((255, 155, 40), (210, 90, 255))
    for link_name, color in zip(ee_links, colors):
        points = paths[link_name]
        add_line_segments(
            server.scene,
            f"/trajectory/{link_name}",
            np.stack((points[:-1], points[1:]), axis=1),
            color,
            line_width=3.0,
        )

    playing = server.gui.add_checkbox("Play", initial_value=args.autoplay)
    looping = server.gui.add_checkbox("Loop", initial_value=args.loop)
    speed = server.gui.add_slider(
        "Playback speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
    )
    timeline = server.gui.add_slider(
        "TOPPRA time [s]",
        min=0.0,
        max=trajectory.duration,
        step=trajectory.duration / 1000.0,
        initial_value=0.0,
    )
    palette = (
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
    )
    chart_legend = {
        "show": True,
        "live": True,
        "isolate": True,
        "markers": {"show": True, "width": 3.0},
    }
    chart_cursor = {
        "show": True,
        "x": True,
        "y": True,
        "sync": {"key": "toppra-robot-trajectory"},
    }
    with server.gui.add_folder("Trajectory curves", expand_by_default=True):
        chart_time = server.gui.add_slider(
            "Chart cursor time [s]",
            min=0.0,
            max=trajectory.duration,
            step=trajectory.duration / 1000.0,
            initial_value=0.0,
            disabled=True,
        )
        tabs = server.gui.add_tab_group()
        half = len(joint_names) // 2
        for side, start, stop in (
            ("Left arm", 0, half),
            ("Right arm", half, len(joint_names)),
        ):
            with tabs.add_tab(side):
                count = stop - start
                labels = tuple(f"J{index + 1}" for index in range(count))
                series = ({"label": "time [s]"},) + tuple(
                    {
                        "label": label,
                        "stroke": palette[index % len(palette)],
                        "width": 1.8,
                        "pxAlign": False,
                        "cap": "round",
                        "points": {"show": False},
                    }
                    for index, label in enumerate(labels)
                )
                joint_table = ["| Curve | URDF joint |", "|:--:|:--|"]
                joint_table.extend(
                    f"| <span style='color:{palette[index]}'>■</span> "
                    f"**{label}** | `{name}` |"
                    for index, (label, name) in enumerate(
                        zip(labels, joint_names[start:stop])
                    )
                )
                server.gui.add_markdown("\n".join(joint_table))
                chart_specs = (
                    (configurations, "Joint position", "position [rad]"),
                    (velocity, "Joint velocity", "velocity [rad/s]"),
                    (
                        acceleration,
                        "Joint acceleration",
                        "acceleration [rad/s²]",
                    ),
                )
                for values, title, y_label in chart_specs:
                    server.gui.add_uplot(
                        (times, *values[:, start:stop].T),
                        series,
                        title=title,
                        height=280,
                        axes=({"label": "time [s]"}, {"label": y_label}),
                        legend=chart_legend,
                        cursor=chart_cursor,
                    )
    status = server.gui.add_markdown("")
    performance = ViserPerformanceMonitor(server.gui, target_fps=args.rate)
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
        chart_time.value = current_time

        q, dq, ddq = trajectory.sample([current_time])
        positions.update(zip(joint_names, q[0]))
        transforms = tree_transforms(robot, positions, topology)
        for link_name, link_handles in handles.items():
            for handle, origin in link_handles:
                pose = transforms[link_name] @ origin
                handle.wxyz, handle.position = pose_components(pose)
        velocity_use = float(np.max(np.abs(dq[0]) / max_velocity))
        acceleration_use = float(np.max(np.abs(ddq[0]) / max_acceleration))
        status.content = (
            "### URDF TOPPRA simulation\n"
            f"- Robot: **{robot.name}**\n"
            f"- Time: **{current_time:.3f} / {trajectory.duration:.3f} s**\n"
            f"- Timed joints: **{len(joint_names)}**\n"
            f"- Velocity utilization: **{100.0 * velocity_use:.1f}%**\n"
            f"- Acceleration utilization: **{100.0 * acceleration_use:.1f}%**"
        )
        performance.record(frame_started)
        time.sleep(max(0.0, 1.0 / args.rate - (time.perf_counter() - frame_started)))


if __name__ == "__main__":
    main()
