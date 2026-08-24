#!/usr/bin/env python3
"""Show two independently solved arm trajectories on the complete robot."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from _bootstrap import import_holistic_motion
from _viser_utils import (
    ViserPerformanceMonitor,
    add_line_segments,
    pose_components,
    visual_mesh,
)


hm = import_holistic_motion()

DEFAULT_URDF = Path(
    "/home/ubuntu/workspace/chase/HumanoidAssets/"
    "Marvin_M6_S_CCS_696_V4.0/robot_with_ee.urdf"
)
RIGHT_ARM_SEED = np.array(
    [-0.51478342, -0.48319332, 0.47513737, -1.72500699,
     2.74124467, 0.75606588, 0.68688189]
)
LEFT_ARM_SEED = RIGHT_ARM_SEED * np.array([-1, 1, -1, 1, -1, 1, -1])


def _motion(joint, value: float) -> np.ndarray:
    result = np.eye(4)
    axis = np.asarray(joint.axis, dtype=float)
    if joint.joint_type in (hm.JointType.REVOLUTE, hm.JointType.CONTINUOUS):
        length = np.linalg.norm(axis)
        if length > 0.0:
            result = trimesh.transformations.rotation_matrix(
                value, axis / length
            )
    elif joint.joint_type == hm.JointType.PRISMATIC:
        result[:3, 3] = axis * value
    return result


def _tree_topology(robot) -> list:
    """Return joints in parent-before-child order for fast repeated FK."""
    ordered = []
    pending = [robot.root_link_name]
    while pending:
        parent_name = pending.pop()
        for joint_name in robot.get_link(parent_name).child_joints:
            joint = robot.get_joint(joint_name)
            ordered.append(joint)
            pending.append(joint.child_link)
    if len(ordered) != len(robot.joints):
        raise RuntimeError("URDF tree contains disconnected links")
    return ordered


def _tree_transforms(
    robot, positions: dict[str, float], topology=None
) -> dict[str, np.ndarray]:
    transforms = {robot.root_link_name: np.eye(4)}
    for joint in topology or _tree_topology(robot):
        value = positions.get(joint.name, 0.0)
        if joint.mimic_joint:
            value = (positions.get(joint.mimic_joint, 0.0)
                     * joint.mimic_multiplier + joint.mimic_offset)
        transforms[joint.child_link] = (
            transforms[joint.parent_link]
            @ np.asarray(joint.origin)
            @ _motion(joint, value)
        )
    return transforms


def _chain_joint_names(robot, base: str, tip: str) -> list[str]:
    names = []
    current = tip
    while current != base:
        link = robot.get_link(current)
        if link is None or not link.parent_joint:
            raise ValueError(f"{tip!r} is not below {base!r}")
        joint = robot.get_joint(link.parent_joint)
        if joint.joint_type != hm.JointType.FIXED:
            names.append(joint.name)
        current = joint.parent_link
    names.reverse()
    return names


def _solve_line(
    solver, base_in_torso, seed, axis, distance, samples,
    center_offset=0.0, direction=1.0,
):
    center = base_in_torso @ solver.forward(seed)
    displacement = np.zeros(3)
    displacement["xyz".index(axis)] = direction
    parameters = np.linspace(
        center_offset - distance / 2.0,
        center_offset + distance / 2.0,
        samples,
    )
    inverse_base = np.linalg.inv(base_in_torso)
    joints = [None] * samples
    errors = [0.0] * samples

    def solve_sample(index, q):
        parameter = parameters[index]
        target = center.copy()
        target[:3, 3] += parameter * displacement
        try:
            q = np.asarray(solver.inverse(inverse_base @ target, q))
        except ValueError as error:
            raise RuntimeError(
                f"IK failed at line sample {index + 1}/{samples}, "
                f"offset={parameter:.4f} m"
            ) from error
        joints[index] = q.copy()
        actual = base_in_torso @ solver.forward(q)
        errors[index] = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        return q

    center_index = int(np.argmin(np.abs(parameters)))
    center_solution = solve_sample(center_index, seed.copy())
    q = center_solution.copy()
    for index in range(center_index - 1, -1, -1):
        q = solve_sample(index, q)
    q = center_solution.copy()
    for index in range(center_index + 1, samples):
        q = solve_sample(index, q)

    joint_path = np.asarray(joints)
    if np.max(np.abs(np.diff(joint_path, axis=0))) > 0.25:
        raise RuntimeError("IK branch jump detected along line")
    points = center[:3, 3] + parameters[:, None] * displacement
    return joint_path, points, max(errors)


def prepare_demo(urdf_path: Path, distance: float, samples: int):
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    robot = hm.Robot(str(urdf_path))
    left = robot.create_kinematics("left_arm_base", "left_ee")
    right = robot.create_kinematics("right_arm_base", "right_ee")
    if left is None or right is None:
        raise RuntimeError("could not construct both arm chains from the URDF")

    topology = _tree_topology(robot)
    zero_tree = _tree_transforms(robot, {}, topology)
    torso = zero_tree["upper_body_base"]
    torso_inverse = np.linalg.inv(torso)
    left_mount = torso_inverse @ zero_tree["left_arm_base"]
    right_mount = torso_inverse @ zero_tree["right_arm_base"]
    # Offsets place each 35 cm segment inside the shared reachable workspace.
    # Mirrored Y directions make both hands move inward/outward together.
    full_length_configuration = {
        "z": (-0.0950, 1.0, 1.0),
        "y": (-0.1450, 1.0, -1.0),
        "x": (-0.0415, 1.0, 1.0),
    }
    offset_scale = np.clip((distance - 0.10) / 0.25, 0.0, 1.0)
    trajectories = {}
    for axis, (offset, left_direction, right_direction) in (
        full_length_configuration.items()
    ):
        offset *= offset_scale
        left_path, left_points, left_error = _solve_line(
            left, left_mount, LEFT_ARM_SEED, axis, distance, samples,
            offset, left_direction,
        )
        right_path, right_points, right_error = _solve_line(
            right, right_mount, RIGHT_ARM_SEED, axis, distance, samples,
            offset, right_direction,
        )
        trajectories[axis] = {
            "left_path": left_path,
            "right_path": right_path,
            "left_points": left_points,
            "right_points": right_points,
            "max_error": max(left_error, right_error),
        }
    arm_context = {
        "left": (left, left_mount),
        "right": (right, right_mount),
    }
    return robot, topology, trajectories, torso, arm_context


def _time_parameterize(
    path, waypoint_count, velocity, acceleration, jerk,
    blend_tolerance=0.005,
):
    indices = np.linspace(0, len(path) - 1, waypoint_count)
    indices = np.unique(np.rint(indices).astype(int))
    limits = (
        np.full(path.shape[1], velocity),
        np.full(path.shape[1], acceleration),
        np.full(path.shape[1], jerk),
    )
    return hm.JointTrajectory7(path[indices], *limits, blend_tolerance)


def _build_time_profiles(trajectories, args):
    profiles = {}
    for axis, path in trajectories.items():
        blend_tolerance = args.blend_tolerance
        if blend_tolerance is None:
            blend_tolerance = 0.016 if axis == "x" else 0.005
        profiles[axis] = {}
        for direction, reverse in (("forward", False), ("reverse", True)):
            left_path = path["left_path"]
            right_path = path["right_path"]
            if reverse:
                left_path = left_path[::-1]
                right_path = right_path[::-1]
            left = _time_parameterize(
                left_path, args.trajectory_waypoints, args.max_velocity,
                args.max_acceleration, args.max_jerk, blend_tolerance,
            )
            right = _time_parameterize(
                right_path, args.trajectory_waypoints, args.max_velocity,
                args.max_acceleration, args.max_jerk, blend_tolerance,
            )
            profiles[axis][direction] = (left, right)
        synchronized_duration = max(
            [args.duration or 0.0]
            + [
                profile.duration
                for direction in ("forward", "reverse")
                for profile in profiles[axis][direction]
            ]
        )
        for direction in ("forward", "reverse"):
            for profile in profiles[axis][direction]:
                profile.set_minimum_duration(synchronized_duration)
        profiles[axis]["duration"] = synchronized_duration
    return profiles


def _verification_times(profile, target_samples=501, minimum_per_segment=9):
    times = [np.linspace(0.0, profile.duration, target_samples)]
    for start, end in zip(profile.breakpoints[:-1], profile.breakpoints[1:]):
        times.append(np.linspace(start, end, minimum_per_segment))
    return np.unique(np.concatenate(times))


def _measure_profile_limits(profiles, samples=501):
    maxima = np.zeros(3)
    for axis in "xyz":
        for direction in ("forward", "reverse"):
            for profile in profiles[axis][direction]:
                times = _verification_times(profile, samples)
                _, velocities, accelerations, jerks = profile.sample(times)
                for index, values in enumerate(
                    (velocities, accelerations, jerks)
                ):
                    maxima[index] = max(
                        maxima[index], np.max(np.abs(values))
                    )
    return maxima


def _measure_line_deviation(profiles, trajectories, arm_context, samples=501):
    maximum = 0.0
    for axis in "xyz":
        for side in ("left", "right"):
            solver, mount = arm_context[side]
            target_points = trajectories[axis][f"{side}_points"]
            start, end = target_points[0], target_points[-1]
            segment = end - start
            squared_length = segment @ segment
            for direction in ("forward", "reverse"):
                profile_index = 0 if side == "left" else 1
                profile = profiles[axis][direction][profile_index]
                times = _verification_times(profile, samples)
                joint_samples = profile.sample(times)[0]
                for joints in joint_samples:
                    point = (mount @ solver.forward(joints))[:3, 3]
                    fraction = np.clip(
                        (point - start) @ segment / squared_length, 0.0, 1.0
                    )
                    closest = start + fraction * segment
                    maximum = max(maximum, np.linalg.norm(point - closest))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--axis", choices="xyz", default="z")
    parser.add_argument("--distance", type=float, default=0.35)
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument(
        "--duration", type=float,
        help="minimum seconds per line leg; constraints may require longer",
    )
    parser.add_argument("--trajectory-waypoints", type=int, default=7)
    parser.add_argument("--max-velocity", type=float, default=0.8)
    parser.add_argument("--max-acceleration", type=float, default=1.5)
    parser.add_argument("--max-jerk", type=float, default=4.0)
    parser.add_argument(
        "--blend-tolerance", type=float,
        help="joint-space blend radius; defaults to 0.005 (Z/Y), 0.016 (X)",
    )
    parser.add_argument(
        "--max-line-deviation", type=float, default=0.003,
        help="maximum smoothed TCP deviation in metres during validation",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    if args.duration is not None and (
        not np.isfinite(args.duration) or args.duration <= 0.0
    ):
        parser.error("--duration must be positive")
    if args.trajectory_waypoints < 3:
        parser.error("--trajectory-waypoints must be at least 3")
    limits = np.array(
        [args.max_velocity, args.max_acceleration, args.max_jerk]
    )
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        parser.error("trajectory limits must be finite and positive")
    if args.blend_tolerance is not None and (
        not np.isfinite(args.blend_tolerance)
        or args.blend_tolerance < 0.0
    ):
        parser.error("--blend-tolerance must be finite and non-negative")
    if (
        not np.isfinite(args.max_line_deviation)
        or args.max_line_deviation <= 0.0
    ):
        parser.error("--max-line-deviation must be finite and positive")
    try:
        robot, topology, trajectories, torso, arm_context = prepare_demo(
            urdf_path, args.distance, args.samples
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    time_profiles = _build_time_profiles(trajectories, args)
    line_deviation = _measure_line_deviation(
        time_profiles, trajectories, arm_context
    )
    if line_deviation > args.max_line_deviation:
        parser.error(
            f"smoothed trajectory deviates {line_deviation * 1000.0:.3f} "
            f"mm (limit {args.max_line_deviation * 1000.0:.3f} mm)"
        )
    left_names = _chain_joint_names(robot, "left_arm_base", "left_ee")
    right_names = _chain_joint_names(robot, "right_arm_base", "right_ee")
    robot.load_visuals()
    duration_summary = ", ".join(
        f"{axis}:{time_profiles[axis]['duration']:.2f}s" for axis in "zyx"
    )

    if args.validate_only:
        measured_limits = _measure_profile_limits(time_profiles)
        for link in robot.links:
            for visual in link.visuals:
                visual_mesh(hm, visual, urdf_path.parent)
        print(
            f"validated {robot.name}: {len(robot.links)} links, "
            f"left/right={len(left_names)}/{len(right_names)} DoF, "
            f"three {args.distance:.3f} m torso-frame lines, "
            f"max IK error="
            f"{max(item['max_error'] for item in trajectories.values()) * 1000.0:.3f} mm, "
            f"max smoothed-line deviation={line_deviation * 1000.0:.3f} mm, "
            f"planned legs={duration_summary}, "
            f"peak |qdot|/|qddot|/|qjerk|="
            f"{measured_limits[0]:.3f}/{measured_limits[1]:.3f}/"
            f"{measured_limits[2]:.3f}"
        )
        return

    server = viser.ViserServer(port=args.port)
    positions = dict.fromkeys(left_names + right_names, 0.0)
    current_joints = {
        "left": LEFT_ARM_SEED.copy(),
        "right": RIGHT_ARM_SEED.copy(),
    }
    handles: dict[str, list[tuple[object, np.ndarray]]] = {}
    initial_tree = _tree_transforms(robot, positions, topology)
    for link in robot.links:
        handles[link.name] = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf_path.parent)
            pose = initial_tree[link.name] @ np.asarray(visual.origin)
            wxyz, position = pose_components(pose)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/visual_{index}", mesh,
                scale=tuple(visual.scale), wxyz=wxyz, position=position
            )
            handles[link.name].append((handle, np.asarray(visual.origin)))

    def add_path(name, torso_points, color):
        world_points = np.array(
            [(torso @ np.r_[point, 1.0])[:3] for point in torso_points]
        )
        segments = np.stack((world_points[:-1], world_points[1:]), axis=1)
        add_line_segments(server.scene, name, segments, color, 3.0)

    path_colors = {
        "z": ((40, 140, 255), (255, 110, 60)),
        "y": ((70, 200, 120), (240, 190, 40)),
        "x": ((170, 90, 230), (230, 80, 160)),
    }
    for axis, trajectory in trajectories.items():
        add_path(
            f"/paths/{axis}/left", trajectory["left_points"],
            path_colors[axis][0],
        )
        add_path(
            f"/paths/{axis}/right", trajectory["right_points"],
            path_colors[axis][1],
        )

    ee_frames = {
        "left_ee": server.scene.add_frame(
            "/frames/left_ee", axes_length=0.08, axes_radius=0.004
        ),
        "right_ee": server.scene.add_frame(
            "/frames/right_ee", axes_length=0.08, axes_radius=0.004
        ),
    }
    performance = ViserPerformanceMonitor(server.gui)

    def update_scene(left_q, right_q, frame_started=None):
        if frame_started is None:
            frame_started = time.perf_counter()
        current_joints["left"] = np.asarray(left_q).copy()
        current_joints["right"] = np.asarray(right_q).copy()
        positions.update(zip(left_names, left_q))
        positions.update(zip(right_names, right_q))
        tree = _tree_transforms(robot, positions, topology)
        for link_name, transform in tree.items():
            for handle, visual_origin in handles[link_name]:
                handle.wxyz, handle.position = pose_components(
                    transform @ visual_origin
                )
        for link_name, frame in ee_frames.items():
            frame.wxyz, frame.position = pose_components(tree[link_name])
        performance.record(frame_started)

    stop_event = threading.Event()
    animation_lock = threading.Lock()
    play_vertical = server.gui.add_button("Play vertical (Z)")
    play_horizontal = server.gui.add_button("Play horizontal (Y)")
    play_depth = server.gui.add_button("Play forward/back (X)")
    stop = server.gui.add_button("Stop")
    reset = server.gui.add_button("Reset")
    status = server.gui.add_markdown(
        "### Dual-arm Trajectory\nPreparing dashboard…"
    )
    playback = {"axis": args.axis, "direction": "ready", "phase": 0.0}

    def update_dashboard() -> None:
        axis = playback["axis"]
        left_solver, left_mount = arm_context["left"]
        right_solver, right_mount = arm_context["right"]
        left_xyz = (left_mount @ left_solver.forward(
            current_joints["left"]
        ))[:3, 3]
        right_xyz = (right_mount @ right_solver.forward(
            current_joints["right"]
        ))[:3, 3]
        status.content = (
            "### Dual-arm Trajectory\n"
            "🔵 Left TCP · 🟠 Right TCP\n\n"
            "| Metric | Value |\n|:--|--:|\n"
            f"| Torso axis | **{axis.upper()}** |\n"
            f"| State | **{playback['direction']}** |\n"
            f"| Progress | **{playback['phase'] * 100.0:.1f}%** |\n"
            f"| Line length | **{args.distance * 100.0:.1f} cm** |\n"
            f"| Leg duration | **{time_profiles[axis]['duration']:.3f} s** |\n"
            f"| Max planned IK error | **{trajectories[axis]['max_error'] * 1000.0:.3f} mm** |\n"
            f"| Smoothed line deviation | **{line_deviation * 1000.0:.3f} mm** |\n\n"
            f"Left XYZ: `{left_xyz[0]:+.3f}, {left_xyz[1]:+.3f}, "
            f"{left_xyz[2]:+.3f}` m  \n"
            f"Right XYZ: `{right_xyz[0]:+.3f}, {right_xyz[1]:+.3f}, "
            f"{right_xyz[2]:+.3f}` m"
        )

    def run_pair(left_profile, right_profile, direction):
        wall_duration = max(left_profile.duration, right_profile.duration)
        start = time.monotonic()
        while not stop_event.is_set():
            frame_started = time.perf_counter()
            phase = min((time.monotonic() - start) / wall_duration, 1.0)
            playback["direction"] = direction
            playback["phase"] = phase
            update_scene(
                left_profile.position(phase * left_profile.duration),
                right_profile.position(phase * right_profile.duration),
                frame_started,
            )
            update_dashboard()
            if phase >= 1.0:
                break
            time.sleep(1.0 / 60.0)
        return wall_duration

    def animate(axis):
        if not animation_lock.acquire(blocking=False):
            return
        try:
            stop_event.clear()
            playback.update(axis=axis, direction="transition", phase=0.0)
            update_dashboard()
            selected = trajectories[axis]
            left_path = selected["left_path"]
            right_path = selected["right_path"]
            transition_needed = max(
                np.linalg.norm(current_joints["left"] - left_path[0]),
                np.linalg.norm(current_joints["right"] - right_path[0]),
            ) > 1e-8
            if transition_needed:
                transition_blend = (
                    0.005 if args.blend_tolerance is None
                    else args.blend_tolerance
                )
                transition_left = _time_parameterize(
                    np.vstack((current_joints["left"], left_path[0])), 2,
                    args.max_velocity, args.max_acceleration, args.max_jerk,
                    transition_blend,
                )
                transition_right = _time_parameterize(
                    np.vstack((current_joints["right"], right_path[0])), 2,
                    args.max_velocity, args.max_acceleration, args.max_jerk,
                    transition_blend,
                )
                transition_duration = max(
                    transition_left.duration, transition_right.duration
                )
                transition_left.set_minimum_duration(transition_duration)
                transition_right.set_minimum_duration(transition_duration)
                run_pair(transition_left, transition_right, "transition")
            profiles = time_profiles[axis]
            while not stop_event.is_set():
                for direction in ("forward", "reverse"):
                    left_profile, right_profile = profiles[direction]
                    run_pair(left_profile, right_profile, direction)
                if not args.loop:
                    break
        finally:
            playback["direction"] = "stopped"
            update_dashboard()
            animation_lock.release()

    def start_animation(axis):
        threading.Thread(target=animate, args=(axis,), daemon=True).start()

    @play_vertical.on_click
    def _on_play_vertical(_event):
        start_animation("z")

    @play_horizontal.on_click
    def _on_play_horizontal(_event):
        start_animation("y")

    @play_depth.on_click
    def _on_play_depth(_event):
        start_animation("x")

    @stop.on_click
    def _on_stop(_event):
        stop_event.set()
        playback["direction"] = "stopping"
        update_dashboard()

    @reset.on_click
    def _on_reset(_event):
        stop_event.set()
        playback.update(axis=args.axis, direction="ready", phase=0.0)
        selected = trajectories[args.axis]
        update_scene(selected["left_path"][0], selected["right_path"][0])
        update_dashboard()

    initial = trajectories[args.axis]
    update_scene(initial["left_path"][0], initial["right_path"][0])
    update_dashboard()
    if args.autoplay:
        start_animation(args.axis)
    print(
        f"Viewing {robot.name} from {urdf_path}; "
        f"planned legs={duration_summary}; press Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_event.set()
        server.stop()


if __name__ == "__main__":
    main()
