#!/usr/bin/env python3
"""Plan and animate a collision-checked dual-arm path on a URDF robot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _import_holistic_motion():
    try:
        import holistic_motion

        return holistic_motion
    except ModuleNotFoundError as error:
        repository = Path(__file__).resolve().parents[3]
        runner = repository / "scripts/run.sh"
        if error.name != "holistic_motion" or os.environ.get("HMOTION_DEMO_REEXEC"):
            raise
        os.environ["HMOTION_DEMO_REEXEC"] = "1"
        os.execv(str(runner), [str(runner), sys.executable, __file__, *sys.argv[1:]])
        raise RuntimeError from error


hm = _import_holistic_motion()

from holistic_motion.visualization.viser import (
    add_line_segments,
    pose_components,
    tree_topology,
    tree_transforms,
    visual_mesh,
)


def parse_args():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=repository / "examples/configs/marvin.json",
    )
    parser.add_argument(
        "--algorithm",
        choices=("rrt-connect", "rrt-star", "informed-rrt-star"),
        default="rrt-connect",
    )
    parser.add_argument(
        "--planning-space",
        choices=("symmetric", "coupled"),
        default="symmetric",
        help="use a mirrored 7-DoF subspace or the full 14-DoF dual-arm space",
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--range", dest="extension_range", type=float, default=0.13)
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument(
        "--shortcut-attempts",
        type=int,
        default=100,
        help="continuous collision-aware shortcut attempts",
    )
    parser.add_argument(
        "--security-margin",
        type=float,
        default=0.005,
        help="minimum collision-mesh clearance in metres",
    )
    parser.add_argument(
        "--planning-padding",
        type=float,
        default=0.001,
        help="extra planning clearance guarding discretized edge checks",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        help="override the profile demo goal with a sinusoidal joint offset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="deterministic seed selected for compact, balanced dual-arm paths",
    )
    parser.add_argument(
        "--no-obstacles", action="store_true", help="disable profile environment boxes"
    )
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def descendants(robot, root):
    result = []
    pending = [root]
    while pending:
        link_name = pending.pop()
        result.append(link_name)
        for joint_name in robot.get_link(link_name).child_joints:
            pending.append(robot.get_joint(joint_name).child_link)
    return result


def update_robot(handles, robot, topology, joint_names, active):
    positions = dict(zip(joint_names, active))
    transforms = tree_transforms(robot, positions, topology)
    for link in robot.links:
        for handle, origin in handles[link.name]:
            wxyz, position = pose_components(transforms[link.name] @ origin)
            handle.wxyz = wxyz
            handle.position = position


def end_effector_paths(robot, topology, joint_names, configurations, end_links):
    paths = {side: [] for side in end_links}
    for active in configurations:
        transforms = tree_transforms(robot, dict(zip(joint_names, active)), topology)
        for side, link_name in end_links.items():
            paths[side].append(transforms[link_name][:3, 3].copy())
    return {side: np.asarray(points) for side, points in paths.items()}


def main() -> None:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    urdf = args.urdf.expanduser().resolve()
    robot = hm.Robot(str(urdf), True)
    collision = hm.CollisionModel(str(urdf), [str(urdf.parent)])
    obstacles = (
        []
        if args.no_obstacles
        else profile.get("planning", {}).get("environment_obstacles", [])
    )
    obstacle_names = set()
    for obstacle in obstacles:
        if obstacle.get("type") != "box":
            raise RuntimeError(f"unsupported obstacle type: {obstacle.get('type')}")
        name = obstacle["name"]
        if name in obstacle_names:
            raise RuntimeError(f"duplicate obstacle name: {name}")
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(obstacle["position"], dtype=float)
        collision.add_box_obstacle(name, obstacle["size"], pose)
        obstacle_names.add(name)
    groups = profile["retargeting"]["joint_groups"]
    joint_names = groups["left_arm"] + groups["right_arm"]
    start = np.asarray(
        profile["arms"]["left"]["seed"] + profile["arms"]["right"]["seed"],
        dtype=float,
    )
    context = collision.configuration_from_joint_positions(
        dict(zip(joint_names, start))
    )

    available = set(collision.collision_link_names)
    arm_links = {}
    for side in ("left", "right"):
        end = profile["arms"][side]["ee"]
        chain_links = [
            robot.get_joint(name).child_link for name in groups[f"{side}_arm"]
        ]
        arm_links[side] = list(
            dict.fromkeys(
                link
                for link in chain_links + descendants(robot, end)
                if link in available
            )
        )
    body_links = sorted(
        available - set(arm_links["left"]) - set(arm_links["right"]) - obstacle_names
    )
    collision_groups = {
        "left_arm": arm_links["left"],
        "right_arm": arm_links["right"],
        "body": body_links,
    }
    collision_group_pairs = [
        ("left_arm", "right_arm"),
        ("left_arm", "left_arm"),
        ("right_arm", "right_arm"),
        ("left_arm", "body"),
        ("right_arm", "body"),
    ]
    if obstacle_names:
        collision_groups["environment"] = sorted(obstacle_names)
        collision_group_pairs.extend(
            [("left_arm", "environment"), ("right_arm", "environment")]
        )
    collision.set_collision_groups(collision_groups, collision_group_pairs)
    collision_profile = profile.get("collision", {})
    disabled_pairs = collision_profile.get("disabled_self_link_pairs", []) + (
        collision_profile.get("disabled_body_link_pairs", [])
    )
    for first, second in disabled_pairs:
        collision.remove_collision_pairs_by_links(first, second)
    if collision.is_within_distance(context, args.security_margin):
        raise RuntimeError("profile start configuration violates collision margin")

    if args.planning_padding < 0.0:
        raise RuntimeError("planning padding must be non-negative")
    planning_margin = args.security_margin + args.planning_padding
    lower = collision.joint_lower_limits(joint_names)
    upper = collision.joint_upper_limits(joint_names)
    left_count = len(groups["left_arm"])
    mirror_signs = np.asarray(
        profile.get("planning", {}).get("right_arm_mirror_signs", []),
        dtype=float,
    )
    if args.amplitude is None and profile.get("planning", {}).get("dual_arm_demo_goal"):
        goal = np.asarray(profile["planning"]["dual_arm_demo_goal"], dtype=float)
    else:
        amplitude = 0.28 if args.amplitude is None else args.amplitude
        if args.planning_space == "symmetric" and mirror_signs.shape == (
            left_count,
        ):
            phase = np.arange(left_count) * 0.73
            left_goal = start[:left_count] + amplitude * np.sin(phase + 0.4)
            goal = np.concatenate((left_goal, mirror_signs * left_goal))
        else:
            phase = np.arange(len(joint_names)) * 0.73
            goal = start + amplitude * np.sin(phase + 0.4)
        goal = np.clip(goal, lower, upper)
    if goal.shape != start.shape or np.any(goal < lower) or np.any(goal > upper):
        raise RuntimeError("planning goal does not match active-joint limits")
    goal_full = collision.configuration_with_joint_positions(context, joint_names, goal)
    if collision.is_within_distance(goal_full, args.security_margin):
        raise RuntimeError("generated goal violates the collision safety margin")

    planning_start = start
    planning_goal = goal

    def expand_planning_state(state):
        return np.asarray(state, dtype=float)

    if args.planning_space == "symmetric":
        mirror = mirror_signs
        if mirror.shape != (left_count,) or not np.all(np.isin(mirror, (-1.0, 1.0))):
            raise RuntimeError(
                "symmetric planning requires one +/-1 mirror sign per arm joint"
            )

        def expand_planning_state(state):
            left = np.asarray(state, dtype=float)
            return np.concatenate((left, mirror * left))

        mirrored_start = expand_planning_state(start[:left_count])
        mirrored_goal = expand_planning_state(goal[:left_count])
        if not np.allclose(start, mirrored_start) or not np.allclose(
            goal, mirrored_goal
        ):
            raise RuntimeError(
                "start/goal are not mirrored; use --planning-space coupled"
            )
        left_lower = lower[:left_count].copy()
        left_upper = upper[:left_count].copy()
        right_lower = lower[left_count:]
        right_upper = upper[left_count:]
        for index, sign in enumerate(mirror):
            if sign > 0.0:
                left_lower[index] = max(left_lower[index], right_lower[index])
                left_upper[index] = min(left_upper[index], right_upper[index])
            else:
                left_lower[index] = max(left_lower[index], -right_upper[index])
                left_upper[index] = min(left_upper[index], -right_lower[index])
        if np.any(left_lower > left_upper):
            raise RuntimeError("mirrored arm joint limits have no common interval")

        def symmetric_state_valid(state):
            active = expand_planning_state(state)
            full = collision.configuration_with_joint_positions(
                context, joint_names, active
            )
            return not collision.is_within_distance(full, planning_margin)

        planner = hm.SamplingPlanner(
            left_lower, left_upper, symmetric_state_valid
        )
        planning_start = start[:left_count]
        planning_goal = goal[:left_count]
    else:
        planner = hm.SamplingPlanner.from_collision_joints(
            collision, joint_names, context, planning_margin
        )
    endpoint_sides = {}
    if obstacles:
        topology_for_layout = tree_topology(robot)
        start_tree = tree_transforms(
            robot, dict(zip(joint_names, start)), topology_for_layout
        )
        goal_tree = tree_transforms(
            robot, dict(zip(joint_names, goal)), topology_for_layout
        )
        for side in ("left", "right"):
            obstacle = next(item for item in obstacles if item["name"].startswith(side))
            center_x = float(obstacle["position"][0])
            half_x = float(obstacle["size"][0]) * 0.5
            start_x = float(start_tree[profile["arms"][side]["ee"]][0, 3])
            goal_x = float(goal_tree[profile["arms"][side]["ee"]][0, 3])
            start_offset = start_x - center_x
            goal_offset = goal_x - center_x
            if (
                start_offset * goal_offset >= 0.0
                or abs(start_offset) <= half_x
                or abs(goal_offset) <= half_x
            ):
                raise RuntimeError(
                    f"{side} path endpoints are not on opposite obstacle sides"
                )
            if min(start_x, goal_x) <= 0.20:
                raise RuntimeError(f"{side} path leaves the chest-front workspace")
            endpoint_sides[side] = (start_x, goal_x)
    endpoint_layout = (
        ", ".join(
            f"{side} x:{values[0]:.3f}->{values[1]:.3f}"
            for side, values in endpoint_sides.items()
        )
        or "unconstrained"
    )
    direct_collision = any(
        collision.is_within_distance(
            collision.configuration_with_joint_positions(
                context, joint_names, (1.0 - alpha) * start + alpha * goal
            ),
            args.security_margin,
        )
        for alpha in np.linspace(0.05, 0.95, 19)
    )

    algorithms = {
        "rrt-connect": hm.SamplingAlgorithm.RRT_CONNECT,
        "rrt-star": hm.SamplingAlgorithm.RRT_STAR,
        "informed-rrt-star": hm.SamplingAlgorithm.INFORMED_RRT_STAR,
    }
    options = hm.PlanningOptions()
    options.algorithm = algorithms[args.algorithm]
    options.timeout_seconds = args.timeout
    options.extension_range = args.extension_range
    options.edge_resolution = args.resolution
    options.random_seed = args.seed
    options.shortcut_attempts = args.shortcut_attempts
    options.interpolate_path = True
    options.interpolation_points = 180
    result = planner.plan(planning_start, planning_goal, options)
    if not result.success:
        raise RuntimeError(f"planning failed: {result.message}")
    stats = result.statistics
    path = np.asarray([expand_planning_state(state) for state in result.path])
    topology = tree_topology(robot)
    end_links = {
        "left": profile["arms"]["left"]["ee"],
        "right": profile["arms"]["right"]["ee"],
    }
    planned_ee_paths = end_effector_paths(
        robot, topology, joint_names, path, end_links
    )
    direct_configurations = np.linspace(start, goal, len(path))
    direct_ee_paths = end_effector_paths(
        robot, topology, joint_names, direct_configurations, end_links
    )
    ee_lengths = {
        side: float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        for side, points in planned_ee_paths.items()
    }
    full_path = [
        collision.configuration_with_joint_positions(context, joint_names, active)
        for active in path
    ]
    invalid_samples = [
        index
        for index, full in enumerate(full_path)
        if collision.is_within_distance(full, args.security_margin)
    ]
    if invalid_samples:
        raise RuntimeError(
            f"final path failed collision verification at samples {invalid_samples[:8]}"
        )
    print(
        f"planned {len(joint_names)}-DoF dual-arm path in "
        f"{args.planning_space} space with {args.algorithm} "
        f"(straight path collision={direct_collision}): "
        f"{stats.planning_time_ms:.2f} ms, {stats.tree_nodes} nodes, "
        f"{stats.collision_checks} collision checks, {len(path)} samples, "
        f"length {stats.initial_path_length:.3f} -> {stats.final_path_length:.3f}, "
        f"margin={args.security_margin * 1000.0:.1f} mm, "
        f"planning margin={planning_margin * 1000.0:.1f} mm, "
        f"EE lengths=left {ee_lengths['left']:.3f} m/right "
        f"{ee_lengths['right']:.3f} m, "
        f"verified samples={len(full_path)}, "
        f"pairs={collision.pair_count}, obstacles={len(obstacles)}, "
        f"waypoints=chest-front ({endpoint_layout})"
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
    for obstacle in obstacles:
        server.scene.add_box(
            f"/environment/{obstacle['name']}",
            color=tuple(obstacle.get("color", (70, 110, 165))),
            dimensions=tuple(obstacle["size"]),
            position=np.asarray(obstacle["position"], dtype=float),
            opacity=0.8,
        )
    initial = tree_transforms(robot, dict(zip(joint_names, start)), topology)
    handles = {}
    for link in robot.links:
        handles[link.name] = []
        for index, visual in enumerate(link.visuals):
            mesh = visual_mesh(hm, visual, urdf.parent)
            origin = np.asarray(visual.origin)
            wxyz, position = pose_components(initial[link.name] @ origin)
            handle = server.scene.add_mesh_trimesh(
                f"/robot/{link.name}/visual_{index}",
                mesh,
                scale=tuple(visual.scale),
                wxyz=wxyz,
                position=position,
            )
            handles[link.name].append((handle, origin))

    path_colors = {"left": (255, 155, 40), "right": (210, 90, 255)}
    planned_lines = {}
    direct_lines = {}
    current_markers = {}
    for side in ("left", "right"):
        planned_points = planned_ee_paths[side]
        direct_points = direct_ee_paths[side]
        planned_lines[side] = add_line_segments(
            server.scene,
            f"/rrt/optimized/{side}",
            np.stack((planned_points[:-1], planned_points[1:]), axis=1),
            path_colors[side],
            4.0,
        )
        direct_lines[side] = add_line_segments(
            server.scene,
            f"/rrt/direct_collision/{side}",
            np.stack((direct_points[:-1], direct_points[1:]), axis=1),
            (220, 55, 55),
            1.5,
        )
        server.scene.add_icosphere(
            f"/rrt/endpoints/{side}_start",
            radius=0.018,
            color=(60, 200, 100),
            material="toon5",
            position=planned_points[0],
        )
        server.scene.add_icosphere(
            f"/rrt/endpoints/{side}_goal",
            radius=0.018,
            color=path_colors[side],
            material="toon5",
            position=planned_points[-1],
        )
        current_markers[side] = server.scene.add_icosphere(
            f"/rrt/current/{side}",
            radius=0.014,
            color=(40, 190, 255),
            material="toon5",
            position=planned_points[0],
        )

    playing = server.gui.add_checkbox("Play path", initial_value=True)
    show_planned = server.gui.add_checkbox(
        "Show optimized RRT paths", initial_value=True
    )
    show_direct = server.gui.add_checkbox(
        "Show colliding direct paths", initial_value=True
    )
    speed = server.gui.add_slider(
        "Playback speed", min=0.1, max=2.0, step=0.1, initial_value=0.7
    )
    timeline = server.gui.add_slider(
        "Path progress", min=0.0, max=1.0, step=1.0 / (len(path) - 1), initial_value=0.0
    )
    server.gui.add_markdown(
        f"**{args.algorithm} + collision-aware shortcut**  \n"
        f"Planning space: `{args.planning_space}`  \n"
        f"Straight path collision: `{direct_collision}`  \n"
        f"Collision policy: `inter-arm + self + body + environment`, margin: "
        f"`{args.security_margin * 1000.0:.1f} mm` "
        f"(planning: `{planning_margin * 1000.0:.1f} mm`)  \n"
        f"Environment obstacles: `{len(obstacles)}`  \n"
        f"Endpoint layout: `near/far sides of symmetric chest-front obstacles`  \n"
        f"Verified path samples: `{len(full_path)}`  \n"
        f"Planning: `{stats.planning_time_ms:.2f} ms`  \n"
        f"Nodes: `{stats.tree_nodes}` · collision checks: `{stats.collision_checks}`  \n"
        f"Path length: `{stats.initial_path_length:.3f} → {stats.final_path_length:.3f}`"
        f"  \nEE path: left `{ee_lengths['left']:.3f} m` · right "
        f"`{ee_lengths['right']:.3f} m`"
    )
    last = time.perf_counter()
    progress = 0.0
    while True:
        now = time.perf_counter()
        if playing.value:
            progress = (progress + (now - last) * speed.value * 0.2) % 1.0
            timeline.value = progress
        else:
            progress = timeline.value
        last = now
        index = min(int(progress * (len(path) - 1)), len(path) - 1)
        update_robot(handles, robot, topology, joint_names, path[index])
        for side in ("left", "right"):
            planned_lines[side].visible = show_planned.value
            direct_lines[side].visible = show_direct.value
            current_markers[side].position = planned_ee_paths[side][index]
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
