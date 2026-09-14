#!/usr/bin/env python3
"""Demonstrate torso-first upper-body IK on an explicitly supplied robot URDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    solve_torso_first,
)


def _frame_pose(solver, configuration, frame_name):
    data = solver.model.createData()
    solver.pin.forwardKinematics(solver.model, data, configuration)
    solver.pin.updateFramePlacements(solver.model, data)
    frame_id = solver.model.getFrameId(frame_name)
    return np.asarray(data.oMf[frame_id].homogeneous).copy()


def _joint_summary(solver, configuration, joint_names):
    values = []
    for name in dict.fromkeys(joint_names):
        joint = solver.model.joints[solver.model.getJointId(name)]
        values.append(f"{name}={configuration[joint.idx_q]:.6g}")
    return ", ".join(values)


def _parse_joint_delta(value):
    name, separator, raw_delta = value.partition("=")
    if not separator or not name or not raw_delta:
        raise argparse.ArgumentTypeError("must use JOINT=DELTA")
    try:
        delta = float(raw_delta)
    except ValueError as error:
        raise argparse.ArgumentTypeError("DELTA must be numeric") from error
    if not np.isfinite(delta):
        raise argparse.ArgumentTypeError("DELTA must be finite")
    return name, delta


def _delta_overrides(parser, values, configured_joints):
    overrides = {}
    for name, delta in values:
        if name in overrides:
            parser.error(f"duplicate --joint-delta for {name!r}")
        if name not in configured_joints:
            parser.error(f"--joint-delta references unconfigured joint {name!r}")
        overrides[name] = delta
    return overrides


def _apply_joint_deltas(solver, configuration, joint_names, default_delta, overrides):
    tangent = np.zeros(solver.model.nv)
    scalar_joints = []
    for name in joint_names:
        joint_id = solver.model.getJointId(name)
        if joint_id == 0 or solver.model.joints[joint_id].nv == 0:
            raise ValueError(f"joint {name!r} must name a movable URDF joint")
        joint = solver.model.joints[joint_id]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(f"classic demo joint {name!r} must be scalar")
        tangent[joint.idx_v] = overrides.get(name, default_delta)
        scalar_joints.append((name, joint.idx_q))
    candidate = np.asarray(solver.pin.integrate(solver.model, configuration, tangent))
    for name, index in scalar_joints:
        lower = solver.model.lowerPositionLimit[index]
        upper = solver.model.upperPositionLimit[index]
        if candidate[index] < lower or candidate[index] > upper:
            raise ValueError(
                f"demo target for joint {name!r} is outside its URDF limits "
                f"[{lower:g}, {upper:g}]"
            )
    return candidate


def _require_arm(parser, mode, frame, joints, label):
    enabled = mode in (label, "dual_arm")
    if enabled and (frame is None or not joints):
        side = label.removesuffix("_arm")
        parser.error(f"{mode} requires --{side}-frame and at least one --{side}-joint")
    return enabled


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--urdf", type=Path, required=True, help="caller-provided robot URDF"
    )
    parser.add_argument(
        "--torso-frame", required=True, help="URDF frame aligned in stage one"
    )
    parser.add_argument(
        "--torso-joint",
        action="append",
        required=True,
        help="scalar torso joint; repeat to define the group",
    )
    parser.add_argument("--left-frame", help="left end-effector URDF frame")
    parser.add_argument(
        "--left-joint",
        action="append",
        default=[],
        help="scalar left-arm joint; repeat to define the group",
    )
    parser.add_argument("--right-frame", help="right end-effector URDF frame")
    parser.add_argument(
        "--right-joint",
        action="append",
        default=[],
        help="scalar right-arm joint; repeat to define the group",
    )
    parser.add_argument(
        "--arm-mode",
        choices=("left_arm", "right_arm", "dual_arm"),
        default="dual_arm",
        help="arm group solved after torso alignment",
    )
    parser.add_argument(
        "--torso-delta",
        type=float,
        default=0.25,
        help="neutral-relative target offset for every torso joint",
    )
    parser.add_argument(
        "--arm-delta",
        type=float,
        default=0.30,
        help="torso-aligned target offset for every selected arm joint",
    )
    parser.add_argument(
        "--joint-delta",
        action="append",
        default=[],
        type=_parse_joint_delta,
        metavar="JOINT=DELTA",
        help="override a configured joint's target offset; repeat as needed",
    )
    args = parser.parse_args()

    urdf = args.urdf.expanduser().resolve()
    if not urdf.is_file():
        parser.error(f"URDF does not exist: {urdf}")
    left = _require_arm(
        parser, args.arm_mode, args.left_frame, args.left_joint, "left_arm"
    )
    right = _require_arm(
        parser, args.arm_mode, args.right_frame, args.right_joint, "right_arm"
    )
    if not np.isfinite([args.torso_delta, args.arm_delta]).all():
        parser.error("joint deltas must be finite")
    configured_joints = {
        *args.torso_joint,
        *(args.left_joint if left else []),
        *(args.right_joint if right else []),
    }
    delta_overrides = _delta_overrides(parser, args.joint_delta, configured_joints)

    frames = {"torso": args.torso_frame}
    groups = {"torso": args.torso_joint}
    frame_tasks = {"torso": FrameTask(position_cost=1.0, orientation_cost=1.0)}
    if left:
        frames["left_hand"] = args.left_frame
        groups["left_arm"] = args.left_joint
        frame_tasks["left_hand"] = FrameTask(position_cost=1.0, orientation_cost=0.2)
    if right:
        frames["right_hand"] = args.right_frame
        groups["right_arm"] = args.right_joint
        frame_tasks["right_hand"] = FrameTask(position_cost=1.0, orientation_cost=0.2)

    try:
        solver = PinkRetargetingSolver(
            urdf,
            frames=frames,
            joint_groups=groups,
            frame_tasks=frame_tasks,
            posture_task=PostureTask(cost=1e-4),
            tolerance=1e-5,
            max_iterations=200,
        )
    except ValueError as error:
        parser.error(str(error))
    seed = np.asarray(solver.pin.neutral(solver.model))
    try:
        torso_configuration = _apply_joint_deltas(
            solver,
            seed,
            args.torso_joint,
            args.torso_delta,
            delta_overrides,
        )
        final_configuration = torso_configuration
        if left:
            final_configuration = _apply_joint_deltas(
                solver,
                final_configuration,
                args.left_joint,
                args.arm_delta,
                delta_overrides,
            )
        if right:
            final_configuration = _apply_joint_deltas(
                solver,
                final_configuration,
                args.right_joint,
                args.arm_delta,
                delta_overrides,
            )
    except ValueError as error:
        parser.error(str(error))
    hand_targets = {}
    if left:
        hand_targets["left_hand"] = _frame_pose(
            solver, final_configuration, args.left_frame
        )
    if right:
        hand_targets["right_hand"] = _frame_pose(
            solver, final_configuration, args.right_frame
        )

    try:
        stages = solve_torso_first(
            solver,
            torso_target=_frame_pose(solver, torso_configuration, args.torso_frame),
            hand_targets=hand_targets,
            seed=seed,
            arm_mode=args.arm_mode,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"torso: success={stages.torso.success} "
        f"reason={stages.torso.termination_reason} "
        f"iterations={stages.torso.iterations} residual={stages.torso.residual:.6g} "
        f"solve_ms={stages.torso.solve_ms:.3f}"
    )
    if stages.arms is None:
        print("arms: skipped because torso alignment failed")
    else:
        print(
            f"arms: success={stages.arms.success} "
            f"reason={stages.arms.termination_reason} "
            f"iterations={stages.arms.iterations} residual={stages.arms.residual:.6g} "
            f"solve_ms={stages.arms.solve_ms:.3f}"
        )
    if stages.success:
        active_arm_joints = [
            *(args.left_joint if left else []),
            *(args.right_joint if right else []),
        ]
        print(
            "aligned joints:",
            _joint_summary(solver, stages.torso.configuration, args.torso_joint),
        )
        print(
            "manipulation joints:",
            _joint_summary(
                solver,
                stages.arms.configuration,
                [*args.torso_joint, *active_arm_joints],
            ),
        )
        print("aligned configuration:", stages.torso.configuration)
        print("manipulation configuration:", stages.arms.configuration)
    return 0 if stages.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
