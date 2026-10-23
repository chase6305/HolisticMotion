"""Weighted frame and balance task assembly."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def task_state(
    solver,
    q: np.ndarray,
    frame_weights: Mapping[str, np.ndarray],
    desired_poses: Mapping[str, object],
    active: np.ndarray,
    regularization: float,
    build_system: bool,
    compute_metrics: bool,
    collision_evaluations: list[int],
    collision_cost_cache: dict[bytes, float],
    collision_gradient_cache: dict[bytes, np.ndarray],
) -> dict:
    center_enabled = (
        solver.center_of_mass_task is not None
        and np.max(solver.center_of_mass_task.cost) > 0.0
    )
    support_enabled = (
        solver.support_polygon_task is not None
        and solver.support_polygon_task.cost > 0.0
    )
    zmp_tracking_enabled = (
        solver.zmp_task is not None and np.max(solver.zmp_task.cost) > 0.0
    )
    # A zero-weight ZMP tracker can still supply the support constraint's
    # plane/gravity model without requiring a point-tracking target.
    zmp_required = zmp_tracking_enabled or (
        support_enabled and solver.support_polygon_task.reference == "zmp"
    )
    frame_jacobians_required = (
        build_system
        and active.size
        and any(np.max(weights) > 0.0 for weights in frame_weights.values())
    )
    center_jacobian_required = (
        build_system
        and active.size
        and (center_enabled or support_enabled or zmp_required)
    )
    if build_system and (frame_jacobians_required or center_jacobian_required):
        # Build the model-wide joint Jacobians once, then extract every
        # active frame below. This matters for dual-arm and full-body
        # modes, where computeFrameJacobian would repeat the traversal for
        # each target.
        solver.pin.computeJointJacobians(solver.model, solver.data, q)
    else:
        solver.pin.forwardKinematics(solver.model, solver.data, q)
    solver.pin.updateFramePlacements(solver.model, solver.data)
    hessian = None
    gradient = None
    if build_system:
        workspace = solver._system_workspace.get(solver.mode)
        if workspace is None or workspace[0].shape != (active.size, active.size):
            workspace = (
                np.empty((active.size, active.size), dtype=float),
                np.empty(active.size, dtype=float),
            )
            solver._system_workspace[solver.mode] = workspace
        hessian, gradient = workspace
        hessian.fill(0.0)
        gradient.fill(0.0)
        hessian.flat[:: active.size + 1] = (
            regularization + solver._joint_motion_weights[active] ** 2
        )
    weighted_error_squared = 0.0
    # The QP gradient scales each task correction by its gain once.
    # Apply that same factor to the merit function used by backtracking
    # and seed ranking; keep residual telemetry independent of gains.
    task_objective = 0.0
    target_residuals = {} if compute_metrics else None
    position_convergence_residual = 0.0
    orientation_convergence_residual = 0.0
    for name in solver._mode_plan().targets:
        task = solver.frame_tasks[name]
        desired = desired_poses[name]
        current = solver.data.oMf[solver._frame_ids[name]]
        relative = current.inverse() * desired
        # Keep translation independent of the target's rotation. The
        # translational part of log6 includes rotation-dependent coupling,
        # so masking its angular components does not disable orientation.
        # Both errors use current-frame axes, matching the LOCAL Jacobian.
        error = np.empty(6)
        error[:3] = relative.translation
        error[3:] = solver.pin.log3(relative.rotation)
        if compute_metrics:
            position_error = float(np.linalg.norm(error[:3]))
            orientation_error = float(np.linalg.norm(error[3:]))
            target_residuals[name] = (position_error, orientation_error)
            position_convergence_residual = max(
                position_convergence_residual,
                position_error
                if np.min(task.position_cost) > 0.0
                else float(np.linalg.norm(error[:3][task.position_cost > 0.0])),
            )
            orientation_convergence_residual = max(
                orientation_convergence_residual,
                orientation_error
                if np.min(task.orientation_cost) > 0.0
                else float(np.linalg.norm(error[3:][task.orientation_cost > 0.0])),
            )
        weights = frame_weights[name]
        weighted_error = weights * error
        error_squared = float(weighted_error @ weighted_error)
        weighted_error_squared += error_squared
        task_objective += 0.5 * task.gain * error_squared
        if build_system:
            if active.size and np.max(weights) > 0.0:
                jacobian = np.asarray(
                    solver.pin.getFrameJacobian(
                        solver.model,
                        solver.data,
                        solver._frame_ids[name],
                        solver.pin.ReferenceFrame.LOCAL,
                    ),
                    dtype=float,
                ).reshape(6, solver.model.nv)[:, active]
                # Unequal axis costs require the residual derivative,
                # including rotation of the current frame's error axes.
                # For isotropic blocks these terms cancel in J.T @ error;
                # retain their existing velocity-task Hessian and fast path.
                if np.any(task.position_cost != task.position_cost[0]):
                    jacobian[:3] -= solver.pin.skew(error[:3]) @ jacobian[3:]
                if np.any(task.orientation_cost != task.orientation_cost[0]):
                    jacobian[3:] = solver.pin.Jlog3(relative.rotation.T) @ jacobian[3:]
                weighted_jacobian = weights[:, None] * jacobian
                hessian += weighted_jacobian.T @ weighted_jacobian
                gradient += weighted_jacobian.T @ (task.gain * weighted_error)
            if task.lm_damping:
                hessian.flat[:: active.size + 1] += (
                    task.lm_damping * task.gain**2 * error_squared
                )

    posture_objective = 0.0
    weight = solver._posture_weights()[active]
    if np.any(weight > 0.0):
        posture_error = np.asarray(
            solver.pin.difference(solver.model, q, solver._posture_q), dtype=float
        )[active]
        weighted_posture_error = weight * posture_error
        posture_objective = (
            0.5
            * solver.posture_task.gain
            * float(weighted_posture_error @ weighted_posture_error)
        )
        if build_system:
            hessian.flat[:: active.size + 1] += weight * weight
            gradient += weight * weight * solver.posture_task.gain * posture_error

    current_center = None
    center_jacobian = None
    if center_enabled or support_enabled or zmp_required:
        if build_system and center_jacobian_required:
            center_jacobian = np.asarray(
                solver.pin.jacobianCenterOfMass(solver.model, solver.data, False),
                dtype=float,
            ).reshape(3, solver.model.nv)[:, active]
            # Joint placements and Jacobians are already current. The
            # no-q overload avoids repeating kinematics, while updating
            # data.com for the residual below.
            current_center = np.asarray(solver.data.com[0], dtype=float).reshape(3)
        else:
            current_center = np.asarray(
                solver.pin.centerOfMass(solver.model, solver.data, False), dtype=float
            ).reshape(3)
            if build_system:
                center_jacobian = np.zeros((3, 0))

    center_of_mass_residual = float("nan")
    center_of_mass_convergence_residual = float("nan")
    if center_enabled:
        center_error = solver._center_of_mass_target - current_center
        if compute_metrics:
            center_of_mass_residual = float(np.linalg.norm(center_error))
            center_of_mass_convergence_residual = (
                center_of_mass_residual
                if np.min(solver.center_of_mass_task.cost) > 0.0
                else float(
                    np.linalg.norm(center_error[solver.center_of_mass_task.cost > 0.0])
                )
            )
        center_weights = solver.center_of_mass_task.cost
        weighted_center_error = center_weights * center_error
        error_squared = float(weighted_center_error @ weighted_center_error)
        weighted_error_squared += error_squared
        task_objective += 0.5 * solver.center_of_mass_task.gain * error_squared
        if build_system:
            if np.max(center_weights) > 0.0:
                weighted_center_jacobian = center_weights[:, None] * center_jacobian
                hessian += weighted_center_jacobian.T @ weighted_center_jacobian
                gradient += weighted_center_jacobian.T @ (
                    solver.center_of_mass_task.gain * weighted_center_error
                )
            if solver.center_of_mass_task.lm_damping:
                hessian.flat[:: active.size + 1] += (
                    solver.center_of_mass_task.lm_damping
                    * solver.center_of_mass_task.gain**2
                    * error_squared
                )

    zmp_residual = float("nan")
    zmp_convergence_residual = float("nan")
    current_zmp = None
    zmp_jacobian = None
    if zmp_required:
        task = solver.zmp_task
        acceleration_scale = solver._center_of_mass_acceleration[:2] / task.gravity
        current_zmp = (
            current_center[:2]
            - (current_center[2] - task.plane_height) * acceleration_scale
        )
        if build_system:
            zmp_jacobian = center_jacobian[:2] - np.outer(
                acceleration_scale, center_jacobian[2]
            )
        if zmp_tracking_enabled:
            zmp_error = solver._zmp_target - current_zmp
            if compute_metrics:
                zmp_residual = float(np.linalg.norm(zmp_error))
                zmp_convergence_residual = (
                    zmp_residual
                    if np.min(task.cost) > 0.0
                    else float(np.linalg.norm(zmp_error[task.cost > 0.0]))
                )
            weighted_zmp_error = task.cost * zmp_error
            error_squared = float(weighted_zmp_error @ weighted_zmp_error)
            weighted_error_squared += error_squared
            task_objective += 0.5 * task.gain * error_squared
            if build_system:
                if np.max(task.cost) > 0.0:
                    weighted_zmp_jacobian = task.cost[:, None] * zmp_jacobian
                    hessian += weighted_zmp_jacobian.T @ weighted_zmp_jacobian
                    gradient += weighted_zmp_jacobian.T @ (
                        task.gain * weighted_zmp_error
                    )
                if task.lm_damping:
                    hessian.flat[:: active.size + 1] += (
                        task.lm_damping * task.gain**2 * error_squared
                    )

    support_polygon_violation = 0.0
    if support_enabled:
        task = solver.support_polygon_task
        support_point = current_zmp if task.reference == "zmp" else current_center[:2]
        support_point_jacobian = None
        if build_system:
            support_point_jacobian = (
                zmp_jacobian if task.reference == "zmp" else center_jacobian[:2]
            )
        # Subtract before projecting to avoid cancellation between large
        # world-coordinate dot products near a support boundary.
        signed_distances = np.einsum(
            "ei,ei->e", task.normals, support_point - task.vertices
        )
        violations = np.maximum(0.0, task.margin - signed_distances)
        if compute_metrics:
            support_polygon_violation = float(np.max(violations))
        weighted_violations = task.cost * violations
        error_squared = float(weighted_violations @ weighted_violations)
        weighted_error_squared += error_squared
        task_objective += 0.5 * task.gain * error_squared
        if build_system and task.cost > 0.0 and np.any(violations > 0.0):
            active_edges = violations > 0.0
            support_jacobian = (
                task.cost * task.normals[active_edges] @ support_point_jacobian
            )
            hessian += support_jacobian.T @ support_jacobian
            gradient += support_jacobian.T @ (
                task.gain * weighted_violations[active_edges]
            )

    if (
        build_system
        and solver.collision_cost_weight > 0.0
        and solver.collision_cost_gradient is not None
    ):
        solver._collision_cost_gradient_value(
            q,
            active,
            collision_evaluations,
            collision_cost_cache,
            collision_gradient_cache,
        )
    collision_cost = solver._collision_cost_value(
        q, collision_evaluations, collision_cost_cache
    )
    collision_objective = solver.collision_cost_weight * collision_cost
    if not np.isfinite(collision_objective):
        raise ValueError("weighted collision cost must be finite")
    if build_system and collision_objective > 0.0:
        with np.errstate(over="ignore", invalid="ignore"):
            collision_update = (
                solver.collision_cost_weight
                * solver._collision_gradient_value(
                    q,
                    collision_cost,
                    active,
                    collision_evaluations,
                    collision_cost_cache,
                    collision_gradient_cache,
                )
            )
        if not np.isfinite(collision_update).all():
            raise ValueError("weighted collision gradient must be finite")
        gradient -= collision_update

    objective = task_objective + posture_objective + collision_objective
    if not np.isfinite(objective):
        raise ValueError("retargeting objective must be finite")
    if not compute_metrics:
        return {"objective": objective}
    position_residual = max(value[0] for value in target_residuals.values())
    orientation_residual = max(value[1] for value in target_residuals.values())
    if build_system and (
        not np.isfinite(hessian).all() or not np.isfinite(gradient).all()
    ):
        raise ValueError("retargeting linear system must be finite")
    return {
        "hessian": hessian,
        "gradient": gradient,
        "objective": objective,
        "residual": float(np.sqrt(weighted_error_squared)),
        "position_residual": position_residual,
        "orientation_residual": orientation_residual,
        "position_convergence_residual": position_convergence_residual,
        "orientation_convergence_residual": orientation_convergence_residual,
        "target_residuals": target_residuals,
        "collision_cost": collision_cost,
        "center_of_mass_residual": center_of_mass_residual,
        "center_of_mass_convergence_residual": center_of_mass_convergence_residual,
        "support_polygon_violation": support_polygon_violation,
        "zmp_residual": zmp_residual,
        "zmp_convergence_residual": zmp_convergence_residual,
    }
