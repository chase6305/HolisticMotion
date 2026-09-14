"""Upper-body IK stages; path generation and execution belong to the caller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .modes import RetargetingMode
from .pink_solver import PinkRetargetingSolver
from .pinocchio_solver import RetargetingResult, RetargetingTarget

_ARM_MODES = (
    RetargetingMode.LEFT_ARM,
    RetargetingMode.RIGHT_ARM,
    RetargetingMode.DUAL_ARM,
)


@dataclass(frozen=True)
class TorsoFirstResult:
    """IK waypoints, including a failed stage for diagnostics.

    ``arms`` is absent if torso alignment failed. These configurations do not
    certify the intervening paths or the robot's measured execution state.
    """

    torso: RetargetingResult
    arms: Optional[RetargetingResult] = None

    def __post_init__(self) -> None:
        if not isinstance(self.torso, RetargetingResult):
            raise TypeError("torso must be RetargetingResult")
        if self.torso.mode is not RetargetingMode.TORSO:
            raise ValueError("torso result must use torso mode")
        if self.arms is not None:
            if not isinstance(self.arms, RetargetingResult):
                raise TypeError("arms must be RetargetingResult or None")
            if not self.torso.success:
                raise ValueError("arm solving requires successful torso alignment")
            if self.arms.mode not in _ARM_MODES:
                raise ValueError("arms result must use an arm-only mode")
            if self.arms.configuration.shape != self.torso.configuration.shape:
                raise ValueError("stage configurations must have the same dimension")

    @property
    def success(self) -> bool:
        return self.arms is not None and self.torso.success and self.arms.success


def solve_torso_first(
    solver: PinkRetargetingSolver,
    torso_target: Union[RetargetingTarget, np.ndarray],
    hand_targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
    *,
    seed: Sequence[float],
    arm_mode: Union[RetargetingMode, str] = RetargetingMode.DUAL_ARM,
) -> TorsoFirstResult:
    """Align the torso with fixed arm joints, then solve arms with fixed torso.

    Uses the solver's ``torso`` and selected arm-only mode. Their active joint
    sets must be disjoint. Hands move with the torso in the first stage; this
    strategy does not maintain world-frame hand poses or a rigid dual grasp.
    A failed first stage prevents the second solve. The original mode is always
    restored; normal returns retain the last attempted stage as the warm start.
    Exceptions, including interrupts, restore configuration, velocity history,
    and multi-seed diagnostics before propagating to the caller.

    These are offline IK solves. The caller must plan and validate each path,
    execute torso alignment, and confirm arrival before executing arm motion.
    """

    if not isinstance(solver, PinkRetargetingSolver):
        raise TypeError("solver must be PinkRetargetingSolver or a subclass")
    arm_mode = RetargetingMode(arm_mode)
    if arm_mode not in _ARM_MODES:
        raise ValueError("arm_mode must be left_arm, right_arm, or dual_arm")
    if seed is None:
        raise ValueError(
            "torso-first retargeting requires an explicit seed configuration"
        )
    torso_target = (
        torso_target
        if isinstance(torso_target, RetargetingTarget)
        else RetargetingTarget(torso_target)
    )
    configuration = solver._configuration(seed)
    previous_mode = solver.mode
    previous_state = solver._snapshot_solve_state()
    try:
        # Validate both stages before either one can update solver history.
        solver.prepare(arm_mode)
        normalized_hands = solver._normalize_targets(hand_targets)
        arm_indices = solver._mode_plan().active_velocity_indices
        solver.prepare(RetargetingMode.TORSO)
        normalized_torso = solver._normalize_targets({"torso": torso_target})
        torso_indices = solver._mode_plan().active_velocity_indices
        if not torso_indices.size:
            raise ValueError("torso-first retargeting requires movable torso joints")
        if np.intersect1d(torso_indices, arm_indices).size:
            raise ValueError("torso and arm modes must have disjoint active joints")

        torso = solver.solve(normalized_torso, seed=configuration)
        if not torso.success:
            return TorsoFirstResult(torso)
        solver.set_mode(arm_mode)
        arms = solver.solve(normalized_hands, seed=torso.configuration)
        return TorsoFirstResult(torso, arms)
    except BaseException:
        solver._restore_solve_state(previous_state)
        raise
    finally:
        solver.set_mode(previous_mode)
