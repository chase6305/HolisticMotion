"""cuRobo-style multi-seed retargeting without a cuRobo runtime dependency.

The solver borrows cuRobo's deterministic seed-and-refine structure while
reusing HolisticMotion's bounded task-space optimizer.  Pinocchio remains the
only optional robotics dependency; Torch, Warp, and CUDA are not imported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from time import perf_counter
from typing import Optional, Union

import numpy as np

from .pink_solver import PinkRetargetingSolver
from .pinocchio_solver import RetargetingResult, RetargetingTarget


class CuroboRetargetingSolver(PinkRetargetingSolver):
    """Deterministic multi-seed, bounded nonlinear retargeting solver.

    Each seed is refined with the inherited adaptive damped optimizer.  The
    best converged result is returned, or the finite result with the smallest
    normalized task error when none of the seeds converges.
    """

    def __init__(
        self,
        *args,
        num_seeds: int = 8,
        seed_spread: float = 0.35,
        sampler_seed: int = 451,
        stop_on_success: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(num_seeds, bool) or not isinstance(num_seeds, Integral):
            raise TypeError("num_seeds must be an integer")
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")
        if not np.isfinite(seed_spread) or seed_spread < 0.0:
            raise ValueError("seed_spread must be finite and non-negative")
        if isinstance(sampler_seed, bool) or not isinstance(sampler_seed, Integral):
            raise TypeError("sampler_seed must be an integer")
        if sampler_seed < 0:
            raise ValueError("sampler_seed must be non-negative")
        if not isinstance(stop_on_success, (bool, np.bool_)):
            raise TypeError("stop_on_success must be boolean")
        self.num_seeds = int(num_seeds)
        self.seed_spread = float(seed_spread)
        self.sampler_seed = int(sampler_seed)
        self.stop_on_success = bool(stop_on_success)
        if self.num_seeds > 1 and self.seed_spread > 0.0 and self.model.nv:
            generator = np.random.default_rng(self.sampler_seed)
            self._seed_samples = generator.uniform(
                -1.0, 1.0, (10 * self.num_seeds, self.model.nv)
            )
        else:
            self._seed_samples = np.empty((0, self.model.nv))
        self._seed_samples.setflags(write=False)
        self.last_seed_index = 0
        self.last_num_seeds_evaluated = 0

    def solve(
        self,
        targets: Mapping[str, Union[RetargetingTarget, np.ndarray]],
        seed: Optional[Sequence[float]] = None,
        *,
        max_iterations: Optional[int] = None,
        enforce_acceleration: bool = False,
    ) -> RetargetingResult:
        """Refine a deterministic seed bank and return its best solution."""

        started = perf_counter()
        desired_poses, frame_weights = self._prepare_targets(targets)
        primary = self._configuration(seed) if seed is not None else self._last_q.copy()
        seeds = [primary.copy()] if enforce_acceleration else self._seed_bank(primary)
        best = None
        best_rank = None
        best_index = 0
        best_velocity = None
        evaluated = 0
        total_collision_evaluations = 0
        total_collision_gradient_evaluations = 0
        initial_q = self._last_q.copy()
        initial_velocity = self._last_velocity.copy()
        collision_cost_cache = {}
        collision_gradient_cache = {}
        try:
            for index, candidate_seed in enumerate(seeds):
                # Acceleration-limited seed trials must share the same history;
                # otherwise seed order would change the dynamic bounds.
                self._last_velocity = initial_velocity.copy()
                result = super()._solve_seed(
                    desired_poses,
                    frame_weights,
                    seed=candidate_seed,
                    max_iterations=max_iterations,
                    enforce_acceleration=enforce_acceleration,
                    collision_cost_cache=collision_cost_cache,
                    collision_gradient_cache=collision_gradient_cache,
                )
                evaluated += 1
                total_collision_evaluations += result.collision_evaluations
                total_collision_gradient_evaluations += (
                    result.collision_gradient_evaluations
                )
                rank = self._rank(result)
                if best_rank is None or rank < best_rank:
                    best = result
                    best_rank = rank
                    best_index = index
                    best_velocity = self._last_velocity.copy()
                if self.stop_on_success and result.success:
                    break
        except Exception:
            self._last_q = initial_q
            self._last_velocity = initial_velocity
            raise

        self._last_q = best.configuration.copy()
        self._last_velocity = best_velocity
        self.last_seed_index = best_index
        self.last_num_seeds_evaluated = evaluated
        return replace(
            best,
            solve_ms=(perf_counter() - started) * 1000.0,
            collision_evaluations=total_collision_evaluations,
            collision_gradient_evaluations=total_collision_gradient_evaluations,
        )

    def _seed_bank(self, primary: np.ndarray) -> list[np.ndarray]:
        seeds = [primary.copy()]
        if self.num_seeds == 1:
            return seeds

        neutral = self._neutral_q
        if not np.allclose(neutral, seeds[0], rtol=0.0, atol=1e-12):
            seeds.append(neutral)

        if self.seed_spread == 0.0:
            return seeds[: self.num_seeds]

        attempts = 0
        maximum_attempts = 10 * self.num_seeds
        while len(seeds) < self.num_seeds and attempts < maximum_attempts:
            tangent = self.seed_spread * self._seed_samples[attempts]
            attempts += 1
            candidate = np.asarray(self.pin.integrate(self.model, primary, tangent))
            candidate = self._project_limits(candidate)
            if not any(
                np.allclose(candidate, existing, rtol=0.0, atol=1e-12)
                for existing in seeds
            ):
                seeds.append(candidate)
        return seeds[: self.num_seeds]

    def _rank(self, result: RetargetingResult) -> tuple[float, ...]:
        values = (
            result.objective,
            result.residual,
            result.position_residual / self.position_tolerance,
            result.orientation_residual / self.orientation_tolerance,
        )
        finite = tuple(
            value if np.isfinite(value) else float("inf") for value in values
        )
        return (0.0 if result.success else 1.0, *finite)


curobo_retargeting_solver = CuroboRetargetingSolver
