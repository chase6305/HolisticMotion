"""Feasible active-set solver for bounded differential-IK subproblems."""

from typing import Optional

import numpy as np


def solve_box_qp(
    hessian: np.ndarray,
    linear_term: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    unconstrained: Optional[np.ndarray] = None,
    linear_solve,
) -> np.ndarray:
    """Minimize ``0.5 * x.T @ hessian @ x - linear_term.T @ x`` within bounds.

    Use a feasible active-set method and the caller's existing linear solver.
    """

    if unconstrained is None:
        unconstrained = linear_solve(hessian, linear_term)
    if np.all(unconstrained >= lower) and np.all(unconstrained <= upper):
        return unconstrained
    solution = np.clip(unconstrained, lower, upper)
    # status: -1 at lower bound, +1 at upper bound, 0 free, 2 fixed.
    # Move toward each reduced minimizer only as far as the first blocking
    # bound. Clipping all violating coordinates independently can increase
    # the objective and cycle between active sets for coupled joints.
    status = np.zeros(solution.size, dtype=np.int8)
    status[unconstrained < lower] = -1
    status[unconstrained > upper] = 1
    status[lower == upper] = 2
    # A common positive objective scale must not change the active set.
    # An absolute floor here can accept arbitrary clipped minimizers when
    # task weights and regularization are uniformly small.
    scale = float(np.max(np.abs(linear_term)))
    kkt_tolerance = 1e-10 * scale
    for _ in range(10 * solution.size + 10):
        free = status == 0
        active = ~free
        if np.any(free):
            rhs = linear_term[free]
            if np.any(active):
                rhs = rhs - hessian[np.ix_(free, active)] @ solution[active]
            free_solution = linear_solve(hessian[np.ix_(free, free)], rhs)
            free_indices = np.flatnonzero(free)
            below = free_solution < lower[free_indices]
            above = free_solution > upper[free_indices]
            if np.any(below | above):
                direction = free_solution - solution[free_indices]
                fractions = np.full(free_indices.size, np.inf)
                fractions[below] = (
                    lower[free_indices[below]] - solution[free_indices[below]]
                ) / direction[below]
                fractions[above] = (
                    upper[free_indices[above]] - solution[free_indices[above]]
                ) / direction[above]
                blocking = int(np.argmin(fractions))
                fraction = np.clip(fractions[blocking], 0.0, 1.0)
                solution[free_indices] = np.clip(
                    solution[free_indices] + fraction * direction,
                    lower[free_indices],
                    upper[free_indices],
                )
                joint = free_indices[blocking]
                status[joint] = -1 if below[blocking] else 1
                solution[joint] = lower[joint] if below[blocking] else upper[joint]
                continue
            solution[free_indices] = free_solution

        objective_gradient = hessian @ solution - linear_term
        lower_violation = (status == -1) & (objective_gradient < -kkt_tolerance)
        upper_violation = (status == 1) & (objective_gradient > kkt_tolerance)
        if not np.any(lower_violation | upper_violation):
            break
        violations = np.zeros(solution.size)
        violations[lower_violation] = -objective_gradient[lower_violation]
        violations[upper_violation] = objective_gradient[upper_violation]
        status[int(np.argmax(violations))] = 0
    return solution
