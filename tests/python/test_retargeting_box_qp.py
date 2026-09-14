"""Check bounded joint updates against an independent exhaustive QP oracle."""

from itertools import product

import numpy as np
import pytest
from holistic_motion.kit.retargeting import PinkRetargetingSolver


def enumerate_minimum(hessian, gradient, lower, upper):
    candidates = []
    for status in product((-1, 0, 1), repeat=gradient.size):
        status = np.asarray(status)
        free = status == 0
        candidate = np.where(status == -1, lower, upper).copy()
        if np.any(free):
            candidate[free] = np.linalg.solve(
                hessian[np.ix_(free, free)],
                gradient[free] - hessian[np.ix_(free, ~free)] @ candidate[~free],
            )
        if np.all(candidate >= lower - 1e-10) and np.all(candidate <= upper + 1e-10):
            candidates.append(candidate)
    return min(candidates, key=lambda q: 0.5 * q @ hessian @ q - gradient @ q)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("fixed_joint", [False, True])
def test_coupled_joint_bounds_reach_the_qp_minimum(order, direction, fixed_joint):
    # A positive-definite, strongly coupled task where clipping every violating
    # coordinate of a reduced minimizer cycles between incompatible active sets.
    basis = np.array([[180.0, -180.0, -150.0], [2.0, -2.0, -1.0], [-3.0, 4.0, 4.0]])
    hessian = (basis.T @ basis + np.eye(3))[np.ix_(order, order)]
    gradient = direction * np.array([16.0, 0.0, -5.0])[list(order)]
    lower, upper = -np.ones(3), np.ones(3)
    if fixed_joint:
        # A fixed, coupled coordinate can arise from zero velocity or emergency
        # braking at a position bound. Its derivative need not be zero.
        coupling = np.array([2.0, -1.0, 3.0])
        expanded = np.zeros((4, 4))
        expanded[:3, :3] = hessian
        expanded[:3, 3] = expanded[3, :3] = coupling
        expanded[3, 3] = 100.0
        hessian = expanded
        gradient = np.r_[gradient + 0.25 * coupling, -100.0]
        lower, upper = np.r_[lower, 0.25], np.r_[upper, 0.25]

    expected = enumerate_minimum(hessian, gradient, lower, upper)
    actual = PinkRetargetingSolver._solve_box_qp(hessian, gradient, lower, upper)

    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert np.all(actual >= lower)
    assert np.all(actual <= upper)


def test_box_qp_matches_exhaustive_minima_across_coupling_and_fixed_bounds():
    generator = np.random.default_rng(731)
    for index in range(64):
        basis = generator.normal(size=(4, 4)) * 10 ** generator.uniform(-1, 3, (4, 1))
        hessian = basis.T @ basis + 0.01 * np.eye(4)
        gradient = generator.normal(size=4) * 10
        lower = -generator.uniform(0.01, 1.0, 4)
        upper = generator.uniform(0.01, 1.0, 4)
        if index % 2 == 0:
            lower[0] = upper[0] = 0.0
        expected = enumerate_minimum(hessian, gradient, lower, upper)
        actual = PinkRetargetingSolver._solve_box_qp(hessian, gradient, lower, upper)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
