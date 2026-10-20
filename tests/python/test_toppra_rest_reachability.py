"""The rest-bound specialization agrees with the general interval solver."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory
from holistic_motion.trajectory.toppra import _IntervalConstraints


class _GeneralToRest(ToppraTrajectory):
    def _solve_to_rest(self, a, b, projected, start_x):
        constraints = [
            _IntervalConstraints(left, right, cap=cap)
            for left, right, cap in zip(a, b, projected)
        ]
        return self._solve_general(constraints, start_x, 0.0)


@pytest.mark.parametrize("seed,dof,count", [(19, 1, 3), (57, 2, 4), (81, 7, 12)])
@pytest.mark.parametrize("grid_kind", ["uniform", "clustered", "default"])
@pytest.mark.parametrize("coordinate,clock", [(1.0, 1.0), (1e200, 1e-80), (1e-6, 1e6)])
@pytest.mark.parametrize("start_speed", [0.0, 0.01])
def test_rest_solver_matches_general_controllable_intervals(
    seed, dof, count, grid_kind, coordinate, clock, start_speed
):
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(count, dof))
    if dof > 1:
        points[:, -1] = 0.4
    velocity = generator.uniform(0.8, 1.5, dof)
    acceleration = generator.uniform(2.0, 4.0, dof)
    options = {"start_path_velocity": start_speed * clock, "grid_size": 65}
    if grid_kind == "uniform":
        options["gridpoints"] = np.linspace(0.0, 1.0, 65)
    elif grid_kind == "clustered":
        options["gridpoints"] = np.r_[
            0.0, np.geomspace(1e-7, 0.49, 24), 1.0 - np.geomspace(0.49, 1e-7, 24), 1.0
        ]
    arguments = (
        points * coordinate,
        (velocity * coordinate) * clock,
        ((acceleration * coordinate) * clock) * clock,
    )
    actual = ToppraTrajectory(*arguments, **options)
    reference = _GeneralToRest(*arguments, **options)
    np.testing.assert_array_equal(actual.result.gridpoints, reference.result.gridpoints)
    np.testing.assert_allclose(
        actual.result.path_speeds / clock,
        reference.result.path_speeds / clock,
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.result.times * clock,
        reference.result.times * clock,
        rtol=2e-10,
        atol=2e-12,
    )
    assert actual.result.path_speeds[0] == pytest.approx(
        start_speed * clock, rel=4.0 * np.finfo(float).eps, abs=0.0
    )
    assert actual.result.path_speeds[-1] == 0.0
    fractions = np.linspace(0.0037, 0.9961, 301)
    motion = actual.sample(fractions * actual.duration)
    expected = reference.sample(fractions * reference.duration)
    for order, (value, target) in enumerate(zip(motion, expected)):
        value, target = value / coordinate, target / coordinate
        for _ in range(order):
            value, target = value / clock, target / clock
        np.testing.assert_allclose(value, target, rtol=2e-9, atol=2e-10)
