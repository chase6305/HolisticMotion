import numpy as np
import pytest


def _limits():
    return np.full(6, -2.0 * np.pi), np.full(6, 2.0 * np.pi)


@pytest.mark.parametrize("family", ["opw", "ur"])
def test_analytic_solver_round_trip_and_branch_count(family):
    import holistic_motion as hm

    lower, upper = _limits()
    if family == "opw":
        params = hm.OPWParameters()
        (params.a1, params.a2, params.b, params.c1, params.c2,
         params.c3, params.c4) = (0.15, -0.10, 0.0, 0.45,
                                      0.60, 0.10, 0.10)
        params.rotation_directions = [1] * 6
        solver = hm.OPWKinematics(params, lower, upper)
    else:
        params = hm.URParameters()
        (params.d1, params.a2, params.a3, params.d4,
         params.d5, params.d6) = (0.089159, -0.425, -0.39225,
                                      0.10915, 0.09465, 0.0823)
        params.rotation_directions = [1] * 6
        solver = hm.URKinematics(params, lower, upper)

    reference = np.array([0.25, -0.70, 0.85, -0.45, 0.65, -0.20])
    target = solver.forward(reference)
    solutions = solver.solve_all(target, reference)
    assert 1 <= len(solutions) <= 8
    nearest = solver.inverse(target, reference)
    np.testing.assert_allclose(solver.forward(nearest), target, atol=1e-7)


def test_analytic_parameters_validate_joint_limits():
    import holistic_motion as hm

    params = hm.URParameters()
    params.a2, params.a3 = -0.4, -0.3
    params.rotation_directions = [1] * 6
    with pytest.raises(ValueError, match="length-6"):
        hm.URKinematics(params, np.zeros(5), np.ones(5))

    lower, upper = _limits()
    params.rotation_directions = [0] * 6
    with pytest.raises(ValueError, match="axis directions"):
        hm.URKinematics(params, lower, upper)


@pytest.mark.parametrize(
    "reference",
    [
        [
            1.5506516568704931,
            1.739176493450888,
            1.708435863837303,
            1.3566595383461058,
            0.0,
            -1.5075893761186798,
        ],
        [
            0.8714005623823116,
            1.3548801678359264,
            1.555609976910908,
            -0.190975494656332,
            0.0,
            1.1633365826976252,
        ],
    ],
)
def test_ur_wrist_boundary_preserves_seeded_branch(reference):
    import holistic_motion as hm

    params = hm.URParameters()
    params.d1, params.a2, params.a3 = 0.089159, -0.425, -0.39225
    params.d4, params.d5, params.d6 = 0.10915, 0.09465, 0.0823
    params.rotation_directions = [1] * 6
    solver = hm.URKinematics(params, *_limits())
    reference = np.array(reference)
    target = solver.forward(reference)
    solutions = solver.solve_all(target, reference)
    assert any(
        solution[4] == 0.0 and np.allclose(solution, reference, atol=1e-7, rtol=0)
        for solution in solutions
    )
    np.testing.assert_allclose(
        solver.inverse(target, reference), reference, atol=1e-7, rtol=0
    )
