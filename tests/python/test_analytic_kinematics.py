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
    warmup = reference.copy()
    warmup[4] = 0.65
    assert solver.solve_all(solver.forward(warmup), warmup)
    target = solver.forward(reference)
    solutions = solver.solve_all(target, reference)
    assert any(
        abs(solution[4]) < 1e-12 and np.allclose(solution, reference, atol=1e-7, rtol=0)
        for solution in solutions
    )
    np.testing.assert_allclose(
        solver.inverse(target, reference), reference, atol=1e-7, rtol=0
    )


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("offset", [0.0, 0.3, -0.7])
@pytest.mark.parametrize("wrist", [0.0, np.pi, -np.pi, 1e-13, -1e-13, 1e-8, -1e-8])
def test_ur_wrist_preserves_physical_seed(direction, offset, wrist):
    import holistic_motion as hm

    params = hm.URParameters()
    params.d1, params.a2, params.a3 = 0.089159, -0.425, -0.39225
    params.d4, params.d5, params.d6 = 0.10915, 0.09465, 0.0823
    params.offsets = [0.10, -0.20, 0.05, 0.30, 0.17, offset]
    params.rotation_directions = [-1, 1, -1, 1, -1, direction]
    reference = (
        np.array([0.20, -0.55, 0.75, 0.90, wrist, -0.35]) + params.offsets
    ) * params.rotation_directions
    solver = hm.URKinematics(params, *_limits())
    target = solver.forward(reference)
    solutions = solver.solve_all(target, reference)
    assert 1 <= len(solutions) <= 8
    for solution in solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        solver.inverse(target, reference), reference, atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize("elbow", [0.0, np.pi])
@pytest.mark.parametrize("wrist", [0.0, np.pi, -np.pi, 1e-13, -1e-13, 1e-8, -1e-8])
@pytest.mark.parametrize("requested_roll", [-1.7, 1.6])
def test_ur_singular_wrist_keeps_arm_reachable(scale, elbow, wrist, requested_roll):
    import holistic_motion as hm

    params = hm.URParameters()
    params.d1, params.a2, params.a3 = np.array([0.089159, -0.425, -0.39225]) * scale
    params.d4, params.d5, params.d6 = np.array([0.10915, 0.09465, 0.0823]) * scale
    params.rotation_directions = [1] * 6
    solver = hm.URKinematics(params, *_limits())
    reference = np.array([0.20, -0.55, elbow, 0.90, wrist, -0.35])
    target = solver.forward(reference)
    seed = reference.copy()
    seed[5] = requested_roll
    original_seed = seed.copy()
    solutions = solver.solve_all(target, seed)
    assert 1 <= len(solutions) <= 8
    for solution in solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(seed, original_seed)
