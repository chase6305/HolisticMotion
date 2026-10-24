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


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("offset", [0.0, 0.3, -0.7])
@pytest.mark.parametrize("wrist", [0.0, np.pi, -np.pi, 1e-13, -1e-13, 1e-8, -1e-8])
def test_opw_wrist_preserves_physical_seed(direction, offset, wrist):
    import holistic_motion as hm

    params = hm.OPWParameters()
    params.a1, params.a2, params.b = 0.15, -0.10, 0.0
    params.c1, params.c2, params.c3, params.c4 = 0.45, 0.60, 0.10, 0.10
    params.offsets = [0.10, -0.20, 0.05, offset, 0.17, -0.11]
    params.rotation_directions = [-1, 1, -1, direction, -1, 1]
    reference = (
        np.array([0.20, -0.55, 0.75, 0.90, wrist, -0.35]) + params.offsets
    ) * params.rotation_directions
    solver = hm.OPWKinematics(params, *_limits())
    target = solver.forward(reference)
    solutions = solver.solve_all(target, reference)
    assert 1 <= len(solutions) <= 8
    for solution in solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        solver.inverse(target, reference), reference, atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("wrist", [0.65, 1e-8, -1e-8, np.pi - 1e-8, np.pi + 1e-8])
def test_opw_wrist_recovers_target_when_seed_azimuth_differs(wrist):
    import holistic_motion as hm

    params = hm.OPWParameters()
    params.a1, params.a2, params.b = 0.15, -0.10, 0.0
    params.c1, params.c2, params.c3, params.c4 = 0.45, 0.60, 0.10, 0.10
    params.rotation_directions = [1] * 6
    solver = hm.OPWKinematics(params, *_limits())
    reference = np.array([0.20, -0.55, 0.75, 0.90, wrist, -0.35])
    target = solver.forward(reference)
    seed = reference.copy()
    seed[3] += 0.8
    solutions = solver.solve_all(target, seed)
    # Joint coordinates are ill-conditioned near a wrist singularity; compare
    # the branch at the solver's 1e-4 deduplication scale and the pose strictly.
    assert any(np.allclose(value, reference, atol=1e-4, rtol=0) for value in solutions)
    for solution in solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10, rtol=0)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("offset", [0.3, -0.7])
def test_opw_shoulder_singularity_uses_internal_seed(direction, offset):
    import holistic_motion as hm

    params = hm.OPWParameters()
    params.a1, params.a2, params.b = 0.0, -0.10, 0.0
    params.c1, params.c2, params.c3, params.c4 = 0.45, 0.60, 0.10, 0.10
    params.offsets = [offset, 0, 0, 0, 0, 0]
    params.rotation_directions = [direction, 1, 1, 1, 1, 1]
    internal = np.array([0.20, 0, -np.arctan2(params.a2, params.c3), 0.90, 0.65, -0.35])
    reference = (internal + params.offsets) * params.rotation_directions
    solver = hm.OPWKinematics(params, *_limits())
    np.testing.assert_allclose(
        solver.inverse(solver.forward(reference), reference),
        reference,
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("index", [0, 3])
@pytest.mark.parametrize(
    "value", [1e200, -1e200, np.finfo(float).max, -np.finfo(float).max]
)
def test_opw_large_seeds_validate_final_wrapped_coordinates(index, value):
    import holistic_motion as hm

    params = hm.OPWParameters()
    params.a1, params.a2, params.b = 0.0 if index == 0 else 0.15, -0.10, 0.0
    params.c1, params.c2, params.c3, params.c4 = 0.45, 0.60, 0.10, 0.10
    params.offsets = [0.30, -0.20, 0.05, 0.30, 0.17, -0.11]
    params.rotation_directions = [-1, 1, -1, -1, -1, 1]
    internal = np.array([0.20, -0.55, 0.75, 0.90, 0.0, -0.35])
    if index == 0:
        internal[1:3] = [0.0, -np.arctan2(params.a2, params.c3)]
        internal[4] = 0.65
    reference = (internal + params.offsets) * params.rotation_directions
    solver = hm.OPWKinematics(params, *_limits())
    target = solver.forward(reference)
    seed = reference.copy()
    seed[index] = value
    original_seed = seed.copy()
    solutions = solver.solve_all(target, seed)
    assert 1 <= len(solutions) <= 8
    for solution in solutions:
        assert np.isfinite(solution).all()
        assert np.all(solution >= -2 * np.pi)
        assert np.all(solution <= 2 * np.pi)
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(seed, original_seed)


@pytest.mark.parametrize("arm_angle", [-1.50, -1.20])
@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize("elbow", [0.0, np.pi])
@pytest.mark.parametrize("wrist", [0.0, np.pi, 0.65, 1e-8, -1e-8])
@pytest.mark.parametrize("seed_shift", [0.8, 0.0003])
def test_opw_collinear_arm_recovers_target(arm_angle, scale, elbow, wrist, seed_shift):
    import holistic_motion as hm

    params = hm.OPWParameters()
    params.a1, params.a2, params.b = 0.15 * scale, -0.10 * scale, 0.0
    params.c1, params.c2, params.c3, params.c4 = (
        np.array([0.45, 0.60, 0.10, 0.10]) * scale
    )
    params.offsets = [0.10, -0.20, 0.05, 0.30, 0.17, -0.11]
    params.rotation_directions = [-1, 1, -1, 1, -1, 1]
    internal = np.array(
        [
            0.20,
            arm_angle,
            elbow - np.arctan2(params.a2, params.c3),
            0.90,
            wrist,
            -0.35,
        ]
    )
    reference = (internal + params.offsets) * params.rotation_directions
    seed = reference.copy()
    seed[3] += seed_shift
    solver = hm.OPWKinematics(params, *_limits())
    target = solver.forward(reference)
    solutions = solver.solve_all(target, seed)
    assert 1 <= len(solutions) <= 8
    for solution in solutions:
        np.testing.assert_allclose(solver.forward(solution), target, atol=1e-10, rtol=0)


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
