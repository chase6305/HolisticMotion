"""Early-stop search should not generate unused initial configurations."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    CuroboRetargetingSolver,
    PinkRetargetingSolver,
    PostureTask,
)


def make_solver(tmp_path, continuous=False, **options):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    urdf = tmp_path / "lazy_seeds.urdf"
    kind = "continuous" if continuous else "revolute"
    limits = "" if continuous else 'lower="-1" upper="1"'
    urdf.write_text(f"""<robot name="lazy_seeds">
        <link name="base"/><link name="tool"/>
        <joint name="joint" type="{kind}">
          <parent link="base"/><child link="tool"/><axis xyz="0 0 1"/>
          <limit {limits} velocity="0.01" effort="1"/>
        </joint></robot>""")
    solver = CuroboRetargetingSolver(
        urdf,
        {"left_hand": "tool"},
        {"left_arm": ["joint"]},
        num_seeds=4,
        seed_spread=0.4,
        sampler_seed=7,
        stop_on_success=True,
        posture_task=PostureTask(cost=0.0),
        **options,
    )
    solver.prepare("left_arm")
    return solver


def configuration(solver, angle):
    return np.asarray(
        solver.pin.integrate(solver.model, solver._neutral_q, np.array([angle]))
    )


def target(angle):
    pose = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return {"left_hand": pose}


@pytest.mark.parametrize("continuous", [False, True])
def test_primary_success_does_not_generate_alternative_seeds(
    tmp_path, monkeypatch, continuous
):
    solver = make_solver(tmp_path, continuous)
    seed = configuration(solver, 0.3)

    def unexpected(*args, **kwargs):
        raise AssertionError("successful primary seed must not generate alternatives")

    monkeypatch.setattr(solver.pin, "integrate", unexpected)
    result = solver.solve(target(0.3), seed=seed)
    assert result.success
    assert solver.last_num_seeds_evaluated == 1
    assert solver.last_seed_index == 0
    np.testing.assert_allclose(result.configuration, seed, atol=1e-15)


@pytest.mark.parametrize("continuous", [False, True])
def test_neutral_success_does_not_access_random_samples(tmp_path, continuous):
    solver = make_solver(tmp_path, continuous)

    class UnusedSamples:
        def __getitem__(self, key):
            raise AssertionError("neutral success must not request random seeds")

    solver._seed_samples = UnusedSamples()
    result = solver.solve(
        target(0.0), seed=configuration(solver, 0.8), max_iterations=1
    )
    assert result.success
    assert solver.last_num_seeds_evaluated == 2
    assert solver.last_seed_index == 1
    np.testing.assert_allclose(result.configuration, solver._neutral_q, atol=1e-15)


@pytest.mark.parametrize("continuous", [False, True])
def test_exhausted_search_preserves_candidate_order_and_result(
    tmp_path, monkeypatch, continuous
):
    solver = make_solver(tmp_path, continuous)
    seed = configuration(solver, 0.8)
    expected = solver._seed_bank(seed)
    calls = []

    def cost(q):
        calls.append(q.copy())
        return 1.0

    solver.collision_cost = cost
    solver.collision_gradient = lambda q: np.zeros(solver.model.nv)
    # Generating the stream must not depend on warm-start updates between trials.
    seen = []
    original = PinkRetargetingSolver._solve_seed

    def solve_seed(self, *args, **kwargs):
        seen.append(kwargs["seed"].copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PinkRetargetingSolver, "_solve_seed", solve_seed)
    first = solver.solve(target(0.2), seed=seed, max_iterations=1)
    assert len(calls) == first.collision_evaluations
    np.testing.assert_allclose(seen, expected, atol=1e-15)
    assert not first.success
    assert solver.last_num_seeds_evaluated == 4
    index = solver.last_seed_index
    solver.stop_on_success = False
    seen.clear()
    calls.clear()
    second = solver.solve(target(0.2), seed=seed, max_iterations=1)
    assert len(calls) == second.collision_evaluations
    np.testing.assert_allclose(seen, expected, atol=1e-15)
    np.testing.assert_allclose(first.configuration, second.configuration, atol=1e-15)
    assert first.objective == second.objective
    assert solver.last_seed_index == index


@pytest.mark.parametrize("continuous", [False, True])
def test_later_candidate_generation_failure_restores_solver_state(tmp_path, continuous):
    solver = make_solver(tmp_path, continuous)
    initial = configuration(solver, -0.3)
    solver.reset(initial)
    solver._last_velocity[:] = 0.02
    solver.last_seed_index = 3
    solver.last_num_seeds_evaluated = 4
    at_failure = []

    class FailingSamples:
        def __getitem__(self, key):
            at_failure.append(solver._last_q.copy())
            raise RuntimeError("candidate generation failed")

    solver._seed_samples = FailingSamples()
    with pytest.raises(RuntimeError, match="candidate generation failed"):
        solver.solve(target(-0.8), seed=configuration(solver, 0.8), max_iterations=1)
    assert len(at_failure) == 1
    assert not np.allclose(at_failure[0], initial)
    np.testing.assert_allclose(solver._last_q, initial, atol=1e-15)
    np.testing.assert_array_equal(solver._last_velocity, [0.02])
    assert solver.last_seed_index == 3
    assert solver.last_num_seeds_evaluated == 4
