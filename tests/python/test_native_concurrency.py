"""Native query workspaces and Python result ownership remain independent."""

import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_fep_batch import _seven_axis_urdf
from test_native_kinematics_inputs import solver_data  # noqa: F401
from test_srs_kinematics import _write_ideal_srs_urdf, _write_offset_srs_urdf


def _write_chain(path, count):
    links = "".join(f'<link name="link{i}"/>' for i in range(count + 1))
    joints = "".join(
        f'<joint name="joint{i}" type="revolute">'
        f'<parent link="link{i}"/><child link="link{i + 1}"/>'
        f'<origin xyz="0.02 -0.01 0.1"/>'
        f'<axis xyz="{("1 0 0", "0 1 0", "0 0 1")[i % 3]}"/>'
        '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        for i in range(count)
    )
    path.write_text('<robot name="concurrent">' + links + joints + "</robot>")


def test_shared_common_ik_returns_independent_reports(solver_data):  # noqa: F811
    solver, original = solver_data
    states = [original + 0.03 * np.sin(i + np.arange(original.size)) for i in range(32)]
    targets = [solver.forward(state) for state in states]
    expected = [
        solver.solve_ik(target, state) for target, state in zip(targets, states)
    ]
    assert all(report.success for report in expected)

    def query(index):
        seed = states[index].copy()
        report = solver.solve_ik(targets[index], seed)
        np.testing.assert_array_equal(seed, states[index])
        seed.fill(np.nan)
        return report

    with ThreadPoolExecutor(4) as workers:
        actual = list(workers.map(query, range(len(states))))
    for target, result, reference in zip(targets, actual, expected):
        assert result.success and result.status == reference.status
        assert result.algorithm == reference.algorithm
        assert result.distances == reference.distances
        np.testing.assert_array_equal(result.solutions, reference.solutions)
        np.testing.assert_allclose(
            solver.forward(result.solutions[0]), target, atol=1e-7
        )


@pytest.mark.parametrize("backend_name", ["CPU", "CUDA"])
def test_shared_fep_batches_use_independent_outputs(tmp_path, backend_name):
    import holistic_motion as hm

    path = tmp_path / "batch.urdf"
    path.write_text(_seven_axis_urdf())
    solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link7")
    if backend_name == "CUDA" and not solver.cuda_available:
        pytest.skip("requires an available CUDA backend")
    batches = [np.random.default_rng(i).uniform(-0.4, 0.4, (129, 7)) for i in range(8)]
    expected = [solver.forward_batch(batch, hm.FEPBackend.CPU) for batch in batches]
    backend = getattr(hm.FEPBackend, backend_name)
    with ThreadPoolExecutor(4) as workers:
        actual = list(
            workers.map(lambda batch: solver.forward_batch(batch, backend), batches)
        )
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=1e-12)
    actual[0].fill(np.nan)
    for result in actual[1:]:
        assert np.isfinite(result).all()


def _create_seven_axis_solver(tmp_path, family):
    import holistic_motion as hm

    path = tmp_path / "seven_axis.urdf"
    if family == "SRS":
        _write_ideal_srs_urdf(path)
        solver = hm.Robot(str(path)).kinematics
    else:
        _write_offset_srs_urdf(path)
        solver = hm.Robot(str(path)).create_fep_kinematics("link0", "link8")
    return solver


@pytest.fixture(params=["SRS", "FEP"])
def seven_axis_solver(tmp_path, request):
    return _create_seven_axis_solver(tmp_path, request.param), request.param


@pytest.mark.parametrize("method_name", ["SEEDED_NUMERICAL", "ALL_CONFIGURATIONS"])
def test_shared_seven_axis_solves_own_outputs(seven_axis_solver, method_name):
    import holistic_motion as hm

    solver, family = seven_axis_solver
    method = getattr(getattr(hm, family + "SolveMethod"), method_name)
    seed = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
    states = [seed + 0.01 * np.sin(i + np.arange(7)) for i in range(8)]
    targets = [solver.forward(state) for state in states]

    def query(index):
        if family == "SRS":
            report = solver.solve_detailed(targets[index], states[index], method)
            assert report.status == hm.SRSSolveStatus.SUCCESS
            return report.solutions
        return solver.solve(targets[index], states[index], method)

    expected = [query(i) for i in range(len(states))]
    with ThreadPoolExecutor(4) as workers:
        actual = list(workers.map(query, range(len(states))))
    for target, solutions, reference in zip(targets, actual, expected):
        assert solutions
        np.testing.assert_array_equal(solutions, reference)
        for solution in solutions:
            np.testing.assert_allclose(solver.forward(solution), target, atol=1e-5)
    assert all(
        not np.shares_memory(first, second)
        for first in actual[0]
        for result in actual[1:]
        for second in result
    )
    if family == "FEP":
        actual[0][0].fill(np.nan)
    assert all(
        np.isfinite(solution).all() for result in actual[1:] for solution in result
    )


@pytest.mark.parametrize(
    "family,query_name",
    [
        ("SRS", "solve"),
        ("SRS", "solve_user"),
        ("SRS", "solve_detailed"),
        ("SRS", "solve_detailed_user"),
        ("FEP", "solve"),
        ("FEP", "solve_configuration"),
    ],
)
def test_seven_axis_logging_exception_restores_gil(tmp_path, family, query_name):
    import holistic_motion as hm

    solver = _create_seven_axis_solver(tmp_path, family)
    seed = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
    target = solver.forward(seed)
    if query_name == "solve_configuration":
        configuration = solver.configuration(seed)
        query = lambda: solver.solve_configuration(target, configuration, seed)
    else:
        method = getattr(hm, family + "SolveMethod").SEEDED_NUMERICAL
        query = lambda: getattr(solver, query_name)(target, seed, method)

    failure = RuntimeError("seven-axis log handler failed")

    class FailingHandler(logging.Handler):
        def emit(self, record):
            raise failure

    try:
        hm.setup_logging("DEBUG", [FailingHandler()])
        with pytest.raises(RuntimeError) as caught:
            query()
        assert caught.value is failure
    finally:
        hm.reset_logging()
    result = query()
    if query_name.startswith("solve_detailed"):
        assert result.status == hm.SRSSolveStatus.SUCCESS
    else:
        assert len(result) > 0


def test_common_ik_allows_python_progress_during_native_work(tmp_path):
    path = tmp_path / "long_chain.urdf"
    _write_chain(path, 128)
    # A large interpreter switch interval excludes a handoff between the
    # Python call and the observation after it. The input matrices are tiny;
    # the other thread must progress while native IK owns the long computation.
    command = """
import sys
from threading import Event, Thread
import numpy as np
import holistic_motion as hm
solver = hm.Robot(sys.argv[1]).kinematics
seed = np.full(128, 0.1)
target = solver.forward(seed)
target[:3, 3] += 100.0
solver.solve_ik(target, seed)
ready, go = Event(), Event()
progress = []
def peer():
    ready.set()
    go.wait()
    progress.append(True)
worker = Thread(target=peer, daemon=True)
worker.start()
assert ready.wait(5)
previous = sys.getswitchinterval()
sys.setswitchinterval(10.0)
try:
    go.set()
    report = solver.solve_ik(target, seed)
    observed = bool(progress)
finally:
    sys.setswitchinterval(previous)
    go.set()
    worker.join(5)
assert not worker.is_alive()
assert report.status == hm.IKSolveStatus.NO_SOLUTION
assert observed, 'native IK blocked Python progress'
"""
    result = subprocess.run(
        [sys.executable, "-c", command, str(path)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ik_logging_callback_exception_recovers_the_gil(tmp_path):
    import holistic_motion as hm

    path = tmp_path / "logging.urdf"
    _write_chain(path, 1)
    solver = hm.Robot(str(path)).kinematics
    seed = np.array([0.2])
    target = solver.forward(seed)

    class QueryLogError(RuntimeError):
        pass

    failure = QueryLogError("query log handler failed")

    class FailingHandler(logging.Handler):
        def emit(self, record):
            raise failure

    try:
        hm.setup_logging("DEBUG", [FailingHandler()])
        with pytest.raises(QueryLogError) as caught:
            solver.solve_ik(target, seed)
        assert caught.value is failure
    finally:
        hm.reset_logging()
    assert solver.solve_ik(target, seed).success


@pytest.mark.parametrize("with_callbacks", [False, True])
def test_shared_optimizer_released_queries_own_outputs(with_callbacks):
    import holistic_motion as hm

    optimizer = hm.PathOptimizer([-2.0, -2.0], [2.0, 2.0])
    if with_callbacks:
        optimizer.set_state_validator(lambda state: np.isfinite(state).all())
        optimizer.set_state_cost(lambda state: float(state @ state))
        optimizer.set_state_cost_gradient(lambda state: 2.0 * state)
    options = hm.PathOptimizationOptions()
    options.max_iterations = 8
    options.state_cost_weight = 0.1 if with_callbacks else 0.0
    paths = [
        np.array([[-0.8, 0.0], [-0.2, 0.4 + 0.01 * index], [0.8, 0.0]])
        for index in range(16)
    ]
    expected = [optimizer.optimize(path, options) for path in paths]
    with ThreadPoolExecutor(4) as workers:
        actual = list(
            workers.map(
                lambda path: optimizer.optimize(path, options, release_gil=True), paths
            )
        )
    for result, reference in zip(actual, expected):
        assert result.success and reference.success
        np.testing.assert_array_equal(result.path, reference.path)
        assert result.statistics.final_objective == reference.statistics.final_objective
    assert all(
        not np.shares_memory(first, second)
        for first in actual[0].path
        for result in actual[1:]
        for second in result.path
    )


@pytest.mark.parametrize("stage", ["sampling", "validator", "cost", "gradient"])
def test_planning_released_callback_exception_recovers(stage):
    import holistic_motion as hm

    failure = RuntimeError("planning callback failed")

    def callback(state):
        raise failure

    if stage == "sampling":
        planner = hm.SamplingPlanner([-1.0], [1.0], callback)
        query = lambda: planner.plan([-0.8], [0.8], release_gil=True)
        reset = lambda: planner.set_state_validator(None)
    else:
        optimizer = hm.PathOptimizer([-1.0], [1.0])
        options = hm.PathOptimizationOptions()
        options.max_iterations = 2
        options.state_cost_weight = 0.0 if stage == "validator" else 1.0
        if stage == "gradient":
            optimizer.set_state_cost(lambda state: float(state @ state))
        setter = getattr(
            optimizer,
            "set_state_" + ("cost_gradient" if stage == "gradient" else stage),
        )
        setter(callback)
        query = lambda: optimizer.optimize(
            [[-0.8], [0.1], [0.8]], options, release_gil=True
        )
        reset = lambda: setter(None)
    with pytest.raises(RuntimeError) as caught:
        query()
    assert caught.value is failure
    reset()
    if stage != "sampling":
        options.state_cost_weight = 0.0
    assert query().success


@pytest.mark.parametrize("kind", ["validator", "cost", "cost_gradient"])
def test_planning_callback_snapshot_releases_python_references(kind):
    import gc
    import weakref

    import holistic_motion as hm

    optimizer = hm.PathOptimizer([-1.0], [1.0])
    options = hm.PathOptimizationOptions()
    options.state_cost_weight = 0.0 if kind == "validator" else 1.0
    options.max_iterations = 2
    if kind == "cost_gradient":
        optimizer.set_state_cost(lambda state: float(state @ state))

    def callback(state):
        if kind == "validator":
            return True
        if kind == "cost":
            return float(state @ state)
        return 2.0 * state

    retained = weakref.ref(callback)
    setter = getattr(optimizer, "set_state_" + kind)
    setter(callback)
    del callback
    gc.collect()
    assert retained() is not None
    assert optimizer.optimize([[-0.8], [0.1], [0.8]], options, release_gil=True).success
    setter(None)
    gc.collect()
    assert retained() is None


@pytest.mark.parametrize("query_kind", ["velocity", "path"])
def test_shared_null_space_queries_own_outputs(tmp_path, query_kind):
    import holistic_motion as hm

    solver = _create_seven_axis_solver(tmp_path, "SRS")
    planner = hm.NullSpacePlanner(solver)
    nominal = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
    states = [nominal + 0.002 * np.sin(i + np.arange(7)) for i in range(16)]
    direction = np.eye(7)[2]

    def query(index):
        state = states[index].copy()
        preference = direction.copy()
        if query_kind == "velocity":
            result = solver.null_space_velocity(state, preference)
        else:
            result = planner.plan(state, preference, 8, 0.001)
        np.testing.assert_array_equal(state, states[index])
        np.testing.assert_array_equal(preference, direction)
        state.fill(np.nan)
        preference.fill(np.nan)
        return result

    expected = [query(i) for i in range(len(states))]
    with ThreadPoolExecutor(4) as workers:
        actual = list(workers.map(query, range(len(states))))
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=2e-12, rtol=0.0)
    actual[0][0] = np.nan
    assert all(np.isfinite(result).all() for result in actual[1:])


@pytest.mark.parametrize("query_kind", ["velocity", "path"])
def test_null_space_queries_allow_python_progress(tmp_path, query_kind):
    path = tmp_path / "srs.urdf"
    _write_ideal_srs_urdf(path)
    command = """
import sys
from threading import Event, Thread
from time import monotonic
import numpy as np
import holistic_motion as hm
solver = hm.Robot(sys.argv[1]).kinematics
planner = hm.NullSpacePlanner(solver)
seed = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
direction = np.eye(7)[2]
if sys.argv[2] == 'path':
    query = lambda: planner.plan(seed, direction, 512, 1e-5)
else:
    query = lambda: solver.null_space_velocity(seed, direction)
query()
ready, go = Event(), Event()
progress = []
def peer():
    ready.set()
    go.wait()
    progress.append(True)
worker = Thread(target=peer, daemon=True)
worker.start()
assert ready.wait(5)
previous = sys.getswitchinterval()
sys.setswitchinterval(10.0)
try:
    go.set()
    # A fixed number of microsecond queries can finish before the OS schedules
    # the peer. Keep offering native GIL release points for a bounded interval;
    # the long switch interval still prevents Python bytecode from helping.
    deadline = monotonic() + 1.0
    while True:
        result = query()
        if progress or monotonic() >= deadline:
            break
    observed = bool(progress)
finally:
    sys.setswitchinterval(previous)
    go.set()
    worker.join(5)
assert not worker.is_alive()
assert np.isfinite(result).all()
assert observed, 'native null-space query blocked Python progress'
"""
    result = subprocess.run(
        [sys.executable, "-c", command, str(path), query_kind],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_null_space_logging_exception_recovers_the_gil(tmp_path):
    import holistic_motion as hm

    solver = _create_seven_axis_solver(tmp_path, "SRS")
    planner = hm.NullSpacePlanner(solver)
    seed = np.array([0.2, -0.35, 0.3, -0.6, 0.25, 0.4, -0.2])
    direction = np.eye(7)[2]
    failure = RuntimeError("null-space log handler failed")

    class FailingHandler(logging.Handler):
        def emit(self, record):
            raise failure

    try:
        hm.setup_logging("DEBUG", [FailingHandler()])
        with pytest.raises(RuntimeError) as caught:
            planner.plan(seed, direction, 2, 0.001)
        assert caught.value is failure
    finally:
        hm.reset_logging()
    assert len(planner.plan(seed, direction, 2, 0.001)) == 3
