"""Nested invalid solver outputs must not hide independent valid candidates."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from holistic_motion.kinematics import FEPContinuousTracker, SRSContinuousTracker


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize("kind", ["complex", "zero_imaginary", "cycle"])
@pytest.mark.parametrize("with_valid", [False, True])
def test_tracker_filters_nested_invalid_candidates_without_losing_history(
    tracker_type, kind, with_valid
):
    nested = np.empty((), dtype=object)
    nested[()] = (
        nested
        if kind == "cycle"
        else np.complex128(0.25 + (1j if kind == "complex" else 0j))
    )
    wrapped = np.empty((), dtype=object)
    wrapped[()] = nested
    malformed = np.zeros(7, dtype=object)
    malformed[0] = wrapped

    class Solver:
        compatible = True
        joint_limits = (np.full(7, -3.0), np.full(7, 3.0))

        @staticmethod
        def configuration(_joints):
            return SimpleNamespace(shoulder=1, elbow=1, wrist=1, redundancy=0.0)

        @staticmethod
        def forward(_joints):
            return np.eye(4)

        @staticmethod
        def jacobian(_joints):
            return np.eye(6, 7)

        @staticmethod
        def solve(_target, _seed, _method):
            return [malformed, np.full(7, 0.1)] if with_valid else [malformed]

    tracker = tracker_type(Solver(), np.zeros(7))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if with_valid:
                result = tracker.solve(np.eye(4), dt=1.0)
                np.testing.assert_array_equal(result.joints, np.full(7, 0.1))
                assert result.candidate_count == 1
                assert tracker.frame_index == 1
            else:
                with pytest.raises(ValueError, match="did not converge"):
                    tracker.solve(np.eye(4), dt=1.0)
                np.testing.assert_array_equal(tracker.joints, np.zeros(7))
                np.testing.assert_array_equal(tracker.velocity, np.zeros(7))
                assert tracker.frame_index == 0
    finally:
        # The caller owns the deliberately cyclic fixture.
        nested[()] = 0.0


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
def test_tracker_keeps_nested_real_and_strided_candidates(tracker_type):
    nested = np.empty((), dtype=object)
    nested[()] = np.array(0.25)
    values = np.zeros(14, dtype=object)[::2]
    values[0] = nested
    tracker = object.__new__(tracker_type)
    candidates = list(tracker._finite_candidates([values]))
    assert len(candidates) == 1
    np.testing.assert_array_equal(candidates[0], [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def output_solver():
    class Solver:
        compatible = True
        joint_limits = (np.full(7, -3.0), np.full(7, 3.0))

        def __init__(self):
            self.candidates = [np.zeros(7), np.full(7, 0.25)]

        @staticmethod
        def configuration(_joints):
            return SimpleNamespace(shoulder=1, elbow=1, wrist=1, redundancy=0.0)

        @staticmethod
        def forward(_joints):
            return np.eye(4)

        @staticmethod
        def jacobian(_joints):
            return np.eye(6, 7)

        def solve(self, _target, _seed, _method):
            return self.candidates

    return Solver()


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize(
    "jacobian,error_type",
    [
        pytest.param(np.zeros(7), ValueError, id="vector"),
        pytest.param(np.zeros((7, 6)), ValueError, id="transposed"),
        pytest.param(np.zeros((0, 7)), ValueError, id="empty"),
        pytest.param(np.full((6, 7), np.nan), ValueError, id="nan"),
        pytest.param(np.full((6, 7), np.inf), ValueError, id="infinity"),
        pytest.param(np.eye(6, 7, dtype=complex), TypeError, id="complex"),
    ],
)
def test_tracker_rejects_invalid_jacobians_without_updating_history(
    tracker_type, jacobian, error_type, output_solver, monkeypatch
):
    tracker = tracker_type(output_solver, np.zeros(7))
    with monkeypatch.context() as patch:
        patch.setattr(output_solver, "jacobian", lambda _joints: jacobian)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(error_type, match="Jacobian"):
                tracker.solve(np.eye(4), dt=1.0)
    assert tracker.frame_index == 0
    np.testing.assert_array_equal(tracker.joints, np.zeros(7))
    assert tracker.solve(np.eye(4), dt=1.0).minimum_singular_value == 1.0
    assert tracker.frame_index == 1


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize("with_valid", [False, True])
@pytest.mark.parametrize(
    "pose",
    [
        pytest.param(np.diag([-1.0, 1.0, 1.0, 1.0]), id="reflection"),
        pytest.param(np.diag([2.0, 2.0, 2.0, 1.0]), id="scaled-rotation"),
        pytest.param(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.25, 0.0, 0.0, 1.0],
                ]
            ),
            id="projective-row",
        ),
        pytest.param(np.eye(4, dtype=complex), id="complex"),
        pytest.param(np.full((4, 4), np.nan), id="nan"),
    ],
)
def test_tracker_filters_invalid_forward_poses(
    tracker_type, with_valid, pose, output_solver, monkeypatch
):
    tracker = tracker_type(output_solver, np.zeros(7))
    if not with_valid:
        output_solver.candidates = [np.zeros(7)]
    monkeypatch.setattr(
        output_solver, "forward", lambda joints: pose if joints[0] == 0.0 else np.eye(4)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if with_valid:
            result = tracker.solve(np.eye(4), dt=1.0)
            np.testing.assert_array_equal(result.joints, np.full(7, 0.25))
            assert result.candidate_count == 1
        else:
            with pytest.raises(ValueError, match="candidate"):
                tracker.solve(np.eye(4), dt=1.0)
            assert tracker.frame_index == 0
            np.testing.assert_array_equal(tracker.joints, np.zeros(7))


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
def test_tracker_preserves_forward_backend_exceptions(
    tracker_type, output_solver, monkeypatch
):
    tracker = tracker_type(output_solver, np.zeros(7))

    def broken_backend(_joints):
        raise ValueError("backend query failed")

    monkeypatch.setattr(output_solver, "forward", broken_backend)
    with pytest.raises(ValueError, match="backend query failed"):
        tracker.solve(np.eye(4), dt=1.0)
    assert tracker.frame_index == 0


@pytest.fixture
def scratch_solver():
    class Solver:
        compatible = True
        joint_limits = (np.full(7, -3.0), np.full(7, 3.0))

        def __init__(self):
            self.operation = None
            self.fail = False

        def scratch(self, values, operation):
            if self.operation in (operation, "all"):
                values[...] = np.nan
                if self.fail:
                    raise RuntimeError("backend scratch failure")

        def configuration(self, joints):
            result = SimpleNamespace(
                shoulder=1, elbow=1, wrist=1, redundancy=float(joints[0])
            )
            self.scratch(joints, "configuration")
            return result

        def forward(self, joints):
            result = np.eye(4)
            result[0, 3] = joints[0]
            self.scratch(joints, "forward")
            return result

        def jacobian(self, joints):
            result = np.zeros((6, 7))
            result[0, 0] = 1.0
            self.scratch(joints, "jacobian")
            return result

        def solve(self, target, seed, _method):
            result = seed.copy()
            result[0] = target[0, 3]
            self.scratch(target, "target")
            self.scratch(seed, "seed")
            return [result]

    return Solver()


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize(
    "operation", ["seed", "target", "configuration", "forward", "jacobian", "all"]
)
def test_tracker_owns_backend_query_inputs(tracker_type, operation, scratch_solver):
    initial = np.zeros(7)
    initial.setflags(write=False)
    tracker = tracker_type(scratch_solver, initial)
    target = np.eye(4)
    target[0, 3] = 0.2
    target.setflags(write=False)
    scratch_solver.operation = operation
    result = tracker.solve(target, dt=1.0)
    expected = np.zeros(7)
    expected[0] = 0.2
    np.testing.assert_array_equal(initial, np.zeros(7))
    assert target[0, 3] == 0.2
    np.testing.assert_array_equal(result.joints, expected)
    np.testing.assert_array_equal(result.velocity, expected)
    np.testing.assert_array_equal(tracker.joints, expected)
    assert result.redundancy == 0.2
    assert tracker.frame_index == 1


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize(
    "operation", ["seed", "target", "configuration", "forward", "jacobian"]
)
def test_backend_scratch_failure_preserves_tracker_history(
    tracker_type, operation, scratch_solver
):
    tracker = tracker_type(scratch_solver, np.zeros(7))
    target = np.eye(4)
    target[0, 3] = 0.1
    previous = tracker.solve(target, dt=1.0)
    scratch_solver.operation = operation
    scratch_solver.fail = True
    target[0, 3] = 0.2
    with pytest.raises(RuntimeError, match="backend scratch failure"):
        tracker.solve(target, dt=1.0)
    np.testing.assert_array_equal(tracker.joints, previous.joints)
    np.testing.assert_array_equal(tracker.velocity, previous.velocity)
    assert tracker.redundancy == previous.redundancy
    assert tracker.frame_index == 1


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize("fail", [False, True])
def test_reset_keeps_caller_coordinates_out_of_backend_scratch(
    tracker_type, fail, scratch_solver
):
    tracker = tracker_type(scratch_solver, np.zeros(7))
    initial = np.full(7, 0.25)
    initial.setflags(write=False)
    scratch_solver.operation = "configuration"
    scratch_solver.fail = fail
    if fail:
        with pytest.raises(RuntimeError, match="backend scratch failure"):
            tracker.reset(initial)
    else:
        tracker.reset(initial)
    np.testing.assert_array_equal(initial, np.full(7, 0.25))
    np.testing.assert_array_equal(tracker.joints, np.zeros(7) if fail else initial)


@pytest.mark.parametrize("tracker_type", [SRSContinuousTracker, FEPContinuousTracker])
@pytest.mark.parametrize("attribute", ["_velocity", "_frame_index"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_interrupted_history_update_keeps_previous_frame(
    tracker_type, attribute, error_type, scratch_solver
):
    class InterruptibleTracker(tracker_type):
        interrupt_attribute = None

        def __setattr__(self, name, value):
            if name == self.interrupt_attribute:
                self.interrupt_attribute = None
                raise error_type("interrupted history update")
            super().__setattr__(name, value)

    tracker = InterruptibleTracker(scratch_solver, np.zeros(7))
    target = np.eye(4)
    target[0, 3] = 0.1
    first = tracker.solve(target, dt=1.0)
    tracker.interrupt_attribute = attribute
    target[0, 3] = 0.2

    with pytest.raises(error_type, match="interrupted history update"):
        tracker.solve(target, dt=1.0)

    np.testing.assert_array_equal(tracker.joints, first.joints)
    np.testing.assert_array_equal(tracker.velocity, first.velocity)
    assert tracker.configuration == first.configuration
    assert tracker.redundancy == first.redundancy
    assert tracker.frame_index == 1

    result = tracker.solve(target, dt=1.0)
    assert result.joints[0] == 0.2
    assert result.velocity[0] == 0.1
    assert result.acceleration[0] == 0.0
    assert tracker.frame_index == 2
