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
