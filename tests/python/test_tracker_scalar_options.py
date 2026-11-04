"""Tracker scalar options must own immutable, validated numeric values."""

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest
from holistic_motion.kinematics import FEPContinuousOptions, SRSContinuousOptions

SCALAR_FIELDS = (
    "position_weight",
    "velocity_weight",
    "acceleration_weight",
    "branch_switch_penalty",
    "joint_limit_weight",
    "singular_value_threshold",
    "position_tolerance",
    "angle_tolerance",
)


@pytest.mark.parametrize("options_type", [SRSContinuousOptions, FEPContinuousOptions])
@pytest.mark.parametrize("name", SCALAR_FIELDS)
def test_tracker_scalar_options_do_not_alias_numpy_inputs(options_type, name):
    value = np.array(0.25)
    options = options_type(**{name: value})
    value[...] = np.nan
    assert isinstance(getattr(options, name), float)
    assert getattr(options, name) == 0.25


@pytest.mark.parametrize("options_type", [SRSContinuousOptions, FEPContinuousOptions])
@pytest.mark.parametrize("name", SCALAR_FIELDS)
@pytest.mark.parametrize("value", [[0.25], np.array([[0.25]])])
def test_tracker_scalar_options_reject_non_scalar_shapes(options_type, name, value):
    with pytest.raises(ValueError, match="scalar"):
        options_type(**{name: value})


@pytest.mark.parametrize("options_type", [SRSContinuousOptions, FEPContinuousOptions])
@pytest.mark.parametrize("value", [np.float64(0.25), np.array(0.25), Decimal("0.25")])
def test_tracker_scalar_options_keep_real_scalar_inputs(options_type, value):
    options = options_type(position_weight=value)
    assert options.position_weight == 0.25
    assert isinstance(options.position_weight, float)


@pytest.mark.parametrize("kind", ["SRS", "FEP"])
def test_scoring_observes_the_public_options_property(kind):
    from holistic_motion import kinematics

    tracker_type = getattr(kinematics, kind + "ContinuousTracker")
    options_type = getattr(kinematics, kind + "ContinuousOptions")
    effective = options_type(
        position_weight=6.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        joint_limit_weight=0.0,
    )

    class Tracker(tracker_type):
        @property
        def options(self):
            return effective

    tracker = object.__new__(Tracker)
    tracker._options = options_type()
    tracker._lower, tracker._upper = np.full(7, -3.0), np.full(7, 3.0)
    tracker._configuration = (1, 1, 1)
    zero = np.zeros(7)
    assert tracker._score(np.ones(7), zero, zero, zero, (1, 1, 1), 1.0) == 42.0


@pytest.fixture(params=["SRS", "FEP"])
def stationary_tracker(request):
    from holistic_motion import kinematics

    tracker_type = getattr(kinematics, request.param + "ContinuousTracker")

    class Solver:
        compatible = True
        joint_limits = (np.full(7, -3.0), np.full(7, 3.0))

        def __init__(self):
            self.solve_calls = 0

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
            self.solve_calls += 1
            return [np.zeros(7)]

    class Tracker(tracker_type):
        # These scalar-contract tests do not require native solve-method enums.
        def _seeded_method(self):
            return None

        def _all_configurations_method(self):
            return None

    return Tracker(Solver(), np.zeros(7))


@pytest.mark.parametrize(
    "dt", [[0.1], np.array([0.1]), np.array([[0.1]]), np.array([0.1, 0.2])]
)
def test_tracker_time_step_rejects_non_scalar_shapes(stationary_tracker, dt):
    with pytest.raises(ValueError, match="scalar"):
        stationary_tracker.solve(np.eye(4), dt=dt)
    assert stationary_tracker.solver.solve_calls == 0
    assert stationary_tracker.frame_index == 0


@pytest.mark.parametrize("dt", [0.1j, np.complex128(0.1 + 0j), np.complex128(0.1 + 1j)])
def test_tracker_time_step_rejects_complex_values(stationary_tracker, dt):
    with pytest.raises(TypeError, match="real"):
        stationary_tracker.solve(np.eye(4), dt=dt)
    assert stationary_tracker.solver.solve_calls == 0
    assert stationary_tracker.frame_index == 0


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.1,
        np.nan,
        np.inf,
        Decimal("NaN"),
        Decimal("Infinity"),
        pytest.param(10**5000, id="overflowing-integer"),
        object(),
    ],
)
def test_tracker_time_step_rejects_invalid_scalars(stationary_tracker, dt):
    with pytest.raises(ValueError, match="dt"):
        stationary_tracker.solve(np.eye(4), dt=dt)
    assert stationary_tracker.solver.solve_calls == 0
    assert stationary_tracker.frame_index == 0


@pytest.mark.parametrize(
    "dt",
    [
        np.float64(0.1),
        np.array(0.1),
        Decimal("0.1"),
        np.array(Decimal("0.1"), dtype=object),
    ],
)
def test_tracker_time_step_accepts_real_scalars(stationary_tracker, dt):
    result = stationary_tracker.solve(np.eye(4), dt=dt)
    assert stationary_tracker.frame_index == 1
    np.testing.assert_array_equal(result.joints, np.zeros(7))
    np.testing.assert_array_equal(result.velocity, np.zeros(7))
