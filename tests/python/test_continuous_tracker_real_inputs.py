from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest


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
        return [np.zeros(7)]


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_complex_tracker_reset_preserves_history(prefix, dtype, imaginary):
    from holistic_motion import kinematics

    tracker_type = getattr(kinematics, prefix + "ContinuousTracker")
    tracker = tracker_type(Solver(), np.zeros(7))
    tracker.solve(np.eye(4), 0.01)
    invalid = np.full(7, np.complex128(0.25 + imaginary * 1j), dtype=dtype)
    with pytest.raises(TypeError, match="real"):
        tracker.reset(invalid)
    np.testing.assert_array_equal(tracker.joints, np.zeros(7))
    np.testing.assert_array_equal(tracker.velocity, np.zeros(7))
    assert tracker.frame_index == 1
    with pytest.raises(TypeError, match="real"):
        tracker_type(Solver(), invalid)


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize("name", ["max_velocity", "max_acceleration"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object])
def test_complex_tracker_limits_are_rejected(prefix, name, dtype):
    from holistic_motion import kinematics

    options_type = getattr(kinematics, prefix + "ContinuousOptions")
    invalid = np.full(7, np.complex128(1.0 + 1j), dtype=dtype)
    with pytest.raises(TypeError, match="real"):
        options_type(**{name: invalid})


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize(
    "name",
    [
        "position_weight",
        "velocity_weight",
        "acceleration_weight",
        "branch_switch_penalty",
        "joint_limit_weight",
        "singular_value_threshold",
        "position_tolerance",
        "angle_tolerance",
    ],
)
def test_complex_tracker_scalar_option_is_rejected(prefix, name):
    from holistic_motion import kinematics

    options_type = getattr(kinematics, prefix + "ContinuousOptions")
    with pytest.raises(TypeError, match="real"):
        options_type(**{name: np.complex128(1.0 + 1j)})


@pytest.fixture
def result_values():
    return {
        "joints": np.zeros(7),
        "velocity": np.zeros(7),
        "acceleration": np.zeros(7),
        "configuration": (1, 1, 1),
        "redundancy": 0.0,
        "branch_changed": False,
        "near_singularity": False,
        "minimum_singular_value": 1.0,
        "position_error": 0.0,
        "angle_error": 0.0,
        "candidate_count": 1,
    }


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize(
    "name",
    [
        "joints",
        "velocity",
        "acceleration",
        "redundancy",
        "position_error",
        "angle_error",
        "minimum_singular_value",
    ],
)
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_complex_tracker_result_is_rejected(prefix, name, imaginary, result_values):
    from holistic_motion import kinematics

    values = result_values
    invalid = np.complex128(0.25 + imaginary * 1j)
    values[name] = (
        np.full(7, invalid)
        if name in ("joints", "velocity", "acceleration")
        else invalid
    )
    result_type = getattr(kinematics, prefix + "ContinuousResult")
    with pytest.raises(TypeError, match="real"):
        result_type(**values)


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize(
    "name", ["redundancy", "minimum_singular_value", "position_error", "angle_error"]
)
@pytest.mark.parametrize(
    "value", [[0.25], [0.25, 0.5], np.array([0.25]), np.array([[0.25]])]
)
def test_tracker_result_metrics_reject_non_scalar_shapes(
    prefix, name, value, result_values
):
    from holistic_motion import kinematics

    result_type = getattr(kinematics, prefix + "ContinuousResult")
    with pytest.raises(ValueError, match="scalar"):
        result_type(**{**result_values, name: value})


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize(
    "name", ["redundancy", "minimum_singular_value", "position_error", "angle_error"]
)
@pytest.mark.parametrize(
    "value",
    [
        np.float64(0.25),
        np.array(0.25),
        Decimal("0.25"),
        np.array(Decimal("0.25"), dtype=object),
    ],
)
def test_tracker_result_metrics_accept_real_scalar_values(
    prefix, name, value, result_values
):
    from holistic_motion import kinematics

    result_type = getattr(kinematics, prefix + "ContinuousResult")
    result = result_type(**{**result_values, name: value})
    assert type(getattr(result, name)) is float
    assert getattr(result, name) == 0.25


@pytest.mark.parametrize("prefix", ["SRS", "FEP"])
@pytest.mark.parametrize(
    "name", ["redundancy", "minimum_singular_value", "position_error", "angle_error"]
)
def test_tracker_result_metrics_own_numpy_scalars(prefix, name, result_values):
    from holistic_motion import kinematics

    value = np.array(0.25)
    result_type = getattr(kinematics, prefix + "ContinuousResult")
    result = result_type(**{**result_values, name: value})
    value[...] = np.nan
    assert getattr(result, name) == 0.25
