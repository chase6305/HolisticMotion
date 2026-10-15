"""Native motion scalar inputs follow the same real-valued contract as arrays."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("factory", [hm.JointTrajectory3, hm.RnTrajectory])
@pytest.mark.parametrize("field", ["blend_tolerance", "minimum_duration"])
@pytest.mark.parametrize("bad", [np.complex128(0.1 + 1j), np.array(0.1 + 1j)])
def test_joint_trajectory_constructor_rejects_complex_options(factory, field, bad):
    with pytest.raises(TypeError, match="real"):
        factory(
            [[0, 0, 0], [0.2, 0.1, -0.1]], [1] * 3, [1] * 3, [1] * 3, **{field: bad}
        )


@pytest.mark.parametrize(
    "method", ["position", "velocity", "acceleration", "jerk", "state"]
)
@pytest.mark.parametrize("bad", [np.complex128(0.1 + 1j), np.array(0.1 + 1j)])
def test_joint_trajectory_rejects_complex_scalar_time(method, bad):
    trajectory = hm.RnTrajectory([[0.0], [0.2]], [1.0], [1.0], [1.0])
    with pytest.raises(TypeError, match="real"):
        getattr(trajectory, method)(bad)


@pytest.mark.parametrize("bad", [np.complex128(2.0 + 1j), np.array(2.0 + 1j)])
def test_rejected_minimum_duration_preserves_timing(bad):
    trajectory = hm.RnTrajectory([[0.0], [0.2]], [1.0], [1.0], [1.0])
    duration = trajectory.duration
    with pytest.raises(TypeError, match="real"):
        trajectory.set_minimum_duration(bad)
    assert trajectory.duration == duration


@pytest.mark.parametrize("method", ["position", "state"])
def test_cartesian_trajectory_rejects_complex_scalar_time(method):
    end = np.eye(4)
    end[0, 3] = 0.2
    trajectory = hm.CartesianLineTrajectory(np.eye(4), end, [1] * 6, [1] * 6, [1] * 6)
    with pytest.raises(TypeError, match="real"):
        getattr(trajectory, method)(np.complex128(0.1 + 1j))


def test_cartesian_trajectory_rejects_complex_minimum_duration():
    end = np.eye(4)
    end[0, 3] = 0.2
    with pytest.raises(TypeError, match="real"):
        hm.CartesianLineTrajectory(
            np.eye(4),
            end,
            [1] * 6,
            [1] * 6,
            [1] * 6,
            minimum_duration=np.complex128(2.0 + 1j),
        )


@pytest.mark.parametrize("field", ["max_velocity", "max_acceleration", "max_jerk"])
@pytest.mark.parametrize("kind", [complex, object])
def test_trajectory_constraints_reject_complex_arrays(field, kind):
    args = {
        name: np.ones(3) for name in ("max_velocity", "max_acceleration", "max_jerk")
    }
    args[field] = np.array([np.complex128(1 + 1j)] * 3, dtype=kind)
    with pytest.raises(TypeError, match="real"):
        hm.TrajectoryConstraints(**args)


@pytest.mark.parametrize("kind", [complex, object])
def test_se3_constructor_rejects_complex_transform(kind):
    transform = np.eye(4).astype(kind)
    transform[0, 3] = np.complex128(0.2 + 1j)
    with pytest.raises(TypeError, match="real"):
        hm.SE3(transform)


def test_native_motion_still_accepts_real_scalars_and_integer_constraints():
    trajectory = hm.RnTrajectory(
        [[0], [1]], [1], [1], [1], minimum_duration=np.float64(2)
    )
    assert np.isfinite(trajectory.position(np.float64(0))).all()
    trajectory.set_minimum_duration(np.float64(3))
    assert trajectory.duration >= 3
    assert hm.TrajectoryConstraints([1], np.array([1, 2])[::2], (1,)).valid
    np.testing.assert_array_equal(hm.SE3(np.eye(4, dtype=int)).matrix(), np.eye(4))
