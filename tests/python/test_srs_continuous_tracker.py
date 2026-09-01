from types import SimpleNamespace

import numpy as np
import pytest


def _write_srs_urdf(path):
    origins = [
        ("0 0 0", "0 0 0", "0 0 1"),
        ("0 0 0.1025", "1.5707963268 0 3.1415926536", "0 0 1"),
        ("0 0.260 0", "1.5707963268 -1.5707963268 3.1415926536", "0 0 1"),
        ("0 0 0", "1.5707963268 0 0", "0 0 1"),
        ("0 0.166 0", "-1.5707963268 3.1415926536 0", "0 0 1"),
        ("0 0 0.098", "1.5707963268 0 0", "0 0 -1"),
        ("0 0 0", "-1.5707963268 3.1415926536 1.5707963268", "0 0 1"),
    ]
    links = "".join(f'<link name="link{i}"/>' for i in range(9))
    joints = []
    for index, (xyz, rpy, axis) in enumerate(origins):
        joints.append(f'''<joint name="joint{index}" type="revolute">
          <parent link="link{index}"/><child link="link{index + 1}"/>
          <origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{axis}"/>
          <limit lower="-3" upper="3" velocity="2" effort="10"/>
        </joint>''')
    joints.append("""<joint name="tool" type="fixed">
      <parent link="link7"/><child link="link8"/>
      <origin xyz="-0.066 0 0" rpy="0 -1.5707963268 0"/>
    </joint>""")
    path.write_text(
        f'<robot name="tracker_srs">{links}{"".join(joints)}</robot>',
        encoding="utf-8",
    )


def test_srs_tracker_preserves_branch_and_reports_continuity(tmp_path):
    import holistic_motion as hm
    from holistic_motion.kinematics import SRSContinuousTracker

    urdf = tmp_path / "srs.urdf"
    _write_srs_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    initial = np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15])
    tracker = SRSContinuousTracker(initial_joints=initial, solver=solver)
    branch = tracker.configuration

    previous = initial
    for offset in np.linspace(0.002, 0.02, 6):
        expected = initial.copy()
        expected[0] += offset
        result = tracker.solve(solver.forward(expected), dt=0.02)
        assert result.configuration == branch
        assert not result.branch_changed
        assert result.position_error < 1e-5
        assert result.angle_error < 1e-5
        assert result.candidate_count >= 1
        assert np.linalg.norm(result.joints - previous) < 0.1
        previous = result.joints


def test_srs_tracker_validates_dynamic_limits_and_inputs(tmp_path):
    import holistic_motion as hm
    from holistic_motion.kinematics import (
        SRSContinuousOptions,
        SRSContinuousTracker,
    )

    urdf = tmp_path / "srs.urdf"
    _write_srs_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    initial = np.array([0.15, -0.35, 0.25, -0.7, 0.2, 0.3, -0.15])
    tracker = SRSContinuousTracker(
        solver,
        initial,
        SRSContinuousOptions(max_velocity=0.01, max_acceleration=1.0),
    )
    moved = initial.copy()
    moved[0] += 0.2
    with pytest.raises(ValueError, match="continuity limits"):
        tracker.solve(solver.forward(moved), dt=0.01)
    with pytest.raises(ValueError, match="dt"):
        tracker.solve(solver.forward(initial), dt=0.0)
    with pytest.raises(ValueError, match="4x4"):
        tracker.solve(np.eye(3), dt=0.01)
    invalid_row = np.eye(4)
    invalid_row[3, 0] = 1.0
    with pytest.raises(ValueError, match="homogeneous"):
        tracker.solve(invalid_row, dt=0.01)
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="proper orthonormal"):
        tracker.solve(reflection, dt=0.01)
    with pytest.raises(ValueError, match="candidate_refresh_interval"):
        SRSContinuousOptions(candidate_refresh_interval=0)
    with pytest.raises(TypeError, match="branch_hysteresis_frames"):
        SRSContinuousOptions(branch_hysteresis_frames=1.5)
    with pytest.raises(TypeError, match="candidate_refresh_interval"):
        SRSContinuousOptions(candidate_refresh_interval=True)


def test_srs_tracker_dependencies_are_read_only(tmp_path):
    import holistic_motion as hm
    from holistic_motion.kinematics import (
        SRSContinuousOptions,
        SRSContinuousTracker,
    )

    urdf = tmp_path / "srs.urdf"
    _write_srs_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics
    tracker = SRSContinuousTracker(solver, np.zeros(7))

    with pytest.raises(AttributeError):
        tracker.solver = solver
    with pytest.raises(AttributeError):
        tracker.options = SRSContinuousOptions()


def test_srs_tracker_isolates_malformed_native_candidates():
    from holistic_motion.kinematics import SRSContinuousTracker

    class CandidateSolver:
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
            return [np.array([np.nan] * 7), np.zeros(6), np.zeros(7)]

    tracker = SRSContinuousTracker(CandidateSolver(), np.zeros(7))
    result = tracker.solve(np.eye(4), dt=0.01)

    assert np.array_equal(result.joints, np.zeros(7))
    assert result.candidate_count == 1


def test_srs_tracker_rejects_wrong_options_type(tmp_path):
    import holistic_motion as hm
    from holistic_motion.kinematics import SRSContinuousTracker

    urdf = tmp_path / "srs.urdf"
    _write_srs_urdf(urdf)
    solver = hm.Robot(str(urdf)).kinematics

    with pytest.raises(TypeError, match="options"):
        SRSContinuousTracker(solver, np.zeros(7), options=object())


def test_srs_tracker_options_and_results_are_deeply_immutable():
    from holistic_motion.kinematics import (
        SRSContinuousOptions,
        SRSContinuousTracker,
    )

    velocity_limit = np.ones(7)
    options = SRSContinuousOptions(max_velocity=velocity_limit)
    velocity_limit[0] = 99.0
    assert options.max_velocity[0] == 1.0

    class IdentitySolver:
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

    result = SRSContinuousTracker(IdentitySolver(), np.zeros(7), options).solve(
        np.eye(4), dt=0.01
    )
    with pytest.raises(ValueError, match="read-only"):
        result.joints[0] = 1.0


def test_srs_continuous_result_validates_public_snapshot():
    from holistic_motion.kinematics import SRSContinuousResult

    values = {
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
    with pytest.raises(ValueError, match="configuration"):
        SRSContinuousResult(**{**values, "configuration": (1, 1)})
    with pytest.raises(ValueError, match="finite"):
        SRSContinuousResult(**{**values, "redundancy": float("nan")})
    with pytest.raises(ValueError, match="non-negative"):
        SRSContinuousResult(**{**values, "position_error": -1.0})
    with pytest.raises(ValueError, match="positive"):
        SRSContinuousResult(**{**values, "candidate_count": 0})
    with pytest.raises(TypeError, match="branch_changed"):
        SRSContinuousResult(**{**values, "branch_changed": 1})
    with pytest.raises(TypeError, match="near_singularity"):
        SRSContinuousResult(**{**values, "near_singularity": "no"})


def test_srs_tracker_reset_is_transactional_and_owns_joint_limits():
    from holistic_motion.kinematics import SRSContinuousTracker

    class ResetSolver:
        compatible = True

        def __init__(self):
            self.lower = np.full(7, -3.0)
            self.upper = np.full(7, 3.0)
            self.fail_configuration = False

        @property
        def joint_limits(self):
            return self.lower, self.upper

        def configuration(self, joints):
            if self.fail_configuration:
                raise RuntimeError("configuration failure")
            return SimpleNamespace(
                shoulder=1, elbow=1, wrist=1, redundancy=float(joints[0])
            )

        @staticmethod
        def forward(_joints):
            return np.eye(4)

        @staticmethod
        def jacobian(_joints):
            return np.eye(6, 7)

        @staticmethod
        def solve(_target, _seed, _method):
            return [np.zeros(7)]

    solver = ResetSolver()
    initial = np.zeros(7)
    tracker = SRSContinuousTracker(solver, initial)
    solver.lower[:] = 1.0
    tracker.reset(np.full(7, 0.25))
    previous = tracker.joints
    previous_redundancy = tracker.redundancy
    solver.fail_configuration = True

    with pytest.raises(RuntimeError, match="configuration failure"):
        tracker.reset(np.full(7, 0.5))

    assert np.array_equal(tracker.joints, previous)
    assert tracker.redundancy == previous_redundancy


def test_srs_tracker_rolls_back_hysteresis_after_native_failure():
    from holistic_motion.kinematics import (
        SRSContinuousOptions,
        SRSContinuousTracker,
    )

    class FailingSolver:
        compatible = True
        joint_limits = (np.full(7, -3.0), np.full(7, 3.0))

        def __init__(self):
            self.jacobian_calls = 0

        @staticmethod
        def configuration(joints):
            branch = -1 if joints[0] < 0.0 else 1
            return SimpleNamespace(
                shoulder=branch, elbow=1, wrist=1, redundancy=0.0
            )

        @staticmethod
        def forward(_joints):
            return np.eye(4)

        def jacobian(self, _joints):
            self.jacobian_calls += 1
            if self.jacobian_calls > 1:
                raise RuntimeError("native Jacobian failure")
            return np.eye(6, 7)

        @staticmethod
        def solve(_target, _seed, _method):
            return [np.full(7, -0.1), np.full(7, 0.5)]

    solver = FailingSolver()
    tracker = SRSContinuousTracker(
        solver,
        np.zeros(7),
        SRSContinuousOptions(branch_hysteresis_frames=3),
    )

    with pytest.raises(RuntimeError, match="native Jacobian failure"):
        tracker.solve(np.eye(4), dt=1.0)

    assert np.array_equal(tracker.joints, np.zeros(7))
    assert np.array_equal(tracker.velocity, np.zeros(7))
    assert tracker.frame_index == 0
    assert tracker._pending_branch is None
    assert tracker._pending_count == 0
