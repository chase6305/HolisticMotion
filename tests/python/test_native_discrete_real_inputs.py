"""Discrete motion parameters must not truncate NumPy complex values."""

import holistic_motion as hm
import numpy as np
import pytest
from test_planning_real_inputs import collision_models as _collision_models

collision_models = _collision_models


@pytest.mark.parametrize(
    "options_type,field",
    [
        (hm.PlanningOptions, field)
        for field in (
            "max_iterations",
            "random_seed",
            "shortcut_attempts",
            "interpolation_points",
            "simplify_path",
            "interpolate_path",
        )
    ]
    + [
        (hm.PathOptimizationOptions, field)
        for field in ("max_iterations", "line_search_steps")
    ],
)
@pytest.mark.parametrize(
    "bad", [np.complex128(2 + 1j), np.complex128(2 + 0j), np.array(2 + 1j)]
)
def test_discrete_planning_option_rejects_complex_without_mutation(
    options_type, field, bad
):
    options = options_type()
    previous = getattr(options, field)
    with pytest.raises(TypeError, match="real"):
        setattr(options, field, bad)
    assert getattr(options, field) == previous


@pytest.fixture(params=["joint", "cartesian"])
def trajectory(request):
    if request.param == "joint":
        return hm.RnTrajectory([[0.0], [0.2]], [1.0], [1.0], [1.0])
    end = np.eye(4)
    end[0, 3] = 0.2
    return hm.CartesianLineTrajectory(np.eye(4), end, [1] * 6, [1] * 6, [1] * 6)


@pytest.mark.parametrize("method", ["sample_uniform", "constraint_report"])
@pytest.mark.parametrize("bad", [np.complex128(7 + 1j), np.array(7 + 1j)])
def test_native_sampling_rejects_complex_counts(trajectory, method, bad):
    with pytest.raises(TypeError, match="real"):
        getattr(trajectory, method)(bad)


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
def test_continuous_joint_indices_reject_complex_values(factory):
    planner = factory([-4.0], [4.0])
    with pytest.raises(TypeError, match="real"):
        planner.set_continuous_joints(np.array([0 + 1j]))


def test_collision_planner_active_indices_reject_complex_values(collision_models):
    _, model = collision_models
    with pytest.raises(TypeError, match="real"):
        hm.SamplingPlanner.from_collision_model(
            [-1.0],
            [1.0],
            model,
            active_indices=np.array([0 + 1j]),
            context_configuration=[0.0],
        )


def test_null_space_steps_reject_complex_value(tmp_path):
    from test_srs_kinematics import _write_ideal_srs_urdf

    urdf = tmp_path / "srs.urdf"
    _write_ideal_srs_urdf(urdf)
    robot = hm.Robot(str(urdf))
    planner = hm.NullSpacePlanner(robot.kinematics)
    with pytest.raises(TypeError, match="real"):
        planner.plan(np.zeros(7), np.ones(7), np.complex128(2 + 1j))


def test_discrete_planning_options_still_accept_real_values():
    options = hm.PlanningOptions()
    options.max_iterations = np.int64(7)
    options.simplify_path = False
    assert options.max_iterations == 7
    assert not options.simplify_path


def test_native_sampling_still_accepts_numpy_integer_counts(trajectory):
    samples = trajectory.sample_uniform(np.int64(7))
    assert len(samples[0]) == 7
    assert trajectory.constraint_report(np.int64(7))["within_limits"]


@pytest.mark.parametrize("factory", [hm.SamplingPlanner, hm.PathOptimizer])
def test_continuous_joint_indices_still_accept_numpy_integers(factory):
    planner = factory([-4.0], [4.0])
    planner.set_continuous_joints(np.array([0], dtype=np.int64))
    if factory is hm.PathOptimizer:
        result = planner.optimize([[0.0], [0.1]])
    else:
        result = planner.plan([0.0], [0.1])
    assert result.success
