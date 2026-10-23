"""Planning-space handoffs preserve named coordinates and physical TCP poses."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import holistic_motion as hm
import numpy as np
import pytest

pytest.importorskip("hypothesis", reason="install the test extra for property tests")
from hypothesis import assume, given
from hypothesis import strategies as st

pytestmark = pytest.mark.property


@st.composite
def mixed_spaces(draw):
    count = draw(st.integers(1, 6))
    periodic = draw(st.lists(st.booleans(), min_size=count, max_size=count))
    centers = np.array(
        draw(st.lists(st.integers(-3, 3), min_size=count, max_size=count)), dtype=float
    )
    radii = (
        np.array(
            draw(st.lists(st.integers(1, 16), min_size=count, max_size=count)),
            dtype=float,
        )
        / 4
    )
    for index, continuous in enumerate(periodic):
        if continuous:
            centers[index] *= 2 * np.pi
            radii[index] = np.pi
    ticks = draw(
        st.lists(
            st.tuples(st.integers(-32, 32), st.integers(-32, 32)),
            min_size=count,
            max_size=count,
        )
    )
    start = centers + radii * np.array([pair[0] for pair in ticks]) / 32
    goal = centers + radii * np.array([pair[1] for pair in ticks]) / 32
    assume(
        any(
            abs(math.remainder(b - a, 2 * np.pi) if circle else b - a) > 1e-6
            for a, b, circle in zip(start, goal, periodic)
        )
    )
    weights = np.power(
        10.0, draw(st.lists(st.integers(-8, 8), min_size=count, max_size=count))
    )
    order = list(draw(st.permutations(range(count))))
    points = draw(st.integers(3, 17))
    return (
        periodic,
        centers - radii,
        centers + radii,
        start,
        goal,
        weights,
        order,
        points,
    )


@given(mixed_spaces())
def test_named_periodic_plan_and_scalar_trajectory_coordinates_agree(case):
    periodic, lower, upper, start, goal, weights, order, points = case
    count = len(periodic)
    names = [f"joint{i}" for i in order]
    space = hm.JointSpace(lower[order], upper[order], names)
    space = space.with_joint_weights(weights[order]).with_continuous_joints(
        [index for index, original in enumerate(order) if periodic[original]]
    )
    planner = hm.SamplingPlanner(space)
    options = hm.PlanningOptions()
    options.interpolate_path = True
    options.interpolation_points = points
    planned = planner.plan(start[order], goal[order], options)
    assert planned.success and len(planned.path) >= 2
    lifted = np.asarray(planned.joint_space.unwrap_path(planned.path))
    assert planned.joint_space.joint_names == names
    np.testing.assert_array_equal(
        lifted[:, [i for i, j in enumerate(order) if not periodic[j]]],
        np.asarray(planned.path)[
            :, [i for i, j in enumerate(order) if not periodic[j]]
        ],
    )
    # The Euclidean lifted path must retain the search metric through the
    # circular cut, including shifted angular intervals and nonuniform weights.
    lifted_length = sum(
        math.hypot(*(np.sqrt(weights[order]) * delta))
        for delta in np.diff(lifted, axis=0)
    )
    assert lifted_length == pytest.approx(
        planned.statistics.final_path_length, rel=1e-10, abs=1e-12
    )
    optimized = hm.PathOptimizer(planned.joint_space).optimize(planned.path)
    assert optimized.success
    assert optimized.statistics.initial_path_length == pytest.approx(
        lifted_length, rel=1e-10, abs=1e-12
    )

    links = "".join(f'<link name="link{i}"/>' for i in range(count + 1))
    joints = "".join(
        f'<joint name="joint{i}" type="{"continuous" if periodic[i] else "prismatic"}">'
        f'<parent link="link{i}"/><child link="link{i + 1}"/>'
        f'<origin xyz="0.13 0.03 -0.02" rpy="0.02 -0.04 0.07"/>'
        f'<axis xyz="{("1 0 0", "0 1 0", "0 0 1")[i % 3]}"/>'
        f'<limit lower="{lower[i]:.17g}" upper="{upper[i]:.17g}" velocity="1" effort="1"/></joint>'
        for i in range(count)
    )
    with TemporaryDirectory(prefix="hm_joint_space_") as folder:
        path = Path(folder) / "chain.urdf"
        path.write_text('<robot name="handoff">' + links + joints + "</robot>")
        robot = hm.Robot(str(path))
        mapping = robot.create_joint_position_map(names)
        context = np.zeros(robot.model_dof)
        for source, unwrapped in zip(planned.path, lifted):
            representative = mapping.configuration_with_joint_positions(context, source)
            trajectory_coordinate = mapping.configuration_with_joint_positions(
                context, unwrapped
            )
            np.testing.assert_allclose(
                robot.kinematics.forward(representative),
                robot.kinematics.forward(trajectory_coordinate),
                atol=1e-10,
                rtol=1e-12,
            )
    # A subsequent planner reconfiguration cannot change either retained path's
    # names, weights or periodic topology.
    planner.set_joint_weights(np.ones(count))
    planner.set_continuous_joints([])
    for result in (planned, optimized):
        assert result.joint_space.joint_names == names
        np.testing.assert_array_equal(result.joint_space.joint_weights, weights[order])
        assert (
            result.joint_space.continuous_joint_indices
            == space.continuous_joint_indices
        )
