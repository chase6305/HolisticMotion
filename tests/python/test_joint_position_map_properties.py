"""Cross-model Cartesian derivatives must retain named joint units and order."""

from pathlib import Path
from tempfile import TemporaryDirectory

import holistic_motion as hm
import numpy as np
import pytest
from holistic_motion.configuration import JointPositionMap

pytest.importorskip("hypothesis", reason="install the test extra for property tests")
from hypothesis import given
from hypothesis import strategies as st

pytestmark = pytest.mark.property


@st.composite
def mapped_chains(draw):
    count = draw(st.integers(2, 7))
    kinds = draw(
        st.lists(
            st.sampled_from(["continuous", "revolute", "prismatic"]),
            min_size=count,
            max_size=count,
        )
    )
    kinds[0], kinds[-1] = "continuous", "prismatic"
    axes = draw(
        st.lists(
            st.tuples(*(st.integers(-3, 3) for _ in range(3))).filter(any),
            min_size=count,
            max_size=count,
        )
    )
    scales = draw(
        st.lists(
            st.sampled_from([0.5, 1.5, 2.0, 1e-200, 1e200]),
            min_size=count,
            max_size=count,
        )
    )
    positions = draw(st.lists(st.integers(-8, 8), min_size=count, max_size=count))
    order = draw(st.permutations(range(count)))
    yaw = draw(st.integers(-8, 8)) / 8.0
    fixed_frames = draw(st.booleans())
    fixed_tip = draw(st.booleans())
    return (
        kinds,
        axes,
        scales,
        np.asarray(positions) / 10.0,
        order,
        yaw,
        fixed_frames,
        fixed_tip,
    )


@given(mapped_chains())
def test_chain_point_jacobian_matches_full_model_distance_gradient(case):
    if not hasattr(hm, "SphereCollisionModel"):
        pytest.skip("requires native collision support")
    kinds, axes, scales, positions, order, yaw, fixed_frames, fixed_tip = case
    links = ['<link name="world"/>', '<link name="base"/>', '<link name="other"/>']
    joints = [
        '<joint name="floating_base" type="floating"><parent link="world"/>'
        '<child link="base"/></joint>',
        '<joint name="a_other" type="prismatic"><parent link="base"/>'
        '<child link="other"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint>',
    ]
    names = []
    for index, (kind, axis, scale) in enumerate(zip(kinds, axes, scales)):
        names.append(f"z_joint_{index}")
        child = f"link_{index}"
        parent = "base" if index == 0 else f"link_{index - 1}"
        if fixed_frames:
            spacer = f"spacer_{index}"
            links.append(f'<link name="{spacer}"/>')
            joints.append(
                f'<joint name="fixed_{index}" type="fixed"><parent link="{parent}"/>'
                f'<child link="{spacer}"/><origin xyz="-0.02 0.04 0.03" '
                'rpy="-0.05 0.09 -0.01"/></joint>'
            )
            parent = spacer
        links.append(f'<link name="{child}"/>')
        axis_text = " ".join(str(scale * value) for value in axis)
        joints.append(
            f'<joint name="{names[-1]}" type="{kind}">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<origin xyz="0.13 0.03 -0.01" rpy="0.07 -0.03 0.02"/>'
            f'<axis xyz="{axis_text}"/>'
            '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        )
    if fixed_tip:
        links.append('<link name="fixed_tip"/>')
        joints.append(
            f'<joint name="terminal" type="fixed"><parent link="{child}"/>'
            '<child link="fixed_tip"/><origin xyz="0.07 -0.02 0.03" '
            'rpy="0.04 0.03 -0.02"/></joint>'
        )
        child = "fixed_tip"
    offset = np.array([0.13, -0.07, 0.02])
    obstacle = np.array([3.0, -2.0, 1.0])
    with TemporaryDirectory(prefix="holistic-motion-mapped-chain-") as folder:
        path = Path(folder) / "chain.urdf"
        path.write_text('<robot name="mapped">' + "".join(links + joints) + "</robot>")
        robot = hm.Robot(str(path))
        chain = robot.create_kinematics("base", child)
        model = hm.SphereCollisionModel(
            str(path),
            [
                hm.CollisionSphere("tool", child, offset, 0.1),
                hm.CollisionSphere("obstacle", "world", obstacle, 0.2),
            ],
        )
        reference = model.configuration_from_joint_positions({"a_other": 0.4})
        floating = next(j for j in model.joint_coordinates if j.name == "floating_base")
        reference[floating.idx_q : floating.idx_q + 3] = [0.3, -0.4, 0.2]
        reference[floating.idx_q + 5 : floating.idx_q + 7] = [
            np.sin(yaw / 2),
            np.cos(yaw / 2),
        ]
        original = reference.copy()
        selected_names = [names[i] for i in order]
        mapping = model.create_joint_position_map(selected_names)
        pure_mapping = JointPositionMap.from_model(model, selected_names)
        full = mapping.configuration_with_joint_positions(
            reference, positions[list(order)]
        )
        np.testing.assert_allclose(
            full,
            pure_mapping.configuration_with_joint_positions(
                reference, positions[list(order)]
            ),
            atol=1e-15,
        )
        np.testing.assert_array_equal(reference, original)
        world_T_base = np.eye(4)
        world_T_base[:3, :3] = [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
        world_T_base[:3, 3] = [0.3, -0.4, 0.2]
        world_T_tool = world_T_base @ chain.forward(positions)
        lever = world_T_tool[:3, :3] @ offset
        center = world_T_tool[:3, 3] + lever
        np.testing.assert_allclose(model.world_spheres(full)[0, :3], center, atol=2e-12)

        jacobian = chain.jacobian(positions)
        linear = world_T_base[:3, :3] @ jacobian[:3]
        angular = world_T_base[:3, :3] @ jacobian[3:]
        point_jacobian = linear + np.cross(angular.T, lever).T
        separation = center - obstacle
        expected = (separation / np.linalg.norm(separation)) @ point_jacobian
        query = model.minimum_distance_with_gradient(full)
        assert query.distance_result.distance == pytest.approx(
            np.linalg.norm(separation) - 0.3, abs=2e-12
        )
        np.testing.assert_allclose(
            mapping.joint_velocities(query.gradient), expected[list(order)], atol=2e-12
        )
