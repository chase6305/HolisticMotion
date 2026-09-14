"""Scale and translation checks for support-region geometry."""

import numpy as np
import pytest
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    RetargetingMode,
    RetargetingModeManager,
    SupportPolygonTask,
)


@pytest.mark.parametrize("scale", [1e-160, 1e-9, 1.0, 1e160])
@pytest.mark.parametrize("clockwise", [False, True])
def test_support_polygon_preserves_unit_normals_across_scales(scale, clockwise):
    vertices = scale * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    if clockwise:
        vertices = vertices[::-1]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        task = SupportPolygonTask(vertices)
    np.testing.assert_allclose(np.linalg.norm(task.normals, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(task.offsets / scale, -1.0, atol=1e-15)
    np.testing.assert_array_equal(
        task.vertices, vertices[::-1] if clockwise else vertices
    )
    assert np.isfinite(task.normals).all() and np.isfinite(task.offsets).all()


@pytest.mark.parametrize("scale", [1e-160, 1e-9, 1.0, 1e160])
def test_crossed_support_polygon_rejected_at_any_scale(scale):
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    vertices = scale * np.column_stack((np.cos(angles), np.sin(angles)))
    with (
        np.errstate(over="raise", invalid="raise", divide="raise"),
        pytest.raises(ValueError),
    ):
        SupportPolygonTask(vertices[[0, 2, 4, 1, 3]])


def test_unrepresentable_polygon_span_rejected_without_floating_error():
    largest = np.finfo(float).max
    with (
        np.errstate(over="raise", invalid="raise", divide="raise"),
        pytest.raises(ValueError, match="finite"),
    ):
        SupportPolygonTask([[-largest, 0], [largest, 0], [0, largest]])


def test_subnormal_edges_still_have_unit_normals():
    vertices = np.nextafter(0.0, 1.0) * np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    task = SupportPolygonTask(vertices)
    np.testing.assert_allclose(np.linalg.norm(task.normals, axis=1), 1.0, atol=1e-15)


@pytest.mark.parametrize("translation", [[0.0, 0.0], [1e9, -1e9]])
def test_support_vertex_remains_on_boundary_in_solver(tmp_path, translation):
    pytest.importorskip("pinocchio", exc_type=ImportError)
    angle = 0.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    vertices = (
        np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]) @ rotation.T
        + translation
    )
    x, y = vertices[0]
    urdf = tmp_path / "mass_at_boundary.urdf"
    urdf.write_text(f'''<robot name="mass_at_boundary">
      <link name="base"/>
      <link name="body"><inertial><mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="body"/>
        <origin xyz="{x} {y} 1"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/>
      </joint></robot>''')
    solver = PinkRetargetingSolver(
        urdf,
        {"left_hand": "body"},
        {"left_arm": []},
        mode_manager=RetargetingModeManager(initial_mode=RetargetingMode.LEFT_ARM),
        frame_tasks={"left_hand": FrameTask(position_cost=0, orientation_cost=0)},
        posture_task=PostureTask(cost=0),
        support_polygon_task=SupportPolygonTask(vertices),
        support_polygon_tolerance=1e-12,
    )
    result = solver.solve({"left_hand": np.eye(4)}, seed=[0.0])
    assert result.success
    assert result.support_polygon_violation <= 1e-16
