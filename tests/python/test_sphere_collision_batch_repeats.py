"""Repeated batch states may be reused without erasing tiny distinct moves."""

import holistic_motion as hm
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(hm, "SphereCollisionModel"),
    reason="native collision support is disabled",
)


@pytest.mark.parametrize("scale", [1e-200, 1e-160, 1.0, 1e150])
@pytest.mark.parametrize("strided", [False, True])
def test_batch_keeps_distinct_states_between_repeated_runs(tmp_path, scale, strided):
    path = tmp_path / "repeated_slider.urdf"
    path.write_text(
        '<robot name="repeated"><link name="base"/><link name="moving"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="moving"/><axis xyz="0 1 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("fixed", "base", [0, 0, 0], 0.25 * scale),
            hm.CollisionSphere("moving", "moving", [0, 0, 0], 0.25 * scale),
        ],
    )
    values = scale * np.array([0.25, 0.25, 0.75, 0.75, 0.25, 0.25])
    configurations = np.empty((12, 1))
    configurations[::2, 0] = values
    configurations[1::2, 0] = np.nan
    batch = configurations[::2] if strided else values[:, None].copy()
    expected = np.array([model.minimum_distance(q).distance for q in batch])
    actual = model.batch_minimum_distances(batch)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual / scale, [-0.25, -0.25, 0.25, 0.25, -0.25, -0.25])
    # Reuse is confined to this call; the next batch starts from its own state.
    np.testing.assert_allclose(
        model.batch_minimum_distances([[scale], [scale]]) / scale, [0.5, 0.5]
    )


def test_repeated_empty_configurations_for_a_fixed_model(tmp_path):
    path = tmp_path / "fixed.urdf"
    path.write_text('<robot name="fixed"><link name="base"/></robot>')
    model = hm.SphereCollisionModel(
        str(path),
        [
            hm.CollisionSphere("first", "base", [0, 0, 0], 0.1),
            hm.CollisionSphere("second", "base", [1, 0, 0], 0.1),
        ],
        exclude_same_link=False,
    )
    np.testing.assert_allclose(model.batch_minimum_distances(np.empty((16, 0))), 0.8)
