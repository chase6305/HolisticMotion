"""Generated minimum selection checks against full pair enumeration."""

from pathlib import Path
from tempfile import TemporaryDirectory

import holistic_motion as hm
import numpy as np
import pytest
from test_sphere_minimum_selection import assert_enumerated_minimum, model_with_pairs

pytest.importorskip("hypothesis", reason="install the test extra for property checks")
from hypothesis import given
from hypothesis import strategies as st

pytestmark = [
    pytest.mark.property,
    pytest.mark.skipif(
        not hasattr(hm, "SphereCollisionModel"),
        reason="native collision support is disabled",
    ),
]


@st.composite
def sphere_clouds(draw):
    count = 2 * draw(st.integers(2, 8))
    scale = draw(st.sampled_from([1e-200, 1e-160, 1e-14, 1.0, 1e150, 1e200]))
    coordinates = draw(
        st.lists(
            st.tuples(*(st.integers(-16, 16) for _ in range(3))),
            min_size=count,
            max_size=count,
        )
    )
    radii = draw(st.lists(st.integers(1, 16), min_size=count, max_size=count))
    return np.asarray(coordinates) * (scale / 8), np.asarray(radii) * (scale / 16)


@given(sphere_clouds())
def test_minimum_agrees_with_all_pairs_at_multiple_scales(cloud):
    centers, radii = cloud
    with TemporaryDirectory(prefix="hm-sphere-minimum-") as directory:
        model = model_with_pairs(Path(directory), radii, centers)
        assert_enumerated_minimum(model, [])
        model.reset_collision_pairs()
        assert_enumerated_minimum(model, [])
