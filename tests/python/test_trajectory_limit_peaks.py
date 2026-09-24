"""Short curved phases need enough local checks to resolve derivative peaks."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("profile", ["trapezoidal", "double_s"])
def test_short_blended_phase_respects_acceleration_between_old_grid_points(profile):
    points = [
        [-1.1240352690494126, -0.5276353872461453],
        [-0.4839164391297848, -2.214423149787105],
        [-2.8017414120155553, 2.5626250209694725],
        [-1.406785454270201, 0.34768012600664555],
        [0.24251209584936406, 0.16119065976187308],
        [0.4250680397439814, -3.0079036053927313],
        [-0.797599567730627, -0.6435492442134808],
        [1.034572648566476, -1.480262741109277],
    ]
    velocity = [6.667222260859432, 0.12647997540335185]
    acceleration = [0.8489751465430547, 0.30843764050491224]
    jerk = [7.92619368409633, 8.398976330132875]
    trajectory = hm.RnTrajectory(
        points,
        velocity,
        acceleration,
        jerk,
        blend_tolerance=0.045461028415368504,
        profile=profile,
    )
    # Dense sampling independently resolves peaks inside each time phase,
    # including short phases which a global uniform grid could skip entirely.
    times = np.unique(
        np.concatenate(
            [
                np.linspace(start, end, 1001)
                for start, end in zip(
                    trajectory.breakpoints[:-1], trajectory.breakpoints[1:]
                )
            ]
        )
    )
    q, dq, ddq, dddq = trajectory.sample(times)
    np.testing.assert_allclose(q[[0, -1]], np.asarray(points)[[0, -1]], atol=1e-12)
    assert np.max(np.abs(dq) / velocity) <= 1.0 + 1e-8
    assert np.max(np.abs(ddq) / acceleration) <= 1.0 + 1e-8
    if profile == "double_s":
        assert np.max(np.abs(dddq) / jerk) <= 1.0 + 1e-8
