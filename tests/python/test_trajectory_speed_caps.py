"""Nearly equal segment speed caps must not trigger unstable backtracking."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "waypoints": [
                [3.516000378628744, -1.713593990643606],
                [5.008518619569767, -1.5381889661973571],
                [5.218936760265947, -2.976082054179771],
                [6.737778499584194, 1.129276432272222],
            ],
            "max_velocity": [1.0806449766373605, 0.198295816316942],
            "max_acceleration": [3.858567497728526, 0.30170178980682866],
            "max_jerk": [1.4650281214212937, 8.418921296737315],
            "blend_tolerance": 0.06423815323887772,
            "profile": "double_s",
        },
        {
            "waypoints": [
                [-20.691711766606982, 7.045042417250765],
                [-27.471730668987497, 9.356168056278548],
                [-25.238764202394027, 13.18415548126543],
                [-40.083575269361134, 18.438270891001178],
                [-48.55991549602662, 25.235157057330493],
                [-37.211743373919404, 25.44558241130081],
                [-27.417933971154785, 18.837176119882617],
            ],
            "max_velocity": [0.14938345815413615, 0.3805095598198231],
            "max_acceleration": [1.7995224686050153, 0.2549148743546786],
            "max_jerk": [7.8319880263708, 1.081973119839531],
            "blend_tolerance": 0.1969388518447756,
            "profile": "double_s",
        },
        {
            "waypoints": [
                [4.77662541496942, 3.8864826489034536, -5.3288695092381735],
                [8.577832856524642, 3.912940377107717, -8.136627989316693],
                [12.622528854522791, 8.804085005442195, -6.6559937063934616],
                [7.296948478084885, 7.420067349003566, -3.6641402355453283],
            ],
            "max_velocity": [
                9.036893257974883,
                0.11557255353479005,
                1.6062985867630226,
            ],
            "max_acceleration": [
                0.5438427054643121,
                3.453534095864047,
                0.6169520487994151,
            ],
            "max_jerk": [0.4432663579464532, 2.834703586758305, 0.5181063168152041],
            "blend_tolerance": 0.12562747776910346,
            "profile": "double_s",
        },
    ],
)
def test_double_s_preserves_roundoff_equivalent_speed_caps(inputs):
    trajectory = hm.RnTrajectory(**inputs)
    times = np.unique(
        np.concatenate(
            [
                np.linspace(start, end, 301)
                for start, end in zip(
                    trajectory.breakpoints[:-1], trajectory.breakpoints[1:]
                )
            ]
        )
    )
    q, dq, ddq, dddq = trajectory.sample(times)
    np.testing.assert_allclose(
        q[[0, -1]], np.asarray(inputs["waypoints"])[[0, -1]], rtol=0.0, atol=1e-11
    )
    for value, name in [
        (dq, "max_velocity"),
        (ddq, "max_acceleration"),
        (dddq, "max_jerk"),
    ]:
        assert np.max(np.abs(value) / inputs[name]) <= 1.0 + 1e-8
    assert trajectory.constraint_report(10001)["within_limits"]
