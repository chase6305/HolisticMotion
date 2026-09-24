"""Trapezoidal joins retain endpoint speeds when adjacent caps differ by roundoff."""

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "waypoints": [
                [-2.7689420092621817, 3.7985050687871023],
                [-4.385726297879007, 11.949321755874948],
                [-1.194641717906189, 11.865072510579743],
                [1.5637246731312702, 18.298048023972413],
                [8.054183172658297, 18.16043107310211],
                [10.80060100428091, 17.30190140310391],
                [12.3821806787286, 16.877329166790233],
            ],
            "max_velocity": [8.994068077425604, 0.20592010234334918],
            "max_acceleration": [0.11125296225923398, 0.1690329271022458],
            "max_jerk": [1.5885565120353546, 2.210330481020611],
            "blend_tolerance": 0.11063441808433969,
            "profile": "trapezoidal",
        },
        {
            "waypoints": [
                [-5.688086840868596, 2.6281865093719, 2.1473141602569545],
                [-18.11728561499656, 20.91667119813893, 4.664165355041616],
                [-19.453122354064067, 16.476972682375727, 6.57524336635457],
                [-19.11142882781169, 17.436682773111496, 7.229297291370769],
                [-19.94900641216744, 12.89745430887188, 13.149541741388362],
                [-21.510801575943017, 18.707424606730214, 12.950302335142228],
                [-24.066699847641363, 22.15063383649094, 15.65445703534081],
            ],
            "max_velocity": [
                0.9088368083720556,
                0.14318784087604167,
                1.0942793315547796,
            ],
            "max_acceleration": [
                6.702231777272789,
                0.10816561409376295,
                3.901700620691342,
            ],
            "max_jerk": [0.35807531491060235, 0.27210706292063114, 9.565889513824901],
            "blend_tolerance": 0.22107965192244378,
            "profile": "trapezoidal",
        },
        {
            "waypoints": [
                [8.542670992438346, -0.34356567418993056],
                [9.515906583082698, -12.35850204863413],
                [10.653842330340137, -13.810005769799112],
                [16.44504171086057, -15.13315116810315],
                [28.513357700518082, -11.396527991979749],
                [25.708206184661183, -17.794176415610313],
                [30.441373975068995, -15.547460776105087],
                [32.22819298658503, -12.507822583711475],
                [23.163217952502656, -14.811355329336934],
            ],
            "max_velocity": [4.74708062678588, 0.12281592983499312],
            "max_acceleration": [3.169812292529015, 6.450257239195547],
            "max_jerk": [0.5704173450402453, 0.20388315654691913],
            "blend_tolerance": 0.15188088985047143,
            "profile": "trapezoidal",
        },
    ],
)
def test_trapezoidal_roundoff_caps_preserve_joins(inputs):
    trajectory = hm.RnTrajectory(**inputs)
    times = np.unique(
        np.concatenate(
            [
                np.linspace(start, end, 201)
                for start, end in zip(
                    trajectory.breakpoints[:-1], trajectory.breakpoints[1:]
                )
            ]
        )
    )
    q, dq, ddq, dddq = trajectory.sample(times)
    assert all(np.isfinite(state).all() for state in (q, dq, ddq, dddq))
    np.testing.assert_allclose(
        q[[0, -1]], np.asarray(inputs["waypoints"])[[0, -1]], rtol=0.0, atol=1e-10
    )
    for values, limits in [
        (dq, inputs["max_velocity"]),
        (ddq, inputs["max_acceleration"]),
    ]:
        assert np.max(np.abs(values) / limits) <= 1.0 + 1e-8
    report = trajectory.constraint_report(10001)
    assert report["within_limits"]
    assert report["velocity_continuous"]
