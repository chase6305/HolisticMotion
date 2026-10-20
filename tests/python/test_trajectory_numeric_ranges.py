"""Representable line lengths must survive intermediate squared norms."""

import math

import holistic_motion as hm
import numpy as np
import pytest


@pytest.mark.parametrize("dof", [1, 2, 7])
@pytest.mark.parametrize("scale", [1e-200, 1e-170, 1e200])
def test_trapezoidal_line_retains_representable_path_length(dof, scale):
    points = np.zeros((2, dof))
    points[1, 0] = scale
    trajectory = hm.RnTrajectory(
        points,
        np.full(dof, math.sqrt(scale)),
        np.ones(dof),
        np.ones(dof),
        profile="trapezoidal",
    )
    assert trajectory.path_length == pytest.approx(scale, rel=1e-13, abs=0.0)
    assert np.isfinite(trajectory.duration) and trajectory.duration > 0.0
    samples = trajectory.sample(np.linspace(0.0, trajectory.duration, 5))
    assert all(np.isfinite(values).all() for values in samples)
    np.testing.assert_allclose(samples[0][-1], points[-1], rtol=1e-13, atol=0.0)
