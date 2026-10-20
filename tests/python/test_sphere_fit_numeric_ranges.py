"""Distance regressions with representable results outside squared range."""

import math
import tracemalloc

import numpy as np
import pytest
from holistic_motion.geometry import (
    SphereFitOptions,
    SphereSpec,
    evaluate_sphere_fit,
    fit_spheres,
)
from holistic_motion.geometry.sphere_fit import (
    _minimum_distances,
    _requires_stable_distances,
)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_minimum_distances_preserve_finite_norms(scale, chunk_size):
    query = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]) * scale
    reference = np.array([[0.0, 0.0, 2.0], [6.0, 8.0, 0.0]]) * scale
    expected = [2.0, 5.0]
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        actual = _minimum_distances(query, reference, chunk_size)
    np.testing.assert_allclose(actual / scale, expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("chunk_size", [1, 3, 32])
@pytest.mark.parametrize("sampled_coverage", [False, True])
def test_scaled_sphere_fit_preserves_radii_and_selection(
    scale, chunk_size, sampled_coverage
):
    interior = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * scale
    surface = np.array([[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = fit_spheres(
            interior,
            surface * scale,
            SphereFitOptions(
                max_spheres=2,
                min_radius=0.1 * scale,
                sampled_coverage=sampled_coverage,
                chunk_size=chunk_size,
            ),
        )
    np.testing.assert_array_equal(
        np.array([sphere.center for sphere in result.spheres]) / scale,
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    expected_radii = [1.0, 0.25] if sampled_coverage else [0.25, 0.25]
    np.testing.assert_allclose(
        np.array([sphere.radius for sphere in result.spheres]) / scale,
        expected_radii,
        rtol=2e-15,
        atol=0,
    )


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("chunk_size", [1, 2, 32])
def test_sphere_metrics_preserve_representable_gaps(scale, chunk_size):
    sphere = SphereSpec((0.0, 0.0, 0.0), scale)
    points = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]) * scale
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = evaluate_sphere_fit(points, [sphere], chunk_size=chunk_size)
    assert math.isclose(result.mean_uncovered_distance / scale, 2.5, rel_tol=2e-15)
    assert math.isclose(result.maximum_uncovered_distance / scale, 4.0, rel_tol=2e-15)


def test_sphere_metrics_mean_does_not_overflow_for_finite_gaps():
    points = [[1e308, 0.0, 0.0]] * 4
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = evaluate_sphere_fit(points, [SphereSpec((0.0, 0.0, 0.0), 1e307)])
    assert math.isclose(result.mean_uncovered_distance, 9e307, rel_tol=2e-15)
    assert result.mean_uncovered_distance == result.maximum_uncovered_distance


@pytest.mark.parametrize("axes", [1, 3])
def test_sphere_metrics_recover_finite_gap_after_distance_overflow(axes):
    coordinate = 1e308 if axes == 1 else 7.5e307
    center = ([-coordinate] * axes) + ([0.0] * (3 - axes))
    point = ([coordinate] * axes) + ([0.0] * (3 - axes))
    radius = 1.5e308
    expected = (math.sqrt(axes) * (coordinate / 1e308 * 2.0) - 1.5) * 1e308
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = evaluate_sphere_fit([point], [SphereSpec(tuple(center), radius)])
    assert math.isclose(result.mean_uncovered_distance, expected, rel_tol=3e-15)
    assert result.mean_uncovered_distance == result.maximum_uncovered_distance


def test_nearby_small_coordinates_retain_their_distance():
    coordinate = 1e-140
    neighbor = np.nextafter(coordinate, np.inf)
    delta = neighbor - coordinate
    actual = _minimum_distances(
        np.array([[coordinate, 0.0, 0.0]]),
        np.array([[neighbor, 0.0, 0.0]]),
        1,
    )
    assert actual[0] == delta


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_finite_nearest_distance_survives_unrepresentable_other_pairs(chunk_size):
    query = np.array([[1e308, 0.0, 0.0]])
    reference = np.array([[-1e308, 0.0, 0.0], [1e308, 3e200, 4e200]])
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = _minimum_distances(query, reference, chunk_size)
    assert math.isclose(result[0], 5e200, rel_tol=2e-15)


@pytest.mark.parametrize(
    "scale", [1e-310, 1e-200, 1e-155, 1e-140, 1e-130, 1.0, 1e150, 1e200, 1e307]
)
def test_distance_tiles_match_independent_hypot_reference(scale):
    rng = np.random.default_rng(1401)
    query = rng.uniform(-1.0, 1.0, (12, 3)) * scale
    reference = rng.uniform(-1.0, 1.0, (17, 3)) * scale
    expected = np.array([min(math.dist(q, r) for r in reference) for q in query])
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        actual = _minimum_distances(query, reference, 5)
    np.testing.assert_allclose(
        actual, expected, rtol=3e-14, atol=4 * np.nextafter(0.0, 1.0)
    )


def test_unrepresentable_final_gap_is_rejected_explicitly():
    with (
        np.errstate(over="raise", under="ignore", invalid="raise"),
        pytest.raises(ValueError, match="uncovered distances must be finite"),
    ):
        evaluate_sphere_fit(
            [[1e308, 1e308, 1e308]],
            [SphereSpec((-1e308, -1e308, -1e308), 1.0)],
        )


def test_exceptional_distance_workspace_is_bounded():
    query = np.zeros((64, 3))
    surface = np.ones((8192, 3)) * 1e200
    tracemalloc.start()
    try:
        actual = _minimum_distances(query, surface, chunk_size=64)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_allclose(actual / 1e200, math.sqrt(3), rtol=3e-15)
    assert peak < 1024 * 1024


def test_distance_range_classification_workspace_is_bounded():
    points = np.ones((200_000, 3))
    tracemalloc.start()
    try:
        assert not _requires_stable_distances(points)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024


@pytest.mark.parametrize("count", [3, 11, 21, 512])
@pytest.mark.parametrize("below", [False, True])
def test_finite_gap_mean_keeps_sum_rounding_headroom(count, below):
    value = np.finfo(float).max / count
    if below:
        value = np.nextafter(value, 0.0)
    points = np.zeros((count, 3))
    points[:, 0] = value
    with np.errstate(over="raise", invalid="raise"):
        result = evaluate_sphere_fit(
            points, [SphereSpec((0.0, 0.0, 0.0), 1.0)], chunk_size=count
        )
    assert math.isclose(result.mean_uncovered_distance, value, rel_tol=2e-15)
    assert result.maximum_uncovered_distance == value
