"""Block pruning preserves stable greedy selection across distance tiles."""

import tracemalloc

import numpy as np
import pytest
from holistic_motion.geometry import SphereFitOptions, fit_spheres


@pytest.mark.parametrize("chunk_size", [32, 65, 512])
@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("sampled_coverage", [False, True])
def test_duplicate_blocks_preserve_more_than_one_tile_of_selected_centers(
    chunk_size, scale, sampled_coverage
):
    centers = np.zeros((67, 3))
    centers[:, 0] = np.arange(len(centers)) * 4.0
    surface = centers + [0.0, 1.0, 0.0]
    # The last group and blocks are deliberately incomplete; a new accepted
    # sphere inside a block must still suppress subsequent surviving copies.
    interior = np.repeat(centers, 97, axis=0)[:-13]
    result = fit_spheres(
        interior * scale,
        surface * scale,
        SphereFitOptions(
            max_spheres=80,
            min_radius=0.1 * scale,
            sampled_coverage=sampled_coverage,
            chunk_size=chunk_size,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray([sphere.center for sphere in result.spheres]), centers * scale
    )
    np.testing.assert_array_equal(
        [sphere.radius for sphere in result.spheres], np.full(len(centers), scale)
    )
    assert result.metrics.sphere_count == len(centers)
    assert result.metrics.sampled_coverage == 1.0


@pytest.mark.parametrize("chunk_size", [1, 3, 16, 32, 127, 512])
@pytest.mark.parametrize("max_spheres", [1, 8, 64])
def test_pruned_candidates_follow_the_original_containment_order(
    chunk_size, max_spheres
):
    generator = np.random.default_rng(1826)
    centers = generator.uniform(-0.6, 0.6, (17, 3))
    interior = np.repeat(centers, 43, axis=0)
    surface = generator.normal(size=(19, 3))
    surface /= np.linalg.norm(surface, axis=1)[:, None]
    radii = np.min(np.linalg.norm(interior[:, None] - surface[None, :], axis=2), axis=1)
    selected = []
    for index in np.argsort(-radii, kind="stable"):
        if any(
            np.linalg.norm(interior[index] - interior[previous])
            <= max(radii[previous] - radii[index], 0.0)
            for previous in selected
        ):
            continue
        selected.append(index)
        if len(selected) == max_spheres:
            break
    result = fit_spheres(
        interior,
        surface,
        SphereFitOptions(
            max_spheres=max_spheres, min_radius=1e-6, chunk_size=chunk_size
        ),
    )
    np.testing.assert_array_equal(
        [sphere.center for sphere in result.spheres], interior[selected]
    )
    np.testing.assert_allclose(
        [sphere.radius for sphere in result.spheres], radii[selected], rtol=1e-14
    )


def test_duplicate_pruning_keeps_pairwise_workspace_bounded():
    generator = np.random.default_rng(281)
    interior = np.repeat(generator.uniform(-0.6, 0.6, (32, 3)), 640, axis=0)
    surface = generator.normal(size=(8, 3))
    surface /= np.linalg.norm(surface, axis=1)[:, None]
    tracemalloc.start()
    try:
        result = fit_spheres(
            interior, surface, SphereFitOptions(max_spheres=128, chunk_size=64)
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.metrics.sphere_count <= 32
    # A single all-candidate by accepted-sphere norm matrix exceeds this
    # budget; query/reference blocks and the linear candidate arrays fit.
    assert peak < 3 * 1024 * 1024
