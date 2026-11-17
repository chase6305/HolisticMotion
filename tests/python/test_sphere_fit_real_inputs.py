"""Real geometry, input ownership, and bounded sphere-fit conversion."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from holistic_motion.geometry import (
    SphereFitMetrics,
    SphereFitOptions,
    SphereSpec,
    evaluate_sphere_fit,
    fit_spheres,
    fit_trimesh,
)
from optional_dependencies import tracemalloc_or_skip


@pytest.mark.parametrize("imaginary", [0.0, 1.0])
@pytest.mark.parametrize("representation", ["array", "object", "list"])
@pytest.mark.parametrize("entry", ["center", "interior", "surface", "evaluation"])
def test_sphere_geometry_rejects_complex_coordinates(entry, representation, imaginary):
    value = np.complex128(0.1 + imaginary * 1j)
    if representation == "array":
        points = np.array([[value, 0.0, 0.0]], dtype=complex)
    elif representation == "object":
        points = np.zeros((1, 3), dtype=object)
        points[0, 0] = value
    else:
        points = [[value, 0.0, 0.0]]
    options = SphereFitOptions(min_radius=0.01)
    with pytest.raises(TypeError):
        if entry == "center":
            SphereSpec(points[0], 0.1)
        elif entry == "interior":
            fit_spheres(points, [[1.0, 0.0, 0.0]], options)
        elif entry == "surface":
            fit_spheres([[0.0, 0.0, 0.0]], points, options)
        else:
            evaluate_sphere_fit(points, [SphereSpec((0.0, 0.0, 0.0), 1.0)])


@pytest.mark.parametrize(
    "value", [np.complex64(0.1 + 1j), np.complex128(0.1), np.array(0.1 + 1j)]
)
@pytest.mark.parametrize(
    "entry", ["radius", "min_radius", "padding", "coverage", "mean", "maximum"]
)
def test_sphere_fit_rejects_complex_scalars(entry, value):
    with pytest.raises(TypeError):
        if entry == "radius":
            SphereSpec((0.0, 0.0, 0.0), value)
        elif entry in ("min_radius", "padding"):
            SphereFitOptions(**{entry: value})
        else:
            values = {"coverage": 0.5, "mean": 0.1, "maximum": 0.2}
            values[entry] = value
            SphereFitMetrics(values["coverage"], values["mean"], values["maximum"], 1)


def test_sphere_geometry_keeps_real_object_arrays_and_strided_samples():
    interior = np.zeros((4, 3), dtype=object)[::2]
    surface = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])[::2]
    result = fit_spheres(
        interior, surface, SphereFitOptions(min_radius=np.float64(0.1))
    )
    assert result.spheres == (SphereSpec((0.0, 0.0, 0.0), np.float64(1.0)),)
    assert result.metrics.sampled_coverage == 1.0


@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_sphere_center_snapshot_precedes_radius_conversion(layout):
    storage = np.arange(6.0)
    center = storage[:3] if layout == "contiguous" else storage[::2]
    if layout == "readonly":
        center.setflags(write=False)
    expected = tuple(center)

    class MutatingRadius(float):
        def __float__(self):
            storage[:] = np.nan
            return 0.1

    sphere = SphereSpec(center, MutatingRadius(0.1))
    assert np.isnan(storage).all()
    assert sphere.center == expected
    assert sphere.radius == 0.1
    assert center.flags.writeable == (layout != "readonly")


@pytest.mark.parametrize("replacement", [np.nan, 99.0])
def test_interior_snapshot_precedes_surface_conversion(replacement):
    interior = np.zeros((1, 3))
    surface = np.concatenate((np.eye(3), -np.eye(3)))
    options = SphereFitOptions(min_radius=0.1)
    expected = fit_spheres(interior.copy(), surface, options)

    class MutatingSurface:
        calls = 0

        def __array__(self, dtype=None, copy=None):
            self.calls += 1
            interior[:] = replacement
            return np.asarray(surface, dtype=dtype)

    source = MutatingSurface()
    actual = fit_spheres(interior, source, options)
    assert source.calls == 1
    np.testing.assert_array_equal(interior, np.full((1, 3), replacement))
    assert actual == expected


@pytest.mark.parametrize("replacement", [np.nan, 0.0])
def test_evaluation_snapshot_precedes_sphere_iteration(replacement):
    points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sphere = SphereSpec((0.0, 0.0, 0.0), 1.0)
    expected = evaluate_sphere_fit(points.copy(), [sphere])
    calls = []

    def spheres():
        calls.append(True)
        points[:] = replacement
        yield sphere

    actual = evaluate_sphere_fit(points, spheres())
    assert calls == [True]
    np.testing.assert_array_equal(points, np.full((2, 3), replacement))
    assert actual == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_evaluation_workspace_stays_bounded_for_real_sample_dtypes(dtype):
    tracemalloc = tracemalloc_or_skip()
    points = np.zeros((100_000, 3), dtype=dtype)
    spheres = [SphereSpec((0.0, 0.0, 0.0), 1.0)]
    tracemalloc.start()
    try:
        result = evaluate_sphere_fit(points, spheres)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.sampled_coverage == 1.0
    assert result.maximum_uncovered_distance == 0.0
    # An owned float64 sample array fits; two complete copies exceed this budget.
    assert peak < 3 * 1024 * 1024


@pytest.mark.parametrize("replacement", [np.nan, 99.0])
def test_trimesh_adapter_captures_voxels_before_surface_sampling(
    monkeypatch, replacement
):
    interior = np.zeros((1, 3))
    surface = np.concatenate((np.eye(3), -np.eye(3)))
    expected = fit_spheres(interior.copy(), surface)
    voxels = SimpleNamespace(points=interior)
    voxels.fill = lambda: voxels
    mesh = SimpleNamespace(extents=np.ones(3), voxelized=lambda _pitch: voxels)
    calls = []

    def sample_surface(_mesh, count, *, seed):
        calls.append((count, seed))
        interior[:] = replacement
        return surface, None

    monkeypatch.setitem(
        sys.modules,
        "trimesh",
        SimpleNamespace(sample=SimpleNamespace(sample_surface=sample_surface)),
    )
    actual = fit_trimesh(mesh, pitch=0.1, surface_samples=6, random_seed=7)
    assert calls == [(6, 7)]
    np.testing.assert_array_equal(interior, np.full((1, 3), replacement))
    assert actual == expected


@pytest.mark.parametrize("entry", ["extents", "pitch", "voxels"])
def test_trimesh_adapter_rejects_complex_inputs_before_conversion(monkeypatch, entry):
    extents = np.ones(3)
    points = np.zeros((1, 3))
    pitch = 0.1
    if entry == "extents":
        extents = np.array([1.0 + 1j, 1.0, 1.0])
    elif entry == "pitch":
        pitch = np.complex128(0.1 + 1j)
    else:
        points = np.zeros((1, 3), dtype=complex)
        points[0, 0] = 1j
    voxels = SimpleNamespace(points=points)
    voxels.fill = lambda: voxels
    mesh = SimpleNamespace(extents=extents, voxelized=lambda _pitch: voxels)
    sampler = SimpleNamespace(
        sample_surface=lambda *_args, **_kwargs: (np.ones((4, 3)), None)
    )
    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(sample=sampler))
    with pytest.raises(TypeError):
        fit_trimesh(mesh, pitch=pitch, surface_samples=4)
