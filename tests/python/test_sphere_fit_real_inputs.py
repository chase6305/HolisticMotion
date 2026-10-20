"""Sphere fitting must reject complex geometry before float conversion."""

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
