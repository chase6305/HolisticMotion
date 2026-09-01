import json

import holistic_motion as hm
import numpy as np
import pytest
from holistic_motion.geometry import (
    SphereFitMetrics,
    SphereFitOptions,
    SphereFitResult,
    SphereSpec,
    evaluate_sphere_fit,
    fit_spheres,
    load_sphere_model,
    make_collision_spheres,
    save_sphere_model,
)


def _unit_sphere_samples():
    surface = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    interior = np.vstack(([0.0, 0.0, 0.0], surface * 0.5))
    return interior, surface


def test_inscribed_fit_is_deterministic_and_reports_metrics():
    interior, surface = _unit_sphere_samples()
    options = SphereFitOptions(max_spheres=4, min_radius=0.1)
    first = fit_spheres(interior, surface, options)
    second = fit_spheres(interior, surface, options)

    assert first == second
    assert first.mode == "inscribed"
    assert first.spheres[0].center == (0.0, 0.0, 0.0)
    assert first.spheres[0].radius == pytest.approx(1.0)
    assert first.metrics.sampled_coverage == pytest.approx(1.0)


def test_sampled_coverage_expands_to_cover_supplied_points():
    interior = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    surface = np.array([[-1.2, 0, 0], [0.0, 0, 0], [1.2, 0, 0]])
    result = fit_spheres(
        interior,
        surface,
        SphereFitOptions(
            max_spheres=2,
            min_radius=0.1,
            padding=0.01,
            sampled_coverage=True,
        ),
    )
    assert result.mode == "sampled_coverage"
    assert result.metrics.sampled_coverage == pytest.approx(1.0)
    assert max(sphere.radius for sphere in result.spheres) >= 1.01
    assert min(sphere.radius for sphere in result.spheres) == pytest.approx(0.21)


def test_metrics_reject_empty_sphere_set():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_sphere_fit([[0, 0, 0]], [])


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_metrics_reject_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        evaluate_sphere_fit(
            [[0, 0, 0]], [SphereSpec((0, 0, 0), 1.0)], chunk_size=chunk_size
        )


def test_sphere_model_json_round_trip_and_native_conversion(tmp_path):
    path = tmp_path / "spheres.json"
    links = {
        "left": (SphereSpec((0, 0, 0), 0.2),),
        "right": (SphereSpec((0.1, 0, 0), 0.1),),
    }
    save_sphere_model(path, links, metadata={"urdf": "robot.urdf"})
    loaded, metadata = load_sphere_model(path)
    assert loaded == links
    assert metadata == {"urdf": "robot.urdf"}

    if not hasattr(hm, "CollisionSphere"):
        pytest.skip("native collision support is disabled")
    native = make_collision_spheres(loaded)
    assert len(native) == 2
    assert native[0].name == "left_sphere_0"
    assert native[1].link_name == "right"

    payload = json.loads(path.read_text())
    payload["version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported"):
        load_sphere_model(path)


def test_sphere_model_save_is_atomic_on_validation_failure(tmp_path):
    path = tmp_path / "spheres.json"
    original = {"body": (SphereSpec((0, 0, 0), 0.2),)}
    save_sphere_model(path, original)
    before = path.read_bytes()

    with pytest.raises(ValueError):
        save_sphere_model(path, {"body": ()})

    assert path.read_bytes() == before
    assert list(tmp_path.glob(".spheres.json.*.tmp")) == []


def test_sphere_model_load_reports_malformed_entries(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text(
        json.dumps(
            {
                "format": "holistic_motion.sphere_model",
                "version": 1,
                "links": {"body": [{"center": [0, 0, 0]}]},
            }
        )
    )
    with pytest.raises(ValueError, match="invalid sphere entry"):
        load_sphere_model(path)

    path.write_text("[]")
    with pytest.raises(TypeError, match="root must be an object"):
        load_sphere_model(path)


def test_sphere_model_public_boundaries_validate_types(tmp_path):
    path = tmp_path / "spheres.json"
    sphere = SphereSpec((0, 0, 0), 0.2)
    with pytest.raises(ValueError, match="string name"):
        save_sphere_model(path, {1: (sphere,)})
    with pytest.raises(TypeError, match="SphereSpec"):
        save_sphere_model(path, {"body": (object(),)})
    with pytest.raises(TypeError, match="metadata"):
        save_sphere_model(path, {"body": (sphere,)}, metadata=[])
    with pytest.raises(TypeError, match="SphereSpec"):
        make_collision_spheres({"body": (object(),)})


def test_sphere_fit_public_boundaries_validate_types():
    points = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="SphereFitOptions"):
        fit_spheres(points, points, options=object())
    with pytest.raises(TypeError, match="SphereSpec"):
        evaluate_sphere_fit(points, [object()])
    with pytest.raises(TypeError, match="iterable"):
        evaluate_sphere_fit(points, None)


def test_sphere_fit_options_validate_values():
    with pytest.raises(ValueError):
        SphereFitOptions(max_spheres=0)
    with pytest.raises(ValueError):
        SphereFitOptions(max_spheres=2.5)
    with pytest.raises(ValueError):
        SphereFitOptions(padding=float("nan"))
    with pytest.raises(TypeError, match="sampled_coverage"):
        SphereFitOptions(sampled_coverage=1)
    options = SphereFitOptions(max_spheres=np.int64(2), chunk_size=np.int64(16))
    assert type(options.max_spheres) is int
    assert type(options.chunk_size) is int
    with pytest.raises(ValueError):
        SphereSpec((0, 0, 0), -1.0)
    with pytest.raises(TypeError, match="radius must be numeric"):
        SphereSpec((0, 0, 0), object())


def test_sphere_fit_result_value_objects_validate_and_own_state():
    sphere = SphereSpec((0, 0, 0), 1.0)
    metrics = SphereFitMetrics(1.0, 0.0, 0.0, 1)
    source = [sphere]
    result = SphereFitResult(source, metrics, "inscribed")
    source.clear()
    assert result.spheres == (sphere,)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SphereFitMetrics(1.1, 0.0, 0.0, 1)
    with pytest.raises(ValueError, match="cannot exceed"):
        SphereFitMetrics(1.0, 1.0, 0.5, 1)
    with pytest.raises(TypeError, match="sampled_coverage must be numeric"):
        SphereFitMetrics(object(), 0.0, 0.0, 1)
    with pytest.raises(TypeError, match="integer"):
        SphereFitMetrics(1.0, 0.0, 0.0, True)
    with pytest.raises(ValueError, match="must match"):
        SphereFitResult((sphere,), SphereFitMetrics(1.0, 0.0, 0.0, 2), "fit")


def test_geometry_is_available_from_package_namespace():
    assert hm.geometry.fit_spheres is fit_spheres
