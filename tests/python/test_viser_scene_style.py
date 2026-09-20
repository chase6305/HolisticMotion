"""Renderer-independent checks for the shared Viser scene environment."""

import pytest
from holistic_motion.visualization.viser import (
    configure_scene,
    configure_scene_from_bounds,
)


class ModernScene:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return "ground" if name == "add_grid" else None

        return record


class LegacyScene:
    def __init__(self):
        self.lights_enabled = False
        self.grid = None

    def enable_default_lights(self, enabled):
        self.lights_enabled = enabled

    def add_grid(self, name, width, height):
        self.grid = (name, width, height)
        return "legacy-ground"


class TransitionalScene:
    def __init__(self):
        self.default_lights_enabled = False
        self.hdri = None
        self.grid = None

    def set_up_direction(self, direction):
        assert direction == "+z"

    def configure_default_lights(self, *, enabled):
        self.default_lights_enabled = enabled

    def configure_environment_map(self, *, hdri):
        self.hdri = hdri

    def add_grid(self, name, width, height, *, position):
        self.grid = (name, width, height, position)
        return "transitional-ground"


def test_configure_scene_applies_studio_environment():
    scene = ModernScene()
    assert (
        configure_scene(scene, ground_width=6.0, ground_height=4.0, ground_z=-0.2)
        == "ground"
    )
    calls = {name: (args, kwargs) for name, args, kwargs in scene.calls}
    assert calls["set_up_direction"][0] == ("+z",)
    assert calls["configure_default_lights"][1] == {
        "enabled": True,
        "cast_shadow": True,
    }
    assert calls["configure_environment_map"][1]["hdri"] == "studio"
    assert calls["configure_environment_map"][1]["background"] is True
    assert calls["add_light_directional"][1]["cast_shadow"] is True
    grid = calls["add_grid"][1]
    assert (grid["width"], grid["height"], grid["position"]) == (
        6.0,
        4.0,
        (0.0, 0.0, -0.2),
    )
    assert grid["plane_opacity"] > 0.0
    assert grid["shadow_opacity"] > 0.0
    assert grid["cell_size"] == pytest.approx(0.2)
    assert grid["section_size"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "extent,expected",
    [(0.4, 0.01), (3.0, 0.1), (8.0, 0.2), (120.0, 5.0)],
)
def test_configure_scene_selects_readable_grid_scale(extent, expected):
    scene = ModernScene()
    configure_scene(scene, ground_width=extent)
    grid = next(kwargs for name, _args, kwargs in scene.calls if name == "add_grid")
    assert grid["cell_size"] == pytest.approx(expected)
    assert grid["section_size"] == pytest.approx(5.0 * expected)


def test_configure_scene_accepts_explicit_grid_scale():
    scene = ModernScene()
    configure_scene(scene, ground_width=4.0, grid_cell_size=0.25)
    grid = next(kwargs for name, _args, kwargs in scene.calls if name == "add_grid")
    assert grid["cell_size"] == pytest.approx(0.25)
    assert grid["section_size"] == pytest.approx(1.25)


def test_configure_scene_handles_smallest_positive_ground_extent():
    scene = ModernScene()
    configure_scene(scene, ground_width=float.fromhex("0x0.0000000000001p-1022"))
    grid = next(kwargs for name, _args, kwargs in scene.calls if name == "add_grid")
    assert grid["cell_size"] > 0.0
    assert grid["section_size"] > grid["cell_size"]


def test_configure_scene_falls_back_to_legacy_grid_and_lights():
    scene = LegacyScene()
    assert configure_scene(scene, ground_width=3) == "legacy-ground"
    assert scene.lights_enabled
    assert scene.grid == ("/environment/ground", 3.0, 3.0)


def test_configure_scene_preserves_position_on_transitional_api():
    scene = TransitionalScene()
    assert (
        configure_scene(
            scene,
            ground_width=3.0,
            ground_height=2.0,
            ground_z=-0.4,
            ground_center_xy=(1.0, 2.0),
        )
        == "transitional-ground"
    )
    assert scene.default_lights_enabled
    assert scene.hdri == "studio"
    assert scene.grid == (
        "/environment/ground",
        3.0,
        2.0,
        (1.0, 2.0, -0.4),
    )


def test_configure_scene_from_bounds_centers_and_clears_floor():
    scene = ModernScene()
    configure_scene_from_bounds(
        scene,
        [[10.0, 20.0, -1.0], [12.0, 21.0, 3.0]],
        margin=0.25,
    )
    grid = next(kwargs for name, _args, kwargs in scene.calls if name == "add_grid")
    assert grid["width"] == pytest.approx(4.0)
    assert grid["height"] == pytest.approx(3.0)
    assert grid["position"] == pytest.approx((11.0, 20.5, -1.08))


def test_configure_scene_from_large_offset_bounds_uses_stable_center():
    scene = ModernScene()
    configure_scene_from_bounds(
        scene,
        [[1.0e308, -1.1e308, 0.0], [1.1e308, -1.0e308, 1.0]],
    )
    grid = next(kwargs for name, _args, kwargs in scene.calls if name == "add_grid")
    assert grid["position"][:2] == pytest.approx((1.05e308, -1.05e308))


@pytest.mark.parametrize(
    "options,error",
    [
        ({"ground_width": 0.0}, ValueError),
        ({"ground_height": float("inf")}, ValueError),
        ({"ground_z": float("nan")}, ValueError),
        ({"ground_center_xy": [0.0]}, ValueError),
        ({"grid_cell_size": 0.0}, ValueError),
        ({"ground_width": object()}, TypeError),
    ],
)
def test_configure_scene_rejects_invalid_geometry(options, error):
    with pytest.raises(error):
        configure_scene(ModernScene(), **options)


@pytest.mark.parametrize(
    "bounds,options,error",
    [
        ([[0.0, 0.0], [1.0, 1.0]], {}, ValueError),
        ([[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]], {}, ValueError),
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], {"margin": -0.1}, ValueError),
        (
            [[-1.0e308, 0.0, 0.0], [1.0e308, 1.0, 1.0]],
            {},
            ValueError,
        ),
        (
            [[0.0, 0.0, 0.0], [1.0e308, 1.0, 1.0]],
            {"margin": 1.0e308},
            ValueError,
        ),
    ],
)
def test_configure_scene_from_bounds_rejects_invalid_input(bounds, options, error):
    with pytest.raises(error):
        configure_scene_from_bounds(ModernScene(), bounds, **options)
