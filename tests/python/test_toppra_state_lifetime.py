"""Trajectory construction owns inputs and releases temporary motion buffers."""

import gc
import weakref

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory
from holistic_motion.trajectory.toppra import _NaturalCubicPath


@pytest.mark.parametrize("limit_name", ["max_velocity", "max_acceleration"])
def test_numpy_waypoints_are_owned_before_converting_limits(limit_name):
    points = np.array([[0.0], [1.0]])
    expected = points.copy()

    class MutatingLimit:
        def __array__(self, dtype=None, copy=None):
            points[-1, 0] = 2.0
            return np.array([1.0], dtype=dtype)

    limits = {"max_velocity": [1.0], "max_acceleration": [2.0]}
    limits[limit_name] = MutatingLimit()
    trajectory = ToppraTrajectory(points, **limits, grid_size=17)
    assert points[-1, 0] == 2.0
    np.testing.assert_array_equal(trajectory.waypoints, expected)
    sampled, *_ = trajectory.sample([0.0, trajectory.duration])
    np.testing.assert_allclose(sampled, expected, atol=1e-15)


@pytest.mark.parametrize("grid_size", [64, 2048])
def test_numpy_grid_positions_are_released_before_constraint_construction(
    monkeypatch, grid_size
):
    positions = []
    evaluate = _NaturalCubicPath.evaluate_all
    compute = ToppraTrajectory._compute

    def observe_positions(path, grid):
        values = evaluate(path, grid)
        positions.append((weakref.ref(values[0]), values[0].nbytes))
        return values

    def compute_after_release(trajectory, *args):
        assert positions
        # PyPy may need a second collection to release extension-backed arrays.
        gc.collect()
        gc.collect()
        assert all(reference() is None for reference, _ in positions)
        return compute(trajectory, *args)

    monkeypatch.setattr(_NaturalCubicPath, "evaluate_all", observe_positions)
    monkeypatch.setattr(ToppraTrajectory, "_compute", compute_after_release)
    trajectory = ToppraTrajectory(
        [[0.0, 0.0, 0.0], [1.0, 0.5, -0.25]],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        grid_size=grid_size,
    )
    assert sum(size for _, size in positions) == grid_size * 3 * 8
    sampled, *_ = trajectory.sample([0.0, trajectory.duration])
    np.testing.assert_allclose(sampled, trajectory.waypoints, atol=1e-15)


@pytest.mark.parametrize("limit_name", ["max_velocity", "max_acceleration"])
def test_torch_validates_owned_waypoints_after_converting_limits(limit_name):
    torch = pytest.importorskip("torch")
    from holistic_motion.trajectory import TorchToppraTrajectory

    points = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    expected = points.clone()

    class MutatingLimit(list):
        @property
        def dtype(self):
            points.fill_(float("nan"))
            return torch.float64

    limits = {"max_velocity": [1.0], "max_acceleration": [2.0]}
    limits[limit_name] = MutatingLimit([1.0])
    trajectory = TorchToppraTrajectory(points, **limits, grid_size=17)
    assert torch.isnan(points).all()
    duration = trajectory.duration
    sampled, *_ = trajectory.sample(torch.stack((torch.zeros_like(duration), duration)))
    torch.testing.assert_close(sampled, expected, rtol=0.0, atol=1e-15)


def test_torch_grid_derivatives_are_released_before_timing_solve(monkeypatch):
    torch = pytest.importorskip("torch")
    from holistic_motion.trajectory import TorchToppraTrajectory

    derivatives = []
    evaluate = TorchToppraTrajectory._evaluate
    solve = TorchToppraTrajectory._solve

    def observe_derivatives(trajectory, grid, **kwargs):
        values = evaluate(trajectory, grid, **kwargs)
        if not kwargs.get("position", True):
            derivatives.extend(weakref.ref(value) for value in values[1:])
        return values

    def solve_after_release(trajectory, *args):
        gc.collect()
        gc.collect()
        assert derivatives and all(reference() is None for reference in derivatives)
        return solve(trajectory, *args)

    monkeypatch.setattr(TorchToppraTrajectory, "_evaluate", observe_derivatives)
    monkeypatch.setattr(TorchToppraTrajectory, "_solve", solve_after_release)
    with torch.no_grad():
        trajectory = TorchToppraTrajectory(
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            1.0,
            2.0,
            grid_size=129,
        )
        assert torch.isfinite(trajectory.duration)


def test_torch_physical_result_sources_are_released_without_changing_sampling(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from holistic_motion.trajectory import TorchToppraTrajectory

    clone_sources = {}
    clone = torch.Tensor.clone

    def observe_clone(source, *args, **kwargs):
        result = clone(source, *args, **kwargs)
        clone_sources[id(result)] = (
            weakref.ref(source),
            source.numel() * source.element_size(),
        )
        return result

    monkeypatch.setattr(torch.Tensor, "clone", observe_clone)
    with torch.no_grad():
        points = torch.tensor([[[0.0], [1.0]], [[0.0], [2.0]]], dtype=torch.float64)
        trajectory = TorchToppraTrajectory(points, 1.0, 2.0, grid_size=129)
        result = trajectory.result
        physical_sources = [
            clone_sources[id(value)]
            for value in (result.path_speeds, result.path_accelerations)
        ]
        gc.collect()
        gc.collect()
        assert all(reference() is None for reference, _ in physical_sources)
        assert sum(size for _, size in physical_sources) == 2 * (129 + 128) * 8

        times = trajectory.duration[:, None] * torch.linspace(0.0, 1.0, 19)
        expected = trajectory.sample(times)
        result.path_speeds.fill_(100.0)
        result.path_accelerations.fill_(-100.0)
        for actual, reference in zip(trajectory.sample(times), expected):
            torch.testing.assert_close(actual, reference)
