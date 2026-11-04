"""Construction-only motion buffers do not remain owned by trajectories."""

import weakref

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory
from holistic_motion.trajectory.toppra import _NaturalCubicPath


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
        assert all(reference() is None for reference, _ in physical_sources)
        assert sum(size for _, size in physical_sources) == 2 * (129 + 128) * 8

        times = trajectory.duration[:, None] * torch.linspace(0.0, 1.0, 19)
        expected = trajectory.sample(times)
        result.path_speeds.fill_(100.0)
        result.path_accelerations.fill_(-100.0)
        for actual, reference in zip(trajectory.sample(times), expected):
            torch.testing.assert_close(actual, reference)
