"""Inactive subnormal limits and geometry must retain zero adjoints."""

import pytest

torch = pytest.importorskip("torch")

from holistic_motion.trajectory import TorchToppraTrajectory

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA device unavailable"
        ),
    ),
]


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("first_scale", [1.0, 1e305, 1e307])
@pytest.mark.parametrize("tiny_scale", [1e-200, 1e-300, 1e-310])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("point_count", [2, 3])
@pytest.mark.parametrize("scale_small_axis_limits", [False, True])
def test_subnormal_inactive_axis_preserves_motion_and_input_gradients(
    device, first_scale, tiny_scale, batched, point_count, scale_small_axis_limits
):
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.3], [0.1, 0.2]],
        dtype=torch.float64,
        device=device,
    )
    points = points[:point_count]
    if batched:
        points = torch.stack((points, points + 0.1))
    points.requires_grad_()
    velocity = torch.tensor(
        [0.8, 2.0], dtype=torch.float64, device=device, requires_grad=True
    )
    acceleration = torch.tensor(
        [1.2, 3.0], dtype=torch.float64, device=device, requires_grad=True
    )

    def evaluate(small):
        scale = torch.tensor([first_scale, small], dtype=torch.float64, device=device)
        limit_scale = torch.tensor(
            [first_scale, small if scale_small_axis_limits else 1.0],
            dtype=torch.float64,
            device=device,
        )
        trajectory = TorchToppraTrajectory(
            points * scale,
            velocity * limit_scale,
            acceleration * limit_scale,
            grid_size=32,
        )
        times, *physical_motion = trajectory.sample_uniform(31)
        motion = [value / scale for value in physical_motion]
        loss = trajectory.duration.sum() + sum(
            0.001 * value.square().mean() for value in motion
        )
        gradients = torch.autograd.grad(loss, (points, velocity, acceleration))
        # Both limits on the second axis are inactive, independently of its
        # chosen coordinate units. Their physical adjoints must stay zero.
        assert gradients[1][1] == 0.0
        assert gradients[2][1] == 0.0
        return trajectory.duration, times, *motion, *gradients

    expected = evaluate(1e-100)
    actual = evaluate(tiny_scale)
    for value, target in zip(actual, expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, rtol=2e-9, atol=2e-10)
