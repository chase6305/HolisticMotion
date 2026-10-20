"""Large coordinate units preserve differentiable timing and sampled motion."""

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
@pytest.mark.parametrize("scale", [1e306, 1e307])
@pytest.mark.parametrize("grid_size", [32, 200])
@pytest.mark.parametrize("end_velocity", [0.0, 0.05])
def test_large_geometry_preserves_values_and_input_gradients(
    device, scale, grid_size, end_velocity
):
    points = torch.tensor(
        [[0.0], [1.0], [0.0]], device=device, dtype=torch.float64, requires_grad=True
    )
    velocity = torch.ones(1, device=device, dtype=torch.float64, requires_grad=True)
    acceleration = torch.ones(1, device=device, dtype=torch.float64, requires_grad=True)

    def evaluate(coordinate_scale):
        trajectory = TorchToppraTrajectory(
            points * coordinate_scale,
            velocity * coordinate_scale,
            acceleration * coordinate_scale,
            grid_size=grid_size,
            end_path_velocity=end_velocity,
        )
        times, *motion = trajectory.sample_uniform(65)
        motion = [value / coordinate_scale for value in motion]
        loss = trajectory.duration + sum(
            0.01 * value.square().mean() for value in motion
        )
        gradients = torch.autograd.grad(loss, (points, velocity, acceleration))
        return trajectory.duration, times, *motion, *gradients

    expected = evaluate(1.0)
    actual = evaluate(scale)
    for value, target in zip(actual, expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_large_spline_fallback_keeps_small_independent_axes(device):
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float64, device=device
    )
    scales = torch.tensor([1e307, 1e-100], dtype=torch.float64, device=device)
    ordinary = TorchToppraTrajectory(points, 1.0, 1.0, grid_size=32)
    large = TorchToppraTrajectory(points * scales, scales, scales, grid_size=32)
    torch.testing.assert_close(large.duration, ordinary.duration)
    for actual, expected in zip(
        large.sample_uniform(101)[1:], ordinary.sample_uniform(101)[1:]
    ):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual / scales, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_finite_coordinate_derivatives_survive_overflowing_total_chord_length(device):
    # Each chord and derivative component is finite, while the sum of the two
    # four-dimensional chord lengths exceeds float64 range.
    points = torch.arange(3, dtype=torch.float64, device=device)[:, None].repeat(1, 4)
    points.requires_grad_()
    outputs = []
    for scale in (1.0, 5e307):
        trajectory = TorchToppraTrajectory(points * scale, scale, scale, grid_size=32)
        motion = [value / scale for value in trajectory.sample_uniform(65)[1:]]
        gradient = torch.autograd.grad(
            trajectory.duration + sum(0.01 * value.square().mean() for value in motion),
            points,
        )[0]
        outputs.append((trajectory.duration, *motion, gradient))
    for actual, expected in zip(outputs[1], outputs[0]):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_large_absolute_offset_preserves_local_spline_motion(device):
    points = torch.tensor(
        [[130.0], [131.0], [130.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    outputs = []
    for scale in (1.0, 1e306):
        trajectory = TorchToppraTrajectory(points * scale, scale, scale, grid_size=32)
        motion = [value / scale for value in trajectory.sample_uniform(65)[1:]]
        gradient = torch.autograd.grad(trajectory.duration, points)[0]
        outputs.append((trajectory.duration, *motion, gradient))
    for actual, expected in zip(outputs[1], outputs[0]):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_large_finite_chord_sum_has_finite_reverse_mode_gradients(device):
    generator = torch.Generator().manual_seed(1)
    points = 0.3 * torch.randn(8, 4, generator=generator, dtype=torch.float64)
    points[:, 0] = torch.linspace(0, 1, 8)
    points = points.to(device).requires_grad_()
    outputs = []
    for scale in (1.0, 1e305):
        trajectory = TorchToppraTrajectory(points * scale, scale, scale, grid_size=17)
        gradient = torch.autograd.grad(trajectory.duration, points)[0]
        outputs.append((trajectory.duration, gradient))
    for actual, expected in zip(outputs[1], outputs[0]):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_large_spline_evaluation_cancels_before_restoring_coordinates(device):
    # The middle segment's second derivative changes from -36 to +36.
    # With this scale, both endpoints and cubic coefficients are finite,
    # while forming the physical difference of those endpoints overflows.
    points = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0]], dtype=torch.float64, device=device
    )
    ordinary = TorchToppraTrajectory(points, 1.0, 1.0, grid_size=32)
    scale = 4e306
    large = TorchToppraTrajectory(points * scale, scale, scale, grid_size=32)
    parameters = torch.linspace(0, 1, 501, dtype=torch.float64, device=device)[None]
    for actual, expected in zip(
        large._evaluate(parameters), ordinary._evaluate(parameters)
    ):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual / scale, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_large_and_ordinary_paths_keep_independent_gradients(device):
    points = torch.tensor(
        [[[0.0, 0.0], [0.6, 0.4], [0.1, 0.2]], [[0.1, 0.2], [0.7, 0.6], [0.2, 0.4]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    velocity = torch.tensor(
        [[0.7, 0.8], [0.8, 0.9]], dtype=torch.float64, device=device, requires_grad=True
    )
    acceleration = torch.tensor(
        [[1.2, 1.4], [1.3, 1.5]], dtype=torch.float64, device=device, requires_grad=True
    )
    scales = torch.tensor([1e306, 1.0], dtype=torch.float64, device=device)[
        :, None, None
    ]
    outputs = []
    for scale in (torch.ones_like(scales), scales):
        trajectory = TorchToppraTrajectory(
            points * scale,
            velocity * scale[:, 0],
            acceleration * scale[:, 0],
            grid_size=32,
        )
        motion = [value / scale for value in trajectory.sample_uniform(65)[1:]]
        loss = trajectory.duration.sum() + 0.01 * motion[0].square().mean()
        gradients = torch.autograd.grad(loss, (points, velocity, acceleration))
        outputs.append((trajectory.duration, *motion, *gradients))
    for actual, expected in zip(outputs[1], outputs[0]):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [1e304, 1e305])
def test_large_geometry_gradients_match_directional_finite_differences(device, scale):
    generator = torch.Generator().manual_seed(1)
    points = 0.3 * torch.randn(8, 4, generator=generator, dtype=torch.float64)
    points[:, 0] = torch.linspace(0, 1, 8)
    inputs = (
        points.to(device).requires_grad_(),
        torch.linspace(
            0.7, 1.1, 4, dtype=torch.float64, device=device
        ).requires_grad_(),
        torch.linspace(
            1.2, 1.8, 4, dtype=torch.float64, device=device
        ).requires_grad_(),
    )
    directions = [
        torch.randn(value.shape, generator=generator, dtype=torch.float64).to(device)
        for value in inputs
    ]
    fractions = torch.linspace(0.013, 0.987, 31, dtype=torch.float64, device=device)

    def objective(points, velocity, acceleration):
        trajectory = TorchToppraTrajectory(
            points * scale, velocity * scale, acceleration * scale, grid_size=17
        )
        position = trajectory.sample(trajectory.duration * fractions)[0] / scale
        return trajectory.duration + 0.01 * position.square().mean()

    gradients = torch.autograd.grad(objective(*inputs), inputs)
    analytic = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(gradients, directions)
    )
    with torch.no_grad():
        step = 1e-6
        numerical = (
            objective(
                *(
                    value + step * direction
                    for value, direction in zip(inputs, directions)
                )
            )
            - objective(
                *(
                    value - step * direction
                    for value, direction in zip(inputs, directions)
                )
            )
        ) / (2 * step)
    assert torch.isfinite(analytic)
    torch.testing.assert_close(analytic, numerical, rtol=2e-5, atol=2e-5)
