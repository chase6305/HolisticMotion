"""Two-waypoint timing agrees with independent NumPy and analytic references."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from holistic_motion.trajectory import ToppraTrajectory, TorchToppraTrajectory

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
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dof", [1, 7])
@pytest.mark.parametrize("grid_size", [3, 32, 101])
@pytest.mark.parametrize("boundary", [(0.0, 0.0), (0.03, 0.06), (0.05, 0.0)])
def test_linear_path_matches_numpy_on_the_same_grid(
    device, batched, dof, grid_size, boundary
):
    rng = np.random.default_rng(5629)
    points = rng.normal(size=(2, 2, dof))
    velocity = rng.uniform(0.7, 1.3, (2, dof))
    acceleration = rng.uniform(1.4, 2.6, (2, dof))
    inputs = (points, velocity, acceleration)
    if not batched:
        inputs = tuple(value[0] for value in inputs)
    actual = TorchToppraTrajectory(
        *(torch.tensor(value, device=device, dtype=torch.float64) for value in inputs),
        grid_size=grid_size,
        start_path_velocity=boundary[0],
        end_path_velocity=boundary[1],
    )
    # Do not compare acceleration exactly at a jump: backend clock rounding
    # can place the same float query on opposite sides of that discontinuity.
    fractions = torch.tensor(
        np.sort(rng.uniform(0.001, 0.999, 65)), device=device, dtype=torch.float64
    )
    times = actual.duration[..., None] * fractions
    motion = actual.sample(times)
    for index in range(2 if batched else 1):

        def select(value, index=index):
            return value[index] if batched else value

        reference = ToppraTrajectory(
            points[index],
            velocity[index],
            acceleration[index],
            gridpoints=select(actual.result.gridpoints).cpu().numpy(),
            start_path_velocity=boundary[0],
            end_path_velocity=boundary[1],
        )
        np.testing.assert_allclose(
            select(actual.result.path_speeds).cpu().numpy(),
            reference.result.path_speeds,
            rtol=1e-10,
            atol=1e-11,
        )
        expected = reference.sample(select(times).cpu().numpy())
        for value, target in zip(motion, expected):
            np.testing.assert_allclose(
                select(value).cpu().numpy(), target, rtol=1e-8, atol=1e-9
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid_size", [3, 32, 101])
def test_linear_samples_at_own_nodes_use_the_right_interval(device, grid_size):
    points = torch.tensor([[0.1, -0.2], [1.3, 0.4]], device=device, dtype=torch.float64)
    trajectory = TorchToppraTrajectory(
        points, [0.8, 1.2], [1.4, 2.1], grid_size=grid_size
    )
    result = trajectory.result
    position, velocity, acceleration = trajectory.sample(result.times)
    delta = points[1] - points[0]
    expected_position = points[0] + result.gridpoints[:, None] * delta
    expected_velocity = result.path_speeds[:, None] * delta
    right_acceleration = (
        torch.cat((result.path_accelerations, result.path_accelerations[-1:]))[:, None]
        * delta
    )
    for actual, expected in zip(
        (position, velocity, acceleration),
        (expected_position, expected_velocity, right_acceleration),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid_size", [3, 33, 201])
def test_linear_triangle_duration_and_gradients_match_analytic_solution(
    device, grid_size
):
    distance = torch.tensor(1.3, dtype=torch.float64, device=device, requires_grad=True)
    acceleration = torch.tensor(
        2.1, dtype=torch.float64, device=device, requires_grad=True
    )
    points = torch.stack((distance * 0.0, distance))[:, None]
    actual = TorchToppraTrajectory(points, 100.0, acceleration, grid_size=grid_size)
    expected = 2.0 * torch.sqrt(distance / acceleration)
    torch.testing.assert_close(actual.duration, expected, rtol=1e-12, atol=1e-13)
    gradients = torch.autograd.grad(actual.duration, (distance, acceleration))
    reference = torch.autograd.grad(expected, (distance, acceleration))
    for value, target in zip(gradients, reference):
        torch.testing.assert_close(value, target, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "start,end,velocity,acceleration",
    [
        (2.0, 0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0, 1.0),
        (0.0, 0.9, 2.0, 0.1),
        (0.9, 0.0, 2.0, 0.1),
    ],
)
def test_linear_path_rejects_unreachable_boundary_velocities(
    device, start, end, velocity, acceleration
):
    points = torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match="infeasible|velocity"):
        TorchToppraTrajectory(
            points,
            velocity,
            acceleration,
            start_path_velocity=start,
            end_path_velocity=end,
            grid_size=17,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_linear_batch_rejects_one_unreachable_boundary(device):
    points = torch.tensor(
        [[[0.0], [1.0]], [[0.0], [1.0]]], dtype=torch.float64, device=device
    )
    with pytest.raises(ValueError, match="infeasible|velocity"):
        TorchToppraTrajectory(
            points, 1.0, 1.0, end_path_velocity=[0.0, 2.0], grid_size=17
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("units", ["space", "time"])
def test_linear_mixed_batch_units_preserve_values_and_gradients(device, units):
    points = (
        torch.tensor([[[0.1, -0.2], [1.3, 0.4]]], dtype=torch.float64, device=device)
        .repeat(4, 1, 1)
        .requires_grad_()
    )
    velocity = torch.tensor(
        [0.83, 1.07], dtype=torch.float64, device=device, requires_grad=True
    )
    acceleration = torch.tensor(
        [1.63, 2.19], dtype=torch.float64, device=device, requires_grad=True
    )
    start = torch.full(
        (4,), 0.12, dtype=torch.float64, device=device, requires_grad=True
    )
    end = torch.tensor(
        [0.0, 0.17, 0.13, 0.0], dtype=torch.float64, device=device, requires_grad=True
    )
    fractions = torch.linspace(0.013, 0.987, 31, dtype=torch.float64, device=device)
    scale = torch.tensor(
        [1.0, 1e100, 1e200, 1e305]
        if units == "space"
        else [1e-120, 1e-60, 1e60, 1e120],
        dtype=torch.float64,
        device=device,
    )

    def evaluate(scale):
        space = scale if units == "space" else torch.ones_like(scale)
        time = scale if units == "time" else torch.ones_like(scale)
        trajectory = TorchToppraTrajectory(
            points * space[:, None, None],
            (velocity * space[:, None]) * time[:, None],
            ((acceleration * space[:, None]) * time[:, None]) * time[:, None],
            start_path_velocity=start * time,
            end_path_velocity=end * time,
            grid_size=32,
        )
        duration = trajectory.duration * time
        q, dq, ddq = trajectory.sample(trajectory.duration[:, None] * fractions)
        motion = (
            q / space[:, None, None],
            dq / space[:, None, None] / time[:, None, None],
            ddq / space[:, None, None] / time[:, None, None] / time[:, None, None],
        )
        loss = duration.sum() + sum(0.001 * value.square().mean() for value in motion)
        gradients = torch.autograd.grad(
            loss, (points, velocity, acceleration, start, end)
        )
        return duration, *motion, *gradients

    expected = evaluate(torch.ones_like(scale))
    actual = evaluate(scale)
    for value, target in zip(actual, expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_linear_waypoint_limit_and_boundary_gradients_match_finite_differences(
    device, batched
):
    points = torch.tensor(
        [[0.11, -0.17], [1.23, 0.47]], dtype=torch.float64, device=device
    )
    if batched:
        points = torch.stack((points, points + 0.25))
    points.requires_grad_()
    velocity = torch.tensor(
        [0.83, 1.07], dtype=torch.float64, device=device, requires_grad=True
    )
    acceleration = torch.tensor(
        [1.63, 2.19], dtype=torch.float64, device=device, requires_grad=True
    )
    start = torch.tensor(0.12, dtype=torch.float64, device=device, requires_grad=True)
    end = torch.tensor(0.17, dtype=torch.float64, device=device, requires_grad=True)
    fractions = torch.tensor(
        [0.083, 0.217, 0.431, 0.637, 0.853], dtype=torch.float64, device=device
    )

    def objective(points, velocity, acceleration, start, end):
        trajectory = TorchToppraTrajectory(
            points,
            velocity,
            acceleration,
            start_path_velocity=start,
            end_path_velocity=end,
            grid_size=17,
        )
        q, dq, _ = trajectory.sample(trajectory.duration[..., None] * fractions)
        return (
            trajectory.duration.sum()
            + 0.001 * q.square().mean()
            + 0.0001 * dq.square().mean()
        )

    assert torch.autograd.gradcheck(
        objective,
        (points, velocity, acceleration, start, end),
        fast_mode=True,
        atol=2e-5,
        rtol=2e-4,
        # Repeated CUDA gather indices accumulate with double-precision
        # atomic summation, as in the general-path gradcheck.
        nondet_tol=1e-12 if device == "cuda" else 0.0,
    )
