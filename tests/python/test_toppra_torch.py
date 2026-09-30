"""CPU/CUDA numerical and autograd contracts for the optional Torch backend."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from holistic_motion.trajectory import (
    ToppraTrajectory,
    TorchToppraTrajectory,
    retime_path_torch,
)

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


def points(device, dtype=torch.float64, batch=False):
    value = torch.tensor(
        [[0.13, -0.12], [0.35, -0.38], [0.79, 0.16], [1.17, 0.27]],
        dtype=dtype,
        device=device,
    )
    return torch.stack((value, value * 0.7 + 0.2)) if batch else value


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("boundary", [(0.0, 0.0), (0.1, 0.2), (0.2, 0.0)])
def test_matches_numpy_on_same_grid_and_continuous_limits(device, batch, boundary):
    p = points(device, batch=batch)
    kwargs = {
        "start_path_velocity": boundary[0],
        "end_path_velocity": boundary[1],
        "grid_size": 22,
    }
    trajectory = retime_path_torch(p, [0.8, 1.1], [1.7, 2.1], **kwargs)
    times, q, dq, ddq = trajectory.sample_uniform(1001)
    grids = trajectory.result.gridpoints
    paths = p if batch else p[None]
    for index, path in enumerate(paths):
        grid = grids[index] if batch else grids
        reference = ToppraTrajectory(
            path.cpu().numpy(),
            [0.8, 1.1],
            [1.7, 2.1],
            gridpoints=grid.cpu().numpy(),
            **kwargs,
        )
        actual_times = times[index] if batch else times
        expected = reference.sample(actual_times.cpu().numpy())
        for actual, target in zip((q, dq, ddq), expected):
            actual = actual[index] if batch else actual
            np.testing.assert_allclose(
                actual.cpu().numpy(), target, rtol=1e-8, atol=1e-9
            )
        actual_speeds = (
            trajectory.result.path_speeds[index]
            if batch
            else trajectory.result.path_speeds
        )
        np.testing.assert_allclose(
            actual_speeds.cpu().numpy(),
            reference.result.path_speeds,
            rtol=1e-10,
            atol=1e-12,
        )
    assert torch.all(
        dq.abs() <= torch.tensor([0.8, 1.1], device=device, dtype=torch.float64) + 1e-9
    )
    assert torch.all(
        ddq.abs() <= torch.tensor([1.7, 2.1], device=device, dtype=torch.float64) + 1e-9
    )
    assert q.device == p.device
    assert times.dtype == torch.float64


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
def test_gradcheck_waypoints_limits_and_boundaries(device, batch):
    p = points(device, batch=batch).requires_grad_()
    v = torch.tensor(
        [0.83, 1.07], dtype=torch.float64, device=device, requires_grad=True
    )
    a = torch.tensor(
        [1.63, 2.19], dtype=torch.float64, device=device, requires_grad=True
    )
    start = torch.tensor(0.12, dtype=torch.float64, device=device, requires_grad=True)
    end = torch.tensor(0.17, dtype=torch.float64, device=device, requires_grad=True)

    def evaluate(p, v, a, start, end):
        trajectory = TorchToppraTrajectory(
            p, v, a, start_path_velocity=start, end_path_velocity=end, grid_size=8
        )
        fractions = torch.tensor(
            [0.137, 0.413, 0.823], dtype=torch.float64, device=device
        )
        sample_times = trajectory.duration[..., None] * fractions
        return (trajectory.duration, *trajectory.sample(sample_times))

    assert torch.autograd.gradcheck(
        evaluate,
        (p, v, a, start, end),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=False,
        # CUDA gather backward accumulates repeated spline indices atomically.
        # Permit double-precision summation roundoff, not Jacobian mismatch.
        nondet_tol=1e-12 if device == "cuda" else 0.0,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_zero_boundaries_low_precision_and_repeated_backward(device, dtype):
    p = points(device, dtype=dtype).requires_grad_()
    v = torch.tensor([0.8, 1.1], device=device, dtype=dtype, requires_grad=True)
    a = torch.tensor([1.7, 2.1], device=device, dtype=dtype, requires_grad=True)
    trajectory = TorchToppraTrajectory(p, v, a, grid_size=13)
    times, q, dq, ddq = trajectory.sample_uniform(17)
    assert all(value.dtype == dtype for value in (q, dq, ddq))
    assert times.dtype == torch.float64
    assert torch.equal(
        trajectory.result.path_speeds[[0, -1]],
        torch.zeros(2, device=device, dtype=torch.float64),
    )
    loss = (
        trajectory.duration
        + q.double().square().mean()
        + 0.03 * dq.double().square().mean()
        + 0.01 * ddq.double().square().mean()
    )
    first = torch.autograd.grad(loss, (p, v, a), retain_graph=True)
    second = torch.autograd.grad(loss, (p, v, a))
    for before, after, input_value in zip(first, second, (p, v, a)):
        assert before.dtype == input_value.dtype
        assert torch.isfinite(before).all()
        assert before.abs().sum() > 0
        torch.testing.assert_close(before, after)


@pytest.mark.parametrize("device", DEVICES)
def test_sample_time_derivative_and_clamping(device):
    trajectory = TorchToppraTrajectory(
        points(device), [0.8, 1.1], [1.7, 2.1], grid_size=17
    )
    query = (
        trajectory.duration.detach()
        * torch.tensor([0.13, 0.43, 0.79], device=device, dtype=torch.float64)
    ).requires_grad_()
    q, dq, _ = trajectory.sample(query)
    gradient = torch.autograd.grad(q.sum(), query)[0]
    torch.testing.assert_close(gradient, dq.sum(-1))
    q, _, _ = trajectory.sample(
        torch.stack((-trajectory.duration, trajectory.duration * 2))
    )
    torch.testing.assert_close(q, points(device)[[0, -1]])
    assert trajectory.sample(torch.empty(0, device=device))[0].shape == (0, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_limits_noncontiguous_inputs_and_independent_gradients(device):
    p = (
        points(device, batch=True)
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
        .requires_grad_()
    )
    v = torch.tensor(
        [[0.8, 1.1], [0.6, 0.9]], device=device, dtype=torch.float64, requires_grad=True
    )
    a = torch.tensor(
        [[1.7, 2.1], [1.2, 1.5]], device=device, dtype=torch.float64, requires_grad=True
    )
    batch = TorchToppraTrajectory(p, v, a, grid_size=14)
    for index in range(2):
        single = TorchToppraTrajectory(p[index], v[index], a[index], grid_size=14)
        torch.testing.assert_close(batch.duration[index], single.duration)
    gradients = torch.autograd.grad(batch.duration[0], (p, v, a))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[1]) == 0
        assert torch.count_nonzero(gradient[0]) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_linear_path_analytic_duration_and_gradient(device):
    distance = torch.tensor(1.3, device=device, dtype=torch.float64, requires_grad=True)
    acceleration = torch.tensor(
        2.1, device=device, dtype=torch.float64, requires_grad=True
    )
    p = torch.stack((distance * 0, distance)).reshape(2, 1)
    trajectory = TorchToppraTrajectory(p, 10.0, acceleration, grid_size=3)
    expected = 2 * torch.sqrt(distance / acceleration)
    torch.testing.assert_close(trajectory.duration, expected)
    actual_grad = torch.autograd.grad(
        trajectory.duration, (distance, acceleration), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected, (distance, acceleration))
    for actual, target in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, target)


@pytest.mark.parametrize("device", DEVICES)
def test_result_and_input_mutation_do_not_change_sampling(device):
    p = points(device)
    v = torch.tensor([0.8, 1.1], device=device)
    trajectory = TorchToppraTrajectory(p, v, 2.0, grid_size=12)
    before = trajectory.sample_uniform(19)
    p.fill_(0)
    v.fill_(100)
    trajectory.result.path_speeds.fill_(100)
    trajectory.result.times.fill_(0)
    for actual, expected in zip(trajectory.sample_uniform(19), before):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"grid_size": 1}, "grid_size"),
        ({"max_velocity": 0.0}, "positive"),
        ({"max_acceleration": float("nan")}, "finite"),
        ({"max_velocity": [1, 2, 3]}, "scalar"),
        ({"start_path_velocity": -1}, "boundary"),
        ({"end_path_velocity": 100}, "infeasible|velocity"),
    ],
)
def test_rejects_invalid_or_infeasible_inputs(device, kwargs, message):
    options = {"max_velocity": 1.0, "max_acceleration": 2.0, "grid_size": 8}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TorchToppraTrajectory(points(device), **options)


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_row_rejects_entire_batch(device):
    p = points(device, batch=True)
    p[1, 1] = p[1, 0]
    with pytest.raises(ValueError, match="distinct"):
        TorchToppraTrajectory(p, 1.0, 2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_current_stream_forward_and_backward_match_cpu():
    cpu = points("cpu", batch=True).requires_grad_()
    gpu = cpu.detach().cuda().requires_grad_()
    cpu_trajectory = TorchToppraTrajectory(cpu, 1.0, 2.0, grid_size=15)
    cpu_loss = (
        cpu_trajectory.duration.sum()
        + cpu_trajectory.sample_uniform(21)[1].square().mean()
    )
    cpu_gradient = torch.autograd.grad(cpu_loss, cpu)[0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gpu_trajectory = TorchToppraTrajectory(gpu, 1.0, 2.0, grid_size=15)
        gpu_loss = (
            gpu_trajectory.duration.sum()
            + gpu_trajectory.sample_uniform(21)[1].square().mean()
        )
        gpu_gradient = torch.autograd.grad(gpu_loss, gpu)[0]
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(gpu_loss.cpu(), cpu_loss)
    torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_boundary_speeds_and_limit_gradients(device):
    p = points(device, batch=True)
    v = torch.tensor(
        [[0.83, 1.07], [0.61, 0.79]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    a = torch.tensor(
        [[1.63, 2.19], [1.21, 1.58]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    start = torch.tensor(
        [0.12, 0.11], device=device, dtype=torch.float64, requires_grad=True
    )
    end = torch.tensor(
        [0.17, 0.23], device=device, dtype=torch.float64, requires_grad=True
    )

    def duration(v, a, start, end):
        return TorchToppraTrajectory(
            p, v, a, start_path_velocity=start, end_path_velocity=end, grid_size=8
        ).duration

    assert torch.autograd.gradcheck(duration, (v, a, start, end), fast_mode=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("time_scale", [1e-4, 1e4])
def test_tensor_retiming_respects_time_units(device, time_scale):
    p = points(device)
    original = TorchToppraTrajectory(p, 1.0, 2.0, grid_size=13)
    scaled = TorchToppraTrajectory(p, time_scale, 2 * time_scale**2, grid_size=13)
    torch.testing.assert_close(scaled.duration * time_scale, original.duration)
    for order, (actual, expected) in enumerate(
        zip(scaled.sample_uniform(101)[1:], original.sample_uniform(101)[1:])
    ):
        torch.testing.assert_close(
            actual / time_scale**order, expected, atol=1e-9, rtol=1e-8
        )


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_backend_does_not_call_numpy_solver(device, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected NumPy fallback")

    monkeypatch.setattr(ToppraTrajectory, "__init__", fail)
    p = points(device).requires_grad_()
    trajectory = TorchToppraTrajectory(p, 1.0, 2.0, grid_size=9)
    trajectory.duration.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_query_validation_and_unrepresentable_half_output(device):
    trajectory = TorchToppraTrajectory(points(device), 1.0, 2.0, grid_size=9)
    with pytest.raises(ValueError, match="finite"):
        trajectory.sample([float("nan")])
    with pytest.raises(ValueError, match="shape"):
        trajectory.sample([[0.0], [0.1]])
    with pytest.raises(TypeError, match="count"):
        trajectory.sample_uniform(True)
    with pytest.raises(ValueError, match="count"):
        trajectory.sample_uniform(1)
    fast = TorchToppraTrajectory(
        points(device, dtype=torch.float16), 1e5, 1e10, grid_size=9
    )
    with pytest.raises(ValueError, match="representable"):
        fast.sample_uniform(17)
