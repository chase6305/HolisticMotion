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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("time_scale", [1e-120, 1e-60, 1e60, 1e120])
def test_time_units_preserve_sampled_loss_and_all_input_gradients(device, time_scale):
    p = points(device).requires_grad_()
    v = torch.tensor(
        [0.83, 1.07], device=device, dtype=torch.float64, requires_grad=True
    )
    a = torch.tensor(
        [1.63, 2.19], device=device, dtype=torch.float64, requires_grad=True
    )
    start = torch.tensor(0.12, device=device, dtype=torch.float64, requires_grad=True)
    end = torch.tensor(0.17, device=device, dtype=torch.float64, requires_grad=True)

    def evaluate(scale):
        trajectory = TorchToppraTrajectory(
            p,
            v * scale,
            (a * scale) * scale,
            start_path_velocity=start * scale,
            end_path_velocity=end * scale,
            grid_size=17,
        )
        times, q, dq, ddq = trajectory.sample_uniform(29)
        outputs = times * scale, q, dq / scale, (ddq / scale) / scale
        loss = trajectory.duration * scale + sum(
            value.square().mean() * 0.03 for value in outputs[1:]
        )
        return outputs, torch.autograd.grad(loss, (p, v, a, start, end))

    expected, expected_gradients = evaluate(1.0)
    actual, gradients = evaluate(time_scale)
    for value, target in zip((*actual, *gradients), (*expected, *expected_gradients)):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("power", [-240, -160, 160, 240])
def test_inactive_constraints_do_not_poison_gradients_at_extreme_limit_ratios(
    device, power
):
    p = points(device).requires_grad_()
    v = torch.tensor(
        [0.83, 1.07], device=device, dtype=torch.float64, requires_grad=True
    )
    a = torch.tensor(
        [1.63, 2.19], device=device, dtype=torch.float64, requires_grad=True
    )
    factor = 10.0**power
    actual = TorchToppraTrajectory(p, v, a * factor, grid_size=17)
    # At these ratios one constraint family is inactive. Compare with a
    # well-scaled problem in which that family is also provably inactive.
    if power < 0:
        reference = TorchToppraTrajectory(p, v * 1e6, a, grid_size=17)
        loss = actual.duration * factor**0.5
    else:
        reference = TorchToppraTrajectory(p, v, a * 1e6, grid_size=17)
        loss = actual.duration
    torch.testing.assert_close(loss, reference.duration, atol=1e-12, rtol=1e-12)
    gradients = torch.autograd.grad(loss, (p, v, a))
    expected = torch.autograd.grad(reference.duration, (p, v, a))
    for value, target in zip(gradients, expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "argument",
    ["max_velocity", "max_acceleration", "start_path_velocity", "end_path_velocity"],
)
def test_complex_constraints_are_rejected_before_real_cast(device, argument):
    options = {"max_velocity": 1.0, "max_acceleration": 2.0, "grid_size": 9}
    options[argument] = torch.tensor(1.0 + 2.0j, device=device)
    with pytest.raises(TypeError, match="real"):
        TorchToppraTrajectory(points(device), **options)


@pytest.mark.parametrize("device", DEVICES)
def test_complex_sample_times_and_numpy_limits_are_rejected(device):
    trajectory = TorchToppraTrajectory(points(device), 1.0, 2.0, grid_size=9)
    with pytest.raises(TypeError, match="real"):
        trajectory.sample(torch.tensor([0.2 + 0.3j], device=device))
    with pytest.raises(TypeError, match="real"):
        TorchToppraTrajectory(points(device), np.array([1 + 2j, 2 + 3j]), 2.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dof", [1, 3, 7, 32, 128])
@pytest.mark.parametrize("zero_coefficients", [False, True])
def test_symmetric_projection_matches_general_half_plane_elimination(
    device, dof, zero_coefficients
):
    from holistic_motion.trajectory.toppra import _IntervalConstraints
    from holistic_motion.trajectory.toppra_torch import _projection_caps

    generator = np.random.default_rng(53 + dof)
    aa = generator.normal(size=(2, 5, 3 * dof))
    bb = generator.normal(size=aa.shape)
    if zero_coefficients:
        aa[..., ::4] = 0.0
        bb[..., ::3] = 0.0
    cap = generator.uniform(0.1, 2.0, size=(2, 5, 1))
    left = np.concatenate((aa, -aa, cap, np.zeros_like(cap)), axis=-1)
    right = np.concatenate((bb, -bb, np.zeros_like(cap), cap), axis=-1)
    expected = np.array(
        [
            [_IntervalConstraints(a, b).cap for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(left, right)
        ]
    )
    actual = _projection_caps(
        torch.tensor(left, device=device), torch.tensor(right, device=device)
    )
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_symmetric_projection_bound_gradients(device):
    from holistic_motion.trajectory.toppra_torch import _projection_caps

    generator = torch.Generator().manual_seed(83)
    aa = (
        torch.randn(2, 3, 6, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )
    bb = (
        torch.randn(2, 3, 6, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )
    cap = (
        torch.rand(2, 3, 1, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )

    def evaluate(aa, bb, cap):
        left = torch.cat((aa, -aa, cap, torch.zeros_like(cap)), dim=-1)
        right = torch.cat((bb, -bb, torch.zeros_like(cap), cap), dim=-1)
        return _projection_caps(left, right)

    assert torch.autograd.gradcheck(
        evaluate, (aa, bb, cap), nondet_tol=1e-12 if device == "cuda" else 0.0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_inference_context_does_not_affect_later_autograd(device, context):
    p = points(device).requires_grad_()
    with context():
        trajectory = TorchToppraTrajectory(p, 1.0, 2.0, grid_size=17)
        expected = trajectory.sample_uniform(29)
        assert all(not value.requires_grad for value in expected)
    differentiable = TorchToppraTrajectory(p, 1.0, 2.0, grid_size=17)
    actual = differentiable.sample_uniform(29)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target)
    differentiable.duration.backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_terminal_speeds_match_independent_values_and_gradients(device):
    p = points(device, batch=True).requires_grad_()
    v = torch.tensor(
        [[0.8, 1.1], [0.9, 0.7]], dtype=torch.float64, device=device, requires_grad=True
    )
    end = torch.tensor(
        [0.0, 0.17], dtype=torch.float64, device=device, requires_grad=True
    )
    batch = TorchToppraTrajectory(p, v, 2.0, end_path_velocity=end, grid_size=17)
    for index in range(2):
        single = TorchToppraTrajectory(
            p[index], v[index], 2.0, end_path_velocity=end[index], grid_size=17
        )
        torch.testing.assert_close(batch.duration[index], single.duration)
        for actual, expected in zip(
            batch.sample_uniform(29), single.sample_uniform(29)
        ):
            torch.testing.assert_close(actual[index], expected)
        actual = torch.autograd.grad(
            batch.duration[index], (p, v, end), retain_graph=True
        )
        expected = torch.autograd.grad(single.duration, (p, v, end))
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
            assert torch.count_nonzero(value[1 - index]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("units", ["space", "time"])
def test_independent_units_preserve_batch_values_and_gradients(device, units):
    p = points(device).requires_grad_()
    v = torch.tensor(
        [0.83, 1.07], device=device, dtype=torch.float64, requires_grad=True
    )
    a = torch.tensor(
        [1.63, 2.19], device=device, dtype=torch.float64, requires_grad=True
    )
    end = torch.tensor(
        [0.0, 0.17, 0.13, 0.0], device=device, dtype=torch.float64, requires_grad=True
    )
    scales = torch.tensor(
        [1e-10, 1.0, 1e155, 1e300] if units == "space" else [1e-120, 1.0, 1e60, 1e120],
        device=device,
        dtype=torch.float64,
    )

    def evaluate(scale):
        space = scale if units == "space" else torch.ones_like(scale)
        time = scale if units == "time" else torch.ones_like(scale)
        trajectory = TorchToppraTrajectory(
            p * space[:, None, None],
            v * scale[:, None],
            (a * scale[:, None]) * time[:, None],
            end_path_velocity=end * time,
            grid_size=17,
        )
        times, q, dq, ddq = trajectory.sample_uniform(29)
        outputs = (
            times * time[:, None],
            q / space[:, None, None],
            dq / scale[:, None, None],
            (ddq / scale[:, None, None]) / time[:, None, None],
        )
        loss = (trajectory.duration * time).sum() + sum(
            value.square().mean() * 0.003 for value in outputs[1:]
        )
        return outputs, torch.autograd.grad(loss, (p, v, a, end))

    expected, expected_gradients = evaluate(torch.ones_like(scales))
    actual, gradients = evaluate(scales)
    for value, target in zip((*actual, *gradients), (*expected, *expected_gradients)):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_unrepresentable_chord_lengths_are_rejected(device):
    p = torch.tensor([[-1e308], [1e308]], device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite chord lengths"):
        TorchToppraTrajectory(p, 1.0, 2.0)


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_preserves_float64_timing_and_input_gradients(device):
    p = points(device, dtype=torch.float32).requires_grad_()
    expected = TorchToppraTrajectory(p, 1.0, 2.0, grid_size=13)
    expected_gradient = torch.autograd.grad(expected.duration, p)[0]
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device, dtype=dtype):
        actual = TorchToppraTrajectory(p, 1.0, 2.0, grid_size=13)
        times, q, dq, ddq = actual.sample_uniform(29)
    assert times.dtype == actual.duration.dtype == torch.float64
    assert all(value.dtype == torch.float32 for value in (q, dq, ddq))
    torch.testing.assert_close(actual.duration, expected.duration)
    torch.testing.assert_close(
        torch.autograd.grad(actual.duration, p)[0], expected_gradient
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_waypoints_propagate_gradients_to_cpu_limit_tensors():
    p = points("cuda").requires_grad_()
    v = torch.tensor([0.8, 1.1], dtype=torch.float32, requires_grad=True)
    a = torch.tensor([1.7, 2.1], dtype=torch.float64, requires_grad=True)
    end = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    actual = TorchToppraTrajectory(p, v, a, end_path_velocity=end, grid_size=17)
    actual_gradients = torch.autograd.grad(actual.duration, (p, v, a, end))
    cpu = p.detach().cpu().requires_grad_()
    expected = TorchToppraTrajectory(cpu, v, a, end_path_velocity=end, grid_size=17)
    expected_gradients = torch.autograd.grad(expected.duration, (cpu, v, a, end))
    torch.testing.assert_close(actual.duration.cpu(), expected.duration)
    for actual_gradient, expected_gradient, source in zip(
        actual_gradients, expected_gradients, (p, v, a, end)
    ):
        assert actual_gradient.device == source.device
        assert actual_gradient.dtype == source.dtype
        torch.testing.assert_close(actual_gradient.cpu(), expected_gradient)


@pytest.mark.parametrize("device", DEVICES)
def test_boundary_only_gradients_with_constant_path_and_limits(device):
    p = points(device)
    start = torch.tensor(0.12, device=device, dtype=torch.float64, requires_grad=True)
    end = torch.tensor(0.17, device=device, dtype=torch.float64, requires_grad=True)

    def evaluate(start, end):
        trajectory = TorchToppraTrajectory(
            p, 1.0, 2.0, start_path_velocity=start, end_path_velocity=end, grid_size=8
        )
        return (trajectory.duration, *trajectory.sample_uniform(7)[1:])

    assert torch.autograd.gradcheck(evaluate, (start, end), atol=2e-5, rtol=2e-4)
    end = torch.zeros((), device=device, dtype=torch.float64, requires_grad=True)
    duration = evaluate(start, end)[0]
    start_gradient, end_gradient = torch.autograd.grad(duration, (start, end))
    assert torch.isfinite(start_gradient)
    assert end_gradient == 0


@pytest.mark.parametrize("device", DEVICES)
def test_projected_stop_has_zero_speed_and_stable_gradients_across_units(device):
    p = torch.tensor(
        [
            0.311,
            0.032,
            -0.279,
            0.247,
            1.865,
            -0.702,
            0.206,
            0.602,
            -0.038,
            0.848,
            0.887,
            1.02,
        ],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    results = []
    for scale in (1.0, 1e-100, 1e100):
        trajectory = TorchToppraTrajectory(
            p[:, None],
            scale,
            2 * scale**2,
            start_path_velocity=0.002 * scale,
            grid_size=9,
        )
        assert trajectory.result.path_speeds[8] == 0.0
        duration = trajectory.duration * scale
        assert duration.item() == pytest.approx(16.6798165269817138, rel=1e-13)
        results.append((duration, torch.autograd.grad(duration, p)[0]))
    for actual in results[1:]:
        for value, target in zip(actual, results[0]):
            torch.testing.assert_close(value, target, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_coarse_grid_with_adjacent_stops_requests_refinement(device):
    rng = np.random.default_rng(63054)
    p = torch.tensor(rng.normal(size=(16, 1)), device=device, dtype=torch.float64)
    velocity = rng.uniform(0.3, 2.0, 1)
    acceleration = rng.uniform(0.4, 3.0, 1)
    with pytest.raises(ValueError, match="zero reachable speed.*increase grid_size"):
        TorchToppraTrajectory(p, velocity, acceleration, grid_size=3)
    trajectory = TorchToppraTrajectory(p, velocity, acceleration, grid_size=61)
    assert 0.0 < trajectory.duration < 100.0
    _, _, dq, ddq = trajectory.sample_uniform(1001)
    assert torch.all(dq.abs() <= torch.as_tensor(velocity, device=device) + 1e-10)
    assert torch.all(ddq.abs() <= torch.as_tensor(acceleration, device=device) + 1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_long_spline_gradient_matches_directional_finite_difference(device):
    generator = torch.Generator().manual_seed(1049)
    p = torch.randn(2, 128, 7, generator=generator, dtype=torch.float64).to(device)
    direction = torch.randn(p.shape, generator=generator, dtype=p.dtype).to(device)
    p.requires_grad_()

    def objective(value):
        trajectory = TorchToppraTrajectory(value, 1.0, 2.0, grid_size=255)
        return trajectory.duration.sum()

    gradient = torch.autograd.grad(objective(p), p)[0]
    analytic = (gradient * direction).sum()
    with torch.no_grad():
        eps = 1e-6
        numerical = (
            objective(p + eps * direction) - objective(p - eps * direction)
        ) / (2 * eps)
    torch.testing.assert_close(analytic, numerical, atol=1e-5, rtol=1e-5)
