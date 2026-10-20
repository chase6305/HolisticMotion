"""Parallel spline elimination, including its implicit higher derivatives."""

import pytest

torch = pytest.importorskip("torch")

from holistic_motion.trajectory.toppra_torch import (
    TorchToppraTrajectory,
    _SplineSolve,
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


def dense_coefficients(h, slopes):
    diagonal = 2.0 * (h[:, :-1] + h[:, 1:])
    off = h[:, 1:-1]
    matrix = (
        torch.diag_embed(diagonal)
        + torch.diag_embed(off, offset=1)
        + torch.diag_embed(off, offset=-1)
    )
    interior = torch.linalg.solve(matrix, 6.0 * torch.diff(slopes, dim=1))
    zero = torch.zeros_like(slopes[:, :1])
    second = torch.cat((zero, interior, zero), dim=1)
    return (
        slopes - h[..., None] * (2.0 * second[:, :-1] + second[:, 1:]) / 6.0,
        second[:, :-1] / 2.0,
        torch.diff(second, dim=1) / (6.0 * h[..., None]),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [15, 16, 17, 31, 32, 33, 64, 129])
@pytest.mark.parametrize("batch,dof", [(1, 1), (3, 7)])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_spline_coefficients_and_adjoints_match_dense_solve(
    device, count, batch, dof, nonuniform
):
    generator = torch.Generator().manual_seed(744031 + count)
    widths = torch.rand(batch, count - 1, generator=generator, dtype=torch.float64)
    widths = (10.0 ** (-8.0 * widths) if nonuniform else 0.5 + widths).to(device)
    h = (widths / widths.sum(1, keepdim=True)).requires_grad_()
    # Exercise strided inputs and several right-hand sides simultaneously.
    storage = torch.randn(
        batch, count - 1, 2 * dof, generator=generator, dtype=torch.float64
    ).to(device)
    slopes = storage[..., ::2].requires_grad_()
    actual = TorchToppraTrajectory._derivative_coefficients(h, slopes)
    expected = dense_coefficients(h, slopes)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=3e-12, atol=2e-12)
    scales = [value.detach().abs().amax().clamp_min(1.0) for value in expected]
    gradients = []
    for coefficients in (actual, expected):
        loss = sum(
            (value / scale).square().mean()
            for scale, value in zip(scales, coefficients)
        )
        gradients.append(torch.autograd.grad(loss, (h, slopes)))
    for value, reference in zip(*gradients):
        torch.testing.assert_close(value, reference, rtol=2e-10, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "requires",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_implicit_spline_solve_first_and_second_derivatives(device, requires):
    widths = torch.linspace(0.3, 1.7, 15, device=device, dtype=torch.float64)[None]
    diagonal = 2.0 * (widths[:, :-1] + widths[:, 1:])
    off = widths[:, 1:-1].clone()
    rhs = torch.linspace(-0.7, 1.3, 28, device=device, dtype=torch.float64).reshape(
        1, 14, 2
    )
    arguments = tuple(
        value.requires_grad_(needed)
        for value, needed in zip((diagonal, off, rhs), requires)
    )
    assert torch.autograd.gradcheck(_SplineSolve.apply, arguments, fast_mode=True)
    assert torch.autograd.gradgradcheck(_SplineSolve.apply, arguments, fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [16, 33, 128])
@pytest.mark.parametrize("coordinate", [1e-100, 1.0, 1e100])
def test_spline_coordinate_units_preserve_values_and_normalized_gradients(
    device, count, coordinate
):
    h = torch.linspace(0.5, 1.5, count - 1, device=device, dtype=torch.float64)[None]
    h = (h / h.sum()).requires_grad_()
    slopes = (
        torch.linspace(-3.0, 8.0, 2 * (count - 1), device=device, dtype=torch.float64)
        .sin()
        .reshape(1, count - 1, 2)
        .requires_grad_()
    )
    results = []
    for scale in (1.0, coordinate):
        coefficients = TorchToppraTrajectory._derivative_coefficients(h, slopes * scale)
        normalized = tuple(value / scale for value in coefficients)
        gradients = torch.autograd.grad(
            sum(
                (value / count**order).square().mean()
                for order, value in enumerate(normalized)
            ),
            (h, slopes),
        )
        results.append((*normalized, *gradients))
    for actual, reference in zip(results[1], results[0]):
        torch.testing.assert_close(actual, reference, rtol=2e-11, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [16, 32, 128])
def test_parallel_spline_retains_linear_saved_tensor_storage(device, count):
    h = torch.full(
        (2, count - 1),
        1.0 / count,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    slopes = (
        torch.linspace(-1, 1, 2 * (count - 1) * 7, dtype=torch.float64, device=device)
        .reshape(2, count - 1, 7)
        .requires_grad_()
    )
    saved_elements = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: (saved_elements.append(value.numel()), value)[1],
        lambda value: value,
    ):
        coefficients = TorchToppraTrajectory._derivative_coefficients(h, slopes)
    # An explicit count of saved elements, not process/GPU peak memory.
    assert sum(saved_elements) <= 12 * slopes.numel()
    assert all(value.shape == slopes.shape for value in coefficients)
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(
            sum(value.square().mean() for value in coefficients), (h, slopes)
        )
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [16, 17, 33])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("end_speed", [0.0, 0.07])
def test_long_spline_trajectory_value_and_gradient(device, count, batch, end_speed):
    from holistic_motion.trajectory import ToppraTrajectory

    s = torch.linspace(0, 1, count, dtype=torch.float64, device=device)
    points = torch.stack((s, 0.08 * (3.0 * s).sin() + 0.04 * s.square()), dim=1)
    if batch:
        points = torch.stack((points, points * 0.7 + 0.2))
    points.requires_grad_()
    velocity = torch.tensor(
        [0.8, 1.2], dtype=torch.float64, device=device, requires_grad=True
    )
    acceleration = torch.tensor(
        [1.8, 2.3], dtype=torch.float64, device=device, requires_grad=True
    )

    def evaluate(p, v, a):
        trajectory = TorchToppraTrajectory(
            p,
            v,
            a,
            grid_size=65,
            start_path_velocity=0.03,
            end_path_velocity=end_speed,
        )
        fractions = torch.tensor(
            [0.137, 0.413, 0.823], dtype=torch.float64, device=device
        )
        return (
            trajectory.duration,
            *trajectory.sample(trajectory.duration[..., None] * fractions),
        )

    assert torch.autograd.gradcheck(
        evaluate,
        (points, velocity, acceleration),
        fast_mode=True,
        atol=2e-5,
        rtol=2e-4,
        nondet_tol=1e-12 if device == "cuda" else 0.0,
    )
    trajectory = TorchToppraTrajectory(
        points,
        velocity,
        acceleration,
        grid_size=65,
        start_path_velocity=0.03,
        end_path_velocity=end_speed,
    )
    times, *motion = trajectory.sample_uniform(1001)
    grids = trajectory.result.gridpoints
    for index, path in enumerate(points if batch else points[None]):
        reference = ToppraTrajectory(
            path.detach().cpu().numpy(),
            velocity.detach().cpu().numpy(),
            acceleration.detach().cpu().numpy(),
            start_path_velocity=0.03,
            end_path_velocity=end_speed,
            gridpoints=(grids[index] if batch else grids).detach().cpu().numpy(),
        )
        expected = reference.sample(
            (times[index] if batch else times).detach().cpu().numpy()
        )
        for value, target in zip(motion, expected):
            torch.testing.assert_close(
                (value[index] if batch else value).detach().cpu(),
                torch.from_numpy(target),
                rtol=1e-7,
                atol=1e-8,
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [16, 33])
def test_spline_hessian_matches_dense_solve(device, count):
    generator = torch.Generator().manual_seed(916440 + count)
    h = (
        (
            (0.5 + torch.rand(1, count - 1, generator=generator, dtype=torch.float64))
            / count
        )
        .to(device)
        .requires_grad_()
    )
    slopes = (
        torch.randn(1, count - 1, 2, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )
    directions = (
        h.detach() * 0.1,
        torch.linspace(
            -0.2, 0.3, slopes.numel(), device=device, dtype=torch.float64
        ).reshape_as(slopes),
    )
    hessians = []
    for solve in (TorchToppraTrajectory._derivative_coefficients, dense_coefficients):
        coefficients = solve(h, slopes)
        loss = sum(
            (value / count**order).square().mean()
            for order, value in enumerate(coefficients)
        )
        gradients = torch.autograd.grad(loss, (h, slopes), create_graph=True)
        directional = sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, directions)
        )
        hessians.append(torch.autograd.grad(directional, (h, slopes)))
    for actual, expected in zip(*hessians):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [16, 33])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_long_spline_preserves_sample_dtype_and_finite_adjoints(device, count, dtype):
    s = torch.linspace(0, 1, count, dtype=torch.float64, device=device)
    points = torch.stack((s, 0.08 * (3.0 * s).sin()), dim=1).to(dtype).requires_grad_()
    trajectory = TorchToppraTrajectory(points, [0.8, 1.2], [1.8, 2.3], grid_size=65)
    reference = TorchToppraTrajectory(
        points.double(), [0.8, 1.2], [1.8, 2.3], grid_size=65
    )
    times, *values = trajectory.sample_uniform(101)
    expected = reference.sample(times)
    assert times.dtype == torch.float64
    for value, correct in zip(values, expected):
        assert value.dtype == dtype and value.device == points.device
        torch.testing.assert_close(value, correct.to(dtype), rtol=0, atol=0)
    gradient = torch.autograd.grad(
        trajectory.duration + values[0].double().square().mean(), points
    )[0]
    assert gradient.dtype == dtype
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch,count,dof", [(32, 16, 64), (128, 34, 256)])
def test_wide_spline_values_adjoints_and_input_ownership(device, batch, count, dof):
    generator = torch.Generator().manual_seed(424001 + count)
    h = (
        (
            (
                0.5
                + torch.rand(batch, count - 1, generator=generator, dtype=torch.float64)
            )
            / count
        )
        .to(device)
        .requires_grad_()
    )
    slopes = (
        torch.randn(batch, count - 1, dof, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )
    snapshots = (h.detach().clone(), slopes.detach().clone())
    actual = TorchToppraTrajectory._derivative_coefficients(h, slopes)
    expected = dense_coefficients(h, slopes)
    gradients = []
    for coefficients in (actual, expected):
        loss = sum(
            (value / count**order).square().mean()
            for order, value in enumerate(coefficients)
        )
        gradients.append(torch.autograd.grad(loss, (h, slopes)))
    for value, correct in zip(actual, expected):
        torch.testing.assert_close(value, correct, rtol=3e-11, atol=2e-9)
    for value, correct in zip(*gradients):
        torch.testing.assert_close(value, correct, rtol=2e-10, atol=1e-10)
    for value, snapshot in zip((h, slopes), snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [15, 31])
@pytest.mark.parametrize("small", [1e-6, 1e-100])
def test_small_solution_components_have_small_backward_error(device, size, small):
    import numpy as np

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("extended-precision reference unavailable")
    widths = np.r_[small, small, np.ones(size - 1)][None]
    diagonal = 2.0 * (widths[:, :-1] + widths[:, 1:])
    off = widths[:, 1:-1]
    truth = np.linspace(0.2, 1.0, size * 2).reshape(1, size, 2)
    truth[:, :2] = 1e-8
    rhs = diagonal[..., None] * truth
    rhs[:, 1:] += off[..., None] * truth[:, :-1]
    rhs[:, :-1] += off[..., None] * truth[:, 1:]
    arguments = tuple(
        torch.tensor(value, dtype=torch.float64, device=device)
        for value in (diagonal, off, rhs)
    )
    solution = _SplineSolve.apply(*arguments).cpu().numpy().astype(np.longdouble)
    # Check the original equations in extended precision, independently of
    # elimination order. A norm over the whole solution hides small components.
    diagonal = diagonal.astype(np.longdouble)
    off = off.astype(np.longdouble)
    rhs = rhs.astype(np.longdouble)
    product = diagonal[..., None] * solution
    scale = np.abs(product) + np.abs(rhs)
    product[:, 1:] += off[..., None] * solution[:, :-1]
    product[:, :-1] += off[..., None] * solution[:, 1:]
    scale[:, 1:] += np.abs(off[..., None] * solution[:, :-1])
    scale[:, :-1] += np.abs(off[..., None] * solution[:, 1:])
    assert np.max(np.abs(product - rhs) / scale) <= 2.0 * np.finfo(np.float64).eps


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "batch,count,dof", [(128, 128, 7), (16, 128, 128), (16, 256, 128)]
)
def test_spline_values_agree_across_gradient_modes(device, batch, count, dof):
    generator = torch.Generator().manual_seed(548020 + count + dof)
    h = (
        (0.5 + torch.rand(batch, count - 1, generator=generator, dtype=torch.float64))
        .to(device)
        .requires_grad_()
    )
    slopes = (
        torch.randn(batch, count - 1, dof, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )
    snapshots = (h.detach().clone(), slopes.detach().clone())
    with torch.no_grad():
        untracked = TorchToppraTrajectory._derivative_coefficients(h, slopes)
    tracked = TorchToppraTrajectory._derivative_coefficients(h, slopes)
    expected = dense_coefficients(h, slopes)
    for actual in (untracked, tracked):
        for value, correct in zip(actual, expected):
            torch.testing.assert_close(value, correct, rtol=3e-11, atol=2e-9)
    gradients = []
    for coefficients in (tracked, expected):
        loss = sum(value.square().mean() for value in coefficients)
        gradients.append(torch.autograd.grad(loss, (h, slopes)))
    for value, correct in zip(*gradients):
        torch.testing.assert_close(value, correct, rtol=2e-10, atol=1e-10)
    for value, snapshot in zip((h, slopes), snapshots):
        assert torch.equal(value, snapshot)
