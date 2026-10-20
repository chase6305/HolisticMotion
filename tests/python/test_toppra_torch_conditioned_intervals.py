"""Reject float64 timing that cannot certify physical acceleration limits."""

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
UNITS = [(1.0, 1.0), (1e100, 1e-100), (1e200, 1e40)]


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def points(device, delta, batch):
    value = torch.tensor(
        [[0.0], [0.2], [0.2 + delta], [1.0]], dtype=torch.float64, device=device
    )
    return torch.stack((value, value + 0.1)) if batch else value


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("delta", [1e-8, 1e-10, 2e-12])
@pytest.mark.parametrize("grid_size", [17, 65, 200])
@pytest.mark.parametrize("coordinate,clock", UNITS)
@pytest.mark.parametrize("batch", [False, True])
def test_short_chord_acceleration_violation_is_rejected(
    device, delta, grid_size, coordinate, clock, batch
):
    # LP residuals subtract large, nearly equal terms. The previous relative
    # LP tolerance admitted physical acceleration overshoots above 2 percent.
    with pytest.raises(ValueError, match="acceleration.*numerically resolvable"):
        TorchToppraTrajectory(
            points(device, delta, batch) * coordinate,
            10.0 * coordinate * clock,
            (coordinate * clock) * clock,
            grid_size=grid_size,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("delta", [1e-8, 1e-10, 2e-12])
@pytest.mark.parametrize("grid_size", [65, 200])
@pytest.mark.parametrize("coordinate,clock", [(1.0, 1.0), (1e200, 1e40)])
@pytest.mark.parametrize("batch", [False, True])
def test_certifiable_short_chord_coasting_retains_limits_and_gradients(
    device, delta, grid_size, coordinate, clock, batch
):
    p = points(device, delta, batch).requires_grad_()
    v = torch.tensor(0.2, dtype=torch.float64, device=device, requires_grad=True)
    a = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)
    trajectory = TorchToppraTrajectory(
        p * coordinate,
        (v * coordinate) * clock,
        ((a * coordinate) * clock) * clock,
        grid_size=grid_size,
    )
    node_times = trajectory.result.times
    times = node_times[..., :-1] + 0.5 * torch.diff(node_times, dim=-1)
    q, dq, ddq = trajectory.sample(times)
    assert bool((dq.abs() / coordinate / clock <= v + 1e-10).all())
    assert bool((ddq.abs() / coordinate / clock / clock <= a + 1e-8).all())
    loss = (trajectory.duration * clock).sum() + 0.001 * (
        q / coordinate
    ).square().mean()
    for gradient in torch.autograd.grad(loss, (p, v, a)):
        assert torch.isfinite(gradient).all()
    assert torch.equal(
        trajectory.result.path_speeds[..., 0], torch.zeros_like(trajectory.duration)
    )
    assert torch.equal(
        trajectory.result.path_speeds[..., -1], torch.zeros_like(trajectory.duration)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid_size", [65, 200])
@pytest.mark.parametrize("batch", [False, True])
def test_certifiable_small_clock_coasting(device, grid_size, batch):
    test_certifiable_short_chord_coasting_retains_limits_and_gradients(
        device, 1e-8, grid_size, 1e100, 1e-100, batch
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("delta", [1e-10, 2e-12])
@pytest.mark.parametrize("grid_size", [65, 200])
@pytest.mark.parametrize("batch", [False, True])
def test_small_clock_coasting_with_acceleration_roundoff_is_rejected(
    device, delta, grid_size, batch
):
    # Independent continuous extrema put the former returned profile above
    # 1.000008 and 1.000395 times the acceleration limit for these fixtures.
    with pytest.raises(ValueError, match="acceleration.*numerically resolvable"):
        TorchToppraTrajectory(
            points(device, delta, batch) * 1e100,
            0.2,
            1e-100,
            grid_size=grid_size,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
def test_high_speed_certifiable_coasting_preserves_boundary_speeds(device, batch):
    p = torch.linspace(0, 1, 5, dtype=torch.float64, device=device)[:, None]
    if batch:
        p = torch.stack((p, p + 0.5))
    trajectory = TorchToppraTrajectory(
        p,
        1024.0,
        1.0,
        start_path_velocity=1024.0,
        end_path_velocity=1024.0,
        grid_size=17,
    )
    _, _, velocity, acceleration = trajectory.sample_uniform(101)
    assert bool((velocity.abs() <= 1024.0 * (1 + 1e-12)).all())
    assert bool((acceleration.abs() <= 1 + 1e-8).all())
    torch.testing.assert_close(
        trajectory.result.path_speeds[..., 0],
        torch.full_like(trajectory.duration, 1024.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        trajectory.result.path_speeds[..., -1],
        torch.full_like(trajectory.duration, 1024.0),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_small_certifiable_roundoff_is_accepted(device):
    trajectory = TorchToppraTrajectory(
        points(device, 1e-6, False), 10.0, 1.0, grid_size=17
    )
    times = trajectory.result.times
    _, _, acceleration = trajectory.sample(times[:-1] + 0.5 * torch.diff(times))
    assert bool((acceleration.abs() <= 1 + 1e-8).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("boundary", [1e3, 1e6])
@pytest.mark.parametrize("grid_size", [17, 65])
def test_large_boundary_speed_acceleration_cancellation_is_rejected(
    device, boundary, grid_size
):
    p = torch.tensor([[0.0], [0.2], [0.6], [1.0]], dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match="acceleration.*numerically resolvable"):
        TorchToppraTrajectory(
            p,
            1e8,
            1.0,
            start_path_velocity=boundary,
            end_path_velocity=boundary,
            grid_size=grid_size,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_overflowed_normalized_velocity_limit_is_checked(device, requires_grad):
    p = torch.tensor(
        [[0.0], [1.0]], dtype=torch.float64, device=device, requires_grad=requires_grad
    )
    # v / speed_scale overflows even though both physical limits are finite.
    # The old interval matrix silently dropped this velocity constraint.
    with pytest.raises(ValueError, match="velocity.*numerically resolvable"):
        TorchToppraTrajectory(
            p * 1e308,
            1e307,
            1e304,
            grid_size=3,
            start_path_velocity=1.0,
            end_path_velocity=1.0,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_certifiable_overflowed_limit_units_remain_supported(device, requires_grad):
    p = torch.tensor(
        [[0.0], [1.0]], dtype=torch.float64, device=device, requires_grad=requires_grad
    )
    trajectory = TorchToppraTrajectory(
        p * 1e308,
        1e307,
        1e304,
        grid_size=3,
        start_path_velocity=0.01,
        end_path_velocity=0.01,
    )
    _, q, dq, ddq = trajectory.sample_uniform(3)
    assert torch.isfinite(q).all()
    assert bool((dq.abs() / 1e307 <= 1 + 1e-8).all())
    assert bool((ddq.abs() / 1e304 <= 1 + 1e-8).all())
    if requires_grad:
        assert torch.isfinite(torch.autograd.grad(trajectory.duration, p)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [32, 64])
@pytest.mark.parametrize("coordinate,clock", [(1.0, 1.0), (1e200, 1e40)])
@pytest.mark.parametrize("unsafe_last_path", [False, True])
def test_conditioned_physical_checks_cover_multiple_chunks(
    device, batch, coordinate, clock, unsafe_last_path
):
    dof = 32
    base = points(device, 1e-8, False)[None].repeat(batch, 1, dof)
    base.requires_grad_()
    waypoints = base * coordinate
    saved = waypoints.detach().clone()
    version = waypoints._version
    velocity = torch.full((batch, dof), 0.2, dtype=torch.float64, device=device)
    if unsafe_last_path:
        velocity[-1] = 10.0
    # 201 intervals exceed both chunk sizes; the final chunk is also partial.
    assert 201 > (1 << 17) // (batch * dof)
    arguments = (
        waypoints,
        (velocity * coordinate) * clock,
        (coordinate * clock) * clock,
    )
    if unsafe_last_path:
        with pytest.raises(ValueError, match="acceleration.*numerically resolvable"):
            TorchToppraTrajectory(*arguments, grid_size=200)
    else:
        trajectory = TorchToppraTrajectory(*arguments, grid_size=200)
        assert trajectory._physical_validation_needed
        node_times = trajectory.result.times
        midpoint = node_times[..., :-1] + 0.5 * torch.diff(node_times, dim=-1)
        q, dq, ddq = trajectory.sample(midpoint)
        assert bool((dq.abs() / coordinate / clock <= velocity[:, None] + 1e-10).all())
        assert bool((ddq.abs() / coordinate / clock / clock <= 1.0 + 1e-8).all())
        loss = (trajectory.duration * clock).mean() + (q / coordinate).square().mean()
        assert torch.isfinite(torch.autograd.grad(loss, base)[0]).all()
    assert waypoints._version == version
    assert torch.equal(waypoints, saved)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [32, 64])
def test_wide_short_chord_coasting_with_physical_overshoot_is_rejected(device, batch):
    # The 32-axis chord normalization rounds differently from the 1-axis
    # fixture. Independent continuous extrema exceed the acceleration limit
    # by 2.3e-5 relative, even with the small coasting velocity cap.
    waypoints = points(device, 1e-10, False)[None].repeat(batch, 1, 32)
    with pytest.raises(ValueError, match="acceleration.*numerically resolvable"):
        TorchToppraTrajectory(waypoints, 0.2, 1.0, grid_size=200)
