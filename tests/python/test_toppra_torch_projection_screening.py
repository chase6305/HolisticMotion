"""CPU projection screening preserves exhaustive choices and their adjoints."""

import pytest

torch = pytest.importorskip("torch")

from holistic_motion.trajectory import toppra_torch as module


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def strips(aa, bb, velocity):
    zero = torch.zeros_like(velocity)
    return (
        torch.cat((aa, -aa, velocity, zero), dim=-1),
        torch.cat((bb, -bb, zero, velocity), dim=-1),
    )


def exhaustive_candidates(a, b):
    """Directly enumerate every unordered strip pair and velocity bound."""
    count = (a.shape[-1] - 2) // 2
    left, right = torch.triu_indices(count, count, 1, device=a.device)
    aa, bb = a[..., :count], b[..., :count]
    bi, bj = bb[..., left], bb[..., right]
    determinant = (aa[..., left] * bj - bi * aa[..., right]).abs()
    valid = (bi != 0) & (bj != 0) & (determinant > 0)
    bounds = torch.where(
        valid,
        (bi.abs() + bj.abs()) / torch.where(valid, determinant, 1.0),
        torch.inf,
    )
    velocity = b[..., -1:]
    divisor = aa.abs() * velocity
    valid = (((aa < 0) & (bb > 0)) | ((aa > 0) & (bb < 0))) & (divisor > 0)
    velocity_bounds = torch.where(
        valid,
        (bb.abs() + velocity) / torch.where(valid, divisor, 1.0),
        torch.inf,
    )
    return torch.cat((bounds, velocity_bounds), dim=-1)


@pytest.mark.parametrize("count", [47, 48, 96, 192, 384])
@pytest.mark.parametrize("pattern", range(17))
@pytest.mark.parametrize("strided", [False, True])
def test_screening_keeps_first_exhaustive_minimum(count, pattern, strided):
    generator = torch.Generator().manual_seed(129731 + pattern)
    aa = torch.randn(2, 3, count, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    if pattern == 1:
        aa *= 10.0 ** (
            180 * torch.rand(aa.shape, generator=generator, dtype=torch.float64) - 90
        )
        bb *= 10.0 ** (
            180 * torch.rand(bb.shape, generator=generator, dtype=torch.float64) - 90
        )
    elif pattern in (2, 3, 4, 5):
        scale = {2: 1e-101, 3: 1e101, 4: 1e-200, 5: 1e155}[pattern]
        aa *= scale
        bb *= scale
    elif pattern in (6, 7):
        aa = bb * 0.7 + (1e-14 * aa if pattern == 7 else 0.0)
    elif pattern == 8:
        aa[..., 1:] = aa[..., :1]
        bb[..., 1:] = bb[..., :1]
    elif pattern == 9:
        aa[..., ::3] = 0.0
        bb[..., ::2] = -0.0
    elif pattern == 10:
        aa[0, 0] *= 1e-101
        bb[0, 1] *= 1e101
        aa[1, 0] = 0.0
        bb[1, 1] = 0.0
    elif pattern == 11:
        aa = aa.round(decimals=1)
        bb = bb.round(decimals=1)
    elif pattern == 12:
        aa.fill_(1.0 - 128 * torch.finfo(torch.float64).eps)
        aa[..., ::2] *= -1
        aa[..., :2] = torch.tensor([1.0, -1.0])
        bb.fill_(1.0)
    elif pattern == 13:
        bb = bb.abs() + 0.1
        aa -= 1e12 * bb
    elif pattern == 14:
        aa = -bb
    elif pattern == 15:
        aa.zero_()
        bb.fill_(1e100)
        aa[..., 1], bb[..., 0] = 1e-100, 1e-100
    elif pattern == 16:
        aa = torch.where(aa > 0, torch.full_like(aa, 1e100), -1e100)
        bb = torch.where(bb > 0, torch.full_like(bb, 1e-100), -1e-100)
    velocity = 0.1 + torch.rand(2, 3, 1, generator=generator, dtype=torch.float64)
    a, b = strips(aa, bb, velocity)
    if strided:
        storage = torch.empty(2, 3, 2 * a.shape[-1], dtype=torch.float64)
        storage[..., ::2] = a
        a = storage[..., ::2]
        storage = torch.empty_like(storage)
        storage[..., ::2] = b
        b = storage[..., ::2]
        assert not a.is_contiguous() and not b.is_contiguous()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    before = (a.clone(), b.clone(), a._version, b._version)
    expected = exhaustive_candidates(a, b)
    pairs = torch.triu_indices(count, count, 1)
    with torch.no_grad():
        choice = module._screened_projection_choices(
            a[..., :count].reshape(-1, count),
            b[..., :count].reshape(-1, count),
            b[..., -1:].reshape(-1, 1),
            pairs,
        )
    assert torch.equal(choice.reshape(2, 3), expected.argmin(-1))
    actual = module._projection_caps(a, b)
    assert torch.equal(actual, expected.min(-1).values)
    assert torch.equal(a, before[0]) and torch.equal(b, before[1])
    assert (a._version, b._version) == before[2:]


@pytest.mark.parametrize("count", [48, 96, 192])
@pytest.mark.parametrize("repeat", [1, 2, 16])
def test_screening_preserves_first_and_second_derivatives(count, repeat):
    generator = torch.Generator().manual_seed(38219)
    aa = torch.randn(2, 3, count, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    if repeat > 1:
        # Sixteen copies force duplicate reduction in the retained set.
        template = (torch.arange(count) // repeat)[
            torch.randperm(count, generator=generator)
        ]
        signs = torch.where(torch.rand(count, generator=generator) > 0.5, 1.0, -1.0)
        aa, bb = aa[..., template] * signs, bb[..., template] * signs
    velocity = torch.full((2, 3, 1), 0.8, dtype=torch.float64)
    arguments = tuple(v.requires_grad_() for v in strips(aa, bb, velocity))
    direction = tuple(torch.randn(v.shape, generator=generator) for v in arguments)
    results = []
    for operation in (
        module._projection_caps,
        lambda a, b: exhaustive_candidates(a, b).min(-1).values,
    ):
        value = operation(*arguments)
        gradient = torch.autograd.grad(
            value.square().sum(), arguments, create_graph=True
        )
        hvp = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(gradient, direction)), arguments
        )
        results.append((value, *gradient, *hvp))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("count", [48, 384, 725])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("retained_counts", [(2, 3, 4), (7, 8, 9), (2, 9)])
def test_mixed_retained_sets_preserve_ties_and_derivatives(
    count, strided, retained_counts
):
    # Distinct oriented normals give exactly tied active strips. Sets of
    # nine exceed the screening limit and use complete enumeration.
    aa = torch.zeros(len(retained_counts), count, dtype=torch.float64)
    bb = torch.arange(1, count + 1, dtype=torch.float64).expand_as(aa)
    for row, retained in enumerate(retained_counts):
        aa[row, :retained] = torch.where(torch.arange(retained) % 2 == 0, 1.0, -1.0)
    velocity = torch.ones(len(retained_counts), 1, dtype=torch.float64)
    arguments = strips(aa, bb, velocity)
    if strided:
        views = []
        for value in arguments:
            storage = torch.empty(value.shape[0], 2 * value.shape[1], dtype=value.dtype)
            storage[:, ::2] = value
            views.append(storage[:, ::2])
        arguments = tuple(views)
    arguments = tuple(value.requires_grad_() for value in arguments)
    before = tuple((value.clone(), value._version) for value in arguments)
    generator = torch.Generator().manual_seed(92817)
    direction = tuple(
        torch.randn(value.shape, generator=generator, dtype=value.dtype)
        for value in arguments
    )
    results = []
    for operation in (
        module._projection_caps,
        lambda a, b: exhaustive_candidates(a, b).min(-1).values,
    ):
        value = operation(*arguments)
        assert torch.equal(value, torch.ones_like(value))
        gradient = torch.autograd.grad(
            value.square().sum(), arguments, create_graph=True
        )
        hvp = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(gradient, direction)), arguments
        )
        results.append((value, *gradient, *hvp))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    for value, (original, version) in zip(arguments, before):
        assert torch.equal(value, original) and value._version == version


@pytest.mark.parametrize("count", [47, 48, 96])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dispatch_uses_screening_only_for_wide_cpu_float64(monkeypatch, count, dtype):
    generator = torch.Generator().manual_seed(812)
    aa = torch.randn(2, 1, count, generator=generator, dtype=dtype)
    bb = torch.randn(aa.shape, generator=generator, dtype=dtype)
    a, b = strips(aa, bb, torch.ones(2, 1, 1, dtype=dtype))
    called = []
    original = module._screened_projection_choices

    def observe(*args):
        called.append(True)
        return original(*args)

    monkeypatch.setattr(module, "_screened_projection_choices", observe)
    module._projection_caps(a, b)
    assert bool(called) == (count >= 48 and dtype == torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_keeps_complete_projection(monkeypatch):
    def unexpected(*args):
        raise AssertionError("CPU screening used on CUDA")

    monkeypatch.setattr(module, "_screened_projection_choices", unexpected)
    aa = torch.randn(2, 3, 96, device="cuda", dtype=torch.float64)
    bb = torch.randn_like(aa)
    a, b = strips(aa, bb, torch.ones(2, 3, 1, device="cuda", dtype=torch.float64))
    assert torch.equal(
        module._projection_caps(a, b), exhaustive_candidates(a, b).min(-1).values
    )


@pytest.mark.parametrize("dof", [16, 32, 64])
@pytest.mark.parametrize("end_speed", [0.0, 1e-5])
@pytest.mark.parametrize("stationary", [False, True])
def test_full_trajectory_profile_samples_and_gradients_are_unchanged(
    monkeypatch, dof, end_speed, stationary
):
    s = torch.linspace(0, 1, 12, dtype=torch.float64)
    axes = torch.arange(dof, dtype=torch.float64)
    q = (
        s[:, None] + 0.07 * torch.sin((1 + axes.remainder(4)) * s[:, None] + 0.3 * axes)
    )[None].repeat(2, 1, 1)
    if stationary:
        q[..., dof // 2 :] = 0.4
    inputs = (
        q.requires_grad_(),
        torch.full((dof,), 2.0, dtype=torch.float64, requires_grad=True),
        torch.full((dof,), 5.0, dtype=torch.float64, requires_grad=True),
    )
    outputs = []
    for complete in (False, True):
        with monkeypatch.context() as context:
            if complete:
                context.setattr(
                    module,
                    "_screened_projection_choices",
                    module._complete_projection_choices,
                )
            trajectory = module.TorchToppraTrajectory(
                *inputs,
                grid_size=31,
                start_path_velocity=1e-5,
                end_path_velocity=end_speed,
            )
            samples = trajectory.sample_uniform(37)
            gradients = torch.autograd.grad(
                sum(v.square().mean() for v in samples[1:]), inputs
            )
            result = trajectory.result
            outputs.append(
                (
                    result.gridpoints,
                    result.path_speeds,
                    result.path_accelerations,
                    result.times,
                    *samples,
                    *gradients,
                )
            )
    for actual, expected in zip(*outputs):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("count", [48, 96, 192])
def test_projection_crosses_screening_and_complete_chunk_boundaries(count):
    rows = (1 << 16) // count + 3
    generator = torch.Generator().manual_seed(9913 + count)
    aa = torch.randn(1, rows, count, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    # Interleave screening with range-guard and crowded-set fallbacks.
    aa[:, ::3] *= 1e-101
    bb[:, ::3] *= 1e-101
    aa[:, 1::3] = -bb[:, 1::3]
    a, b = strips(aa, bb, torch.ones(1, rows, 1, dtype=torch.float64))
    expected = exhaustive_candidates(a, b).min(-1).values
    assert torch.equal(module._projection_caps(a, b), expected)


@pytest.mark.parametrize("unique_count", [1, 2, 3, 8, 9, 17])
@pytest.mark.parametrize("perturb", [False, True])
@pytest.mark.parametrize("scale", [1e-101, 1.0, 1e101])
def test_exact_oriented_duplicates_keep_original_first_pair(
    unique_count, perturb, scale
):
    count = 96
    generator = torch.Generator().manual_seed(81819 + unique_count)
    template = torch.arange(count) % unique_count
    aa = torch.where(template % 2 == 0, 1.0, -1.0).double()
    bb = (template + 1).double()
    permutation = torch.randperm(count, generator=generator)
    signs = torch.where(torch.rand(count, generator=generator) > 0.5, 1.0, -1.0)
    aa, bb = (aa[permutation] * signs)[None], (bb[permutation] * signs)[None]
    if perturb:
        aa[0, 0] = torch.nextafter(
            aa[0, 0], torch.tensor(torch.inf, dtype=torch.float64)
        )
    a, b = strips(aa * scale, bb * scale, torch.ones(1, 1, dtype=torch.float64))
    candidates = exhaustive_candidates(a, b)
    with torch.no_grad():
        choice = module._screened_projection_choices(
            a[:, :count], b[:, :count], b[:, -1:], torch.triu_indices(count, count, 1)
        )
    assert torch.equal(choice, candidates.argmin(-1))
    assert torch.equal(module._projection_caps(a, b), candidates.min(-1).values)


@pytest.mark.parametrize("relation", ["opposite", "equal", "same_oriented"])
@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1.0, 1e100, 1e155])
def test_zero_determinant_shortcuts_preserve_extreme_ranges(relation, scale):
    generator = torch.Generator().manual_seed(9012)
    bb = scale * torch.randn(4, 96, generator=generator, dtype=torch.float64)
    aa = -bb if relation == "opposite" else bb.clone()
    if relation == "same_oriented":
        signs = bb.sign()
        aa, bb = 0.7 * scale * signs, scale * signs
    a, b = strips(aa, bb, torch.ones(4, 1, dtype=torch.float64))
    candidates = exhaustive_candidates(a, b)
    with torch.no_grad():
        choice = module._screened_projection_choices(
            a[:, :96], b[:, :96], b[:, -1:], torch.triu_indices(96, 96, 1)
        )
    assert torch.equal(choice, candidates.argmin(-1))
    assert torch.equal(module._projection_caps(a, b), candidates.min(-1).values)


@pytest.mark.parametrize("ratio", [-1e12, -1.0, -1e-12, 0.0, 0.7])
@pytest.mark.parametrize("perturbation", [0.0, 1e-14, 1e-10])
def test_nearly_proportional_strips_preserve_exact_projection(ratio, perturbation):
    generator = torch.Generator().manual_seed(79021)
    bb = 0.1 + torch.rand(4, 96, generator=generator, dtype=torch.float64)
    aa = ratio * bb + perturbation * torch.randn(
        bb.shape, generator=generator, dtype=torch.float64
    )
    a, b = strips(aa, bb, torch.ones(4, 1, dtype=torch.float64))
    candidates = exhaustive_candidates(a, b)
    with torch.no_grad():
        choice = module._screened_projection_choices(
            aa, bb, b[:, -1:], torch.triu_indices(96, 96, 1)
        )
    assert torch.equal(choice, candidates.argmin(-1))
    assert torch.equal(module._projection_caps(a, b), candidates.min(-1).values)


@pytest.mark.parametrize(
    "a_dtype,b_dtype", [(torch.float64, torch.float32), (torch.float32, torch.float64)]
)
def test_mixed_coefficient_types_preserve_complete_projection(a_dtype, b_dtype):
    generator = torch.Generator().manual_seed(20991)
    aa = torch.randn(4, 96, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    a, b = strips(aa, bb, torch.ones(4, 1, dtype=torch.float64))
    a, b = a.to(a_dtype), b.to(b_dtype)
    assert torch.equal(
        module._projection_caps(a, b), exhaustive_candidates(a, b).min(-1).values
    )
