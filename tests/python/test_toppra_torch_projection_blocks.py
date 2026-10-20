"""Wide CPU projections retain exhaustive choices with bounded pair storage."""

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
        torch.cat((aa, -aa, velocity, zero), -1),
        torch.cat((bb, -bb, zero, velocity), -1),
    )


@pytest.mark.parametrize("count", [2, 3, 47, 383, 384, 724, 725, 1025])
def test_pair_rank_decoding_matches_complete_triangle(count):
    expected = torch.triu_indices(count, count, 1)
    ranks = torch.arange(expected.shape[1])
    actual = torch.stack(module._pair_coordinates(ranks, count))
    assert torch.equal(actual, expected)


def test_pair_rank_decoding_keeps_integer_precision_and_row_boundaries():
    count = 100_003
    left = torch.arange(count - 1).repeat_interleave(3)
    right = torch.stack(
        (torch.arange(1, count), torch.full((count - 1,), count - 1)), -1
    )
    middle = (torch.arange(count - 1) + count) // 2
    right = torch.stack((right[:, 0], middle, right[:, 1]), -1).reshape(-1)
    ranks = left * (2 * count - left - 1) // 2 + right - left - 1
    assert ranks.max() > 2**32
    actual_left, actual_right = module._pair_coordinates(ranks, count)
    assert torch.equal(actual_left, left)
    assert torch.equal(actual_right, right)


def test_screened_wide_projection_avoids_impractical_pair_index_allocation(monkeypatch):
    # A complete triangle would contain over five billion pairs. This exact
    # strip family needs only one velocity candidate and linear input storage.
    count = 100_003
    aa = -torch.ones(1, count, dtype=torch.float64)
    bb = torch.ones_like(aa)
    a, b = strips(aa, bb, torch.ones(1, 1, dtype=torch.float64))
    a.requires_grad_()
    b.requires_grad_()

    def unexpected(*args, **kwargs):
        raise AssertionError("screened projection attempted a pair index allocation")

    monkeypatch.setattr(torch, "triu_indices", unexpected)
    cap = module._projection_caps(a, b)
    assert cap.item() == 2.0
    da, db = torch.autograd.grad(cap.sum(), (a, b))
    expected_a, expected_b = torch.zeros_like(a), torch.zeros_like(b)
    expected_a[0, 0] = 2.0
    expected_b[0, 0], expected_b[0, -1] = 1.0, -1.0
    assert torch.equal(da, expected_a)
    assert torch.equal(db, expected_b)


@pytest.mark.parametrize("count", [384, 724, 725, 1025])
@pytest.mark.parametrize("pattern", range(10))
@pytest.mark.parametrize("strided", [False, True])
def test_blocked_enumeration_preserves_first_minimum(count, pattern, strided):
    generator = torch.Generator().manual_seed(36171 + pattern)
    aa = torch.randn(2, count, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    velocity = torch.full((2, 1), 0.7, dtype=torch.float64)
    if pattern == 1:
        aa.fill_(1.0)
        aa[:, 1::2] = -1.0
        bb.fill_(1.0)
    elif pattern == 2:
        aa *= 1e155
        bb *= 1e155
    elif pattern == 3:
        aa *= 1e-155
        bb *= 1e-155
    elif pattern == 4:
        aa = 0.7 * bb + 1e-15 * aa
    elif pattern == 5:
        aa.zero_()
        bb.zero_()
    elif pattern == 6:
        aa[:, ::3] = 0.0
        bb[:, ::2] = -0.0
    elif pattern == 7:
        aa[:, 1:] = aa[:, :1]
        bb[:, 1:] = bb[:, :1]
    elif pattern == 8:
        # Finite inputs can yield NaN velocity candidates through inf / inf.
        # argmin must retain its first NaN, including across pair blocks.
        aa.fill_(-1e308)
        bb.fill_(1e308)
        velocity.fill_(1e308)
    elif pattern == 9:
        aa[0].fill_(1e308)
        aa[0, 1::2] = -1e308
        bb[0].fill_(1e308)
        aa[1].zero_()
        bb[1].zero_()
        aa[1, 0], aa[1, 1], bb[1, :2] = 1.0, -1.0, 1.0
        aa[1, -2], aa[1, -1], bb[1, -2:] = 1e308, -1e308, 1e308
        # An inf/inf pair in the last block must precede all velocity NaNs,
        # and must replace any earlier finite or infinite strip minimum.
        velocity.fill_(1e308)
    if strided:
        storage = torch.empty(2, 2 * count, dtype=torch.float64)
        storage[:, ::2] = aa
        aa = storage[:, ::2]
        storage = torch.empty_like(storage)
        storage[:, ::2] = bb
        bb = storage[:, ::2]
    before = (aa.clone(), bb.clone(), aa._version, bb._version)
    pairs = torch.triu_indices(count, count, 1)
    expected = module._complete_projection_choices(aa, bb, velocity, pairs)
    actual = module._blocked_projection_choices(aa, bb, velocity)
    assert torch.equal(actual, expected)
    screened = module._screened_projection_choices(aa, bb, velocity, None)
    assert torch.equal(screened, expected)
    assert torch.equal(aa, before[0]) and torch.equal(bb, before[1])
    assert (aa._version, bb._version) == before[2:]


@pytest.mark.parametrize("count", [725, 1536])
def test_late_block_winners_and_ties_keep_global_candidate_order(count):
    aa = torch.zeros(4, count, dtype=torch.float64)
    bb = torch.zeros_like(aa)
    aa[:3, -2], aa[:3, -1] = 0.5, -0.5
    bb[:3, -2:] = 1.0
    # The second row ties its last block's optimum with the first pair.
    aa[1, 0], aa[1, 1], bb[1, :2] = 0.5, -0.5, 1.0
    velocity = torch.ones(4, 1, dtype=torch.float64)
    # The third row's first velocity bound rounds to the same value as the
    # last strip pair; every strip pair precedes every velocity candidate.
    velocity[2] = 2.0**54
    aa[3, 5], bb[3, 5] = -1.0, 1.0
    pairs = count * (count - 1) // 2
    expected = torch.tensor([pairs - 1, 0, pairs - 1, pairs + 5])
    actual = module._blocked_projection_choices(aa, bb, velocity)
    assert torch.equal(actual, expected)
    a, b = strips(aa, bb, velocity)
    assert torch.equal(module._projection_caps(a, b), torch.full((4,), 2.0))


@pytest.mark.parametrize("pattern", ["random", "crowded", "outside_guard"])
def test_wide_projection_uses_bounded_pair_blocks(monkeypatch, pattern):
    count = 1536
    generator = torch.Generator().manual_seed(7611)
    aa = torch.randn(2, count, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    if pattern == "crowded":
        aa = torch.where(aa > 0, 1.0, -1.0).to(torch.float64)
        bb.fill_(1.0)
    elif pattern == "outside_guard":
        aa *= 1e101
        bb *= 1e101
    a, b = strips(aa, bb, torch.ones(2, 1, dtype=torch.float64))
    pairs = torch.triu_indices(count, count, 1)
    choice = module._complete_projection_choices(aa, bb, b[:, -1:], pairs)
    pair_count = pairs.shape[1]
    is_pair = choice < pair_count
    lookup = choice.clamp_max(pair_count - 1)
    left = torch.where(is_pair, pairs[0, lookup], choice - pair_count)
    right = torch.where(is_pair, pairs[1, lookup], count)
    padded_a = torch.cat((aa, torch.zeros(2, 1)), 1)
    padded_b = torch.cat((bb, b[:, -1:]), 1)
    rows = torch.arange(2)
    ai, aj = padded_a[rows, left], padded_a[rows, right]
    bi, bj = padded_b[rows, left], padded_b[rows, right]
    determinant = (ai * bj - bi * aj).abs()
    expected = (bi.abs() + bj.abs()) / determinant

    original = torch.triu_indices
    allocations = []

    def bounded(*args, **kwargs):
        result = original(*args, **kwargs)
        allocations.append(result.shape[1])
        assert result.shape[1] <= 1 << 18
        return result

    monkeypatch.setattr(torch, "triu_indices", bounded)
    actual = module._projection_caps(a, b)
    assert torch.equal(actual, expected)
    if pattern == "outside_guard":
        assert len(allocations) > 1


@pytest.mark.parametrize("count", [384, 725])
@pytest.mark.parametrize("pattern", ["random", "duplicate", "near_parallel"])
def test_wide_projection_keeps_values_gradients_and_hessian_products(count, pattern):
    generator = torch.Generator().manual_seed(61509)
    aa = torch.randn(2, count, generator=generator, dtype=torch.float64)
    bb = torch.randn(aa.shape, generator=generator, dtype=torch.float64)
    if pattern == "duplicate":
        indices = (torch.arange(count) // 16)[
            torch.randperm(count, generator=generator)
        ]
        aa, bb = aa[:, indices], bb[:, indices]
    elif pattern == "near_parallel":
        aa = 0.7 * bb + 1e-14 * aa
    a, b = strips(aa, bb, torch.full((2, 1), 0.8, dtype=torch.float64))
    arguments = (a.requires_grad_(), b.requires_grad_())
    directions = tuple(
        torch.randn(v.shape, generator=generator, dtype=torch.float64)
        for v in arguments
    )
    original = module._screened_projection_choices

    def complete(aa, bb, velocity, pairs):
        return module._complete_projection_choices(
            aa, bb, velocity, torch.triu_indices(count, count, 1)
        )

    results = []
    try:
        for operation in (original, complete):
            module._screened_projection_choices = operation
            value = module._projection_caps(*arguments)
            gradient = torch.autograd.grad(
                value.square().sum(), arguments, create_graph=True
            )
            hvp = torch.autograd.grad(
                sum((g * d).sum() for g, d in zip(gradient, directions)), arguments
            )
            results.append((value, *gradient, *hvp))
    finally:
        module._screened_projection_choices = original
    for actual, expected in zip(*results):
        assert torch.equal(actual, expected)
