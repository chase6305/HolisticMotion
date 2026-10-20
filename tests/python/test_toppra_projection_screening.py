"""Screening preserves the complete symmetric-strip projection, including ties."""

import numpy as np
import pytest
from holistic_motion.trajectory import ToppraTrajectory, toppra


def _complete_caps(a, b):
    """Evaluate every distinct strip pair without screening or row chunks."""

    count = (a.shape[1] - 2) // 2
    left, right = np.triu_indices(count, k=1)
    aa, bb = a[:, :count], b[:, :count]
    determinant = np.abs(aa[:, left] * bb[:, right] - bb[:, left] * aa[:, right])
    valid = (bb[:, left] != 0.0) & (bb[:, right] != 0.0) & (determinant > 0.0)
    pair = np.divide(
        np.abs(bb[:, left]) + np.abs(bb[:, right]),
        determinant,
        out=np.full_like(determinant, np.inf),
        where=valid,
    )
    velocity = b[:, -1:]
    divisor = np.abs(aa) * velocity
    valid = (((aa < 0.0) & (bb > 0.0)) | ((aa > 0.0) & (bb < 0.0))) & (divisor > 0.0)
    bound = np.divide(
        np.abs(bb) + velocity,
        divisor,
        out=np.full_like(divisor, np.inf),
        where=valid,
    )
    return np.minimum(
        np.min(pair, axis=1, initial=np.inf),
        np.min(bound, axis=1, initial=np.inf),
    ).astype(float)


def _half_planes(aa, bb, velocity):
    zeros = np.zeros_like(velocity)
    return (
        np.concatenate((aa, -aa, velocity, zeros), axis=1),
        np.concatenate((bb, -bb, zeros, velocity), axis=1),
    )


@pytest.mark.parametrize("count", [47, 48, 96, 384])
@pytest.mark.parametrize("pattern", range(13))
@pytest.mark.parametrize("strided", [False, True])
def test_screening_preserves_every_pair_minimum(count, pattern, strided):
    generator = np.random.default_rng(72193 + pattern)
    aa = generator.normal(size=(7, count))
    bb = generator.normal(size=aa.shape)
    if pattern == 1:
        aa *= 10.0 ** generator.uniform(-90.0, 90.0, aa.shape)
        bb *= 10.0 ** generator.uniform(-90.0, 90.0, bb.shape)
    elif pattern == 2:
        aa *= 1e-101  # The original product range is valid, outside screening.
        bb *= 1e-101
    elif pattern == 3:
        aa *= 1e101
        bb *= 1e101
    elif pattern == 4:
        aa = bb * generator.normal(size=(7, 1))
    elif pattern == 5:
        aa = bb * generator.normal(size=(7, 1)) + 1e-14 * aa
    elif pattern == 6:
        aa[:, 1:] = aa[:, :1]
        bb[:, 1:] = bb[:, :1]
    elif pattern == 7:
        aa[:, ::3] = 0.0
        bb[:, ::2] = -0.0
    elif pattern == 8:
        aa[0] *= 1e-101
        bb[1] *= 1e101
        aa[2] = 0.0
        bb[3] = 0.0
    elif pattern == 9:
        aa = np.round(aa, 1)
        bb = np.round(bb, 1)
    elif pattern == 10:
        aa[:, :] = 1.0 - 128.0 * np.finfo(float).eps
        aa[:, ::2] *= -1.0
        aa[:, :2] = [1.0, -1.0]
        bb[:, :] = 1.0
    elif pattern == 11:
        # A large center makes the residual error bound deliberately coarse.
        bb = generator.uniform(0.1, 2.0, bb.shape)
        aa = -1e12 * bb + aa
    elif pattern == 12:
        aa = -bb
    velocity = generator.uniform(0.1, 2.0, (7, 1))
    a, b = _half_planes(aa, bb, velocity)
    if strided:
        storage_a = np.empty((a.shape[0], 2 * a.shape[1]))
        storage_b = np.empty_like(storage_a)
        storage_a[:, ::2], storage_b[:, ::2] = a, b
        a, b = storage_a[::-2, ::2], storage_b[::-2, ::2]
        assert not a.flags.c_contiguous
        assert a.strides[0] < 0 and a.strides[1] > a.itemsize
    before = (a.tobytes(), b.tobytes())
    expected = _complete_caps(a, b)
    actual = toppra._projection_caps(a, b)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert (a.tobytes(), b.tobytes()) == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_other_coefficient_dtypes_keep_complete_projection(dtype, scale):
    generator = np.random.default_rng(719)
    aa = (scale * generator.normal(size=(5, 48))).astype(dtype)
    bb = (scale * generator.normal(size=aa.shape)).astype(dtype)
    aa[:, ::3], bb[:, ::4] = 0.0, -0.0
    a, b = _half_planes(aa, bb, np.full((5, 1), scale, dtype=dtype))
    assert toppra._projection_caps(a, b).tobytes() == _complete_caps(a, b).tobytes()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_large_projection_rejects_nonfinite_coefficients(invalid):
    a, b = _half_planes(np.ones((2, 48)), np.ones((2, 48)), np.ones((2, 1)))
    b[1, -3] = invalid
    with pytest.raises(ValueError, match="constraints must be finite"):
        toppra._projection_caps(a, b)


@pytest.mark.parametrize("count", [47, 48, 384])
def test_projection_accepts_empty_row_batches(count):
    a = np.empty((0, 2 * count + 2))
    assert toppra._projection_caps(a, a.copy()).shape == (0,)


@pytest.mark.parametrize("scale", [1e-200, 1e-101, 1.0, 1e101, 1e155])
@pytest.mark.parametrize("zero_columns", [False, True])
def test_exact_difference_strips_preserve_underflow_and_overflow(scale, zero_columns):
    generator = np.random.default_rng(7231)
    bb = scale * generator.uniform(-1.0, 1.0, (3, 48))
    if zero_columns:
        bb[:, ::3] = -0.0
    a, b = _half_planes(-bb, bb, np.ones((3, 1)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = _complete_caps(a, b)
        actual = toppra._projection_caps(a, b)
    assert actual.tobytes() == expected.tobytes()


def test_negligible_pilot_products_preserve_caller_floating_error_policy():
    aa = np.zeros((1, 48))
    bb = np.full_like(aa, 1e100)
    aa[0, 1], bb[0, 0] = 1e-100, 1e-100
    a, b = _half_planes(aa, bb, np.ones((1, 1)))
    with np.errstate(all="raise"):
        expected = _complete_caps(a, b)
        actual = toppra._projection_caps(a, b)
        assert actual.tobytes() == expected.tobytes()
        assert all(policy == "raise" for policy in np.geterr().values())


def test_complete_fallback_preserves_underflow_exceptions():
    generator = np.random.default_rng(719)
    aa = 1e-200 * generator.normal(size=(1, 48))
    bb = 1e-200 * generator.normal(size=aa.shape)
    a, b = _half_planes(aa, bb, np.ones((1, 1)))
    with np.errstate(under="raise"), pytest.raises(FloatingPointError):
        toppra._projection_caps(a, b)


@pytest.mark.parametrize("unique_count", [1, 2, 3, 8, 9, 17])
@pytest.mark.parametrize("perturb", [False, True])
def test_exact_candidate_duplicates_preserve_distinct_and_nearby_normals(
    unique_count, perturb
):
    generator = np.random.default_rng(19121)
    template = np.arange(96) % unique_count
    aa = np.where(template % 2, -1.0, 1.0)
    bb = (template + 1).astype(float)
    permutation = generator.permutation(96)
    signs = generator.choice([-1.0, 1.0], size=96)
    aa, bb = (aa[permutation] * signs)[None, :], (bb[permutation] * signs)[None, :]
    if perturb:
        aa[0, 0] = np.nextafter(aa[0, 0], np.inf)
    a, b = _half_planes(aa, bb, np.ones((1, 1)))
    with np.errstate(all="raise"):
        expected = _complete_caps(a, b)
        actual = toppra._projection_caps(a, b)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dof,active", [(16, 16), (32, 16), (64, 64)])
@pytest.mark.parametrize("coordinate,clock", [(1.0, 1.0), (1e200, 1e-80), (1e-6, 1e6)])
@pytest.mark.parametrize("grid_kind", ["uniform", "clustered", "default"])
@pytest.mark.parametrize("end_speed", [0.0, 1e-6])
def test_screened_trajectories_preserve_profiles_and_motion(
    monkeypatch, dof, active, coordinate, clock, grid_kind, end_speed
):
    generator = np.random.default_rng(19371)
    points = generator.normal(size=(12, dof))
    if active < dof:
        points[:, active:] = points[0, active:]
    velocity = generator.uniform(0.8, 1.5, dof)
    acceleration = generator.uniform(2.0, 4.0, dof)
    options = {
        "start_path_velocity": 1e-6 * clock,
        "end_path_velocity": end_speed * clock,
        "grid_size": 65,
    }
    if grid_kind == "uniform":
        options["gridpoints"] = np.linspace(0.0, 1.0, 65)
    elif grid_kind == "clustered":
        options["gridpoints"] = np.r_[
            0.0, np.geomspace(1e-7, 0.49, 24), 1.0 - np.geomspace(0.49, 1e-7, 24), 1.0
        ]
    arguments = (
        points * coordinate,
        (velocity * coordinate) * clock,
        ((acceleration * coordinate) * clock) * clock,
    )
    actual = ToppraTrajectory(*arguments, **options)
    with monkeypatch.context() as context:
        context.setattr(toppra, "_projection_caps", _complete_caps)
        reference = ToppraTrajectory(*arguments, **options)
    for name in ("gridpoints", "path_speeds", "path_accelerations", "times"):
        assert (
            getattr(actual.result, name).tobytes()
            == getattr(reference.result, name).tobytes()
        )
    times = np.r_[
        actual.result.times, actual.duration * generator.uniform(0.0, 1.0, 37)
    ]
    for value, expected in zip(actual.sample(times), reference.sample(times)):
        assert value.tobytes() == expected.tobytes()
