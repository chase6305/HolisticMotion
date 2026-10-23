"""Discrete projection selection and differentiable tensor bounds."""

from __future__ import annotations

from math import isqrt

import torch

from ._torch_numeric import _divide


def _projection_caps(
    a: torch.Tensor, b: torch.Tensor, *, screened_choices=None
) -> torch.Tensor:
    """Eliminate y using symmetric acceleration strips and its velocity cap.

    The first K columns and the next K columns are opposite half-planes:
    abs(a_i*x + b_i*y) <= 1. Each distinct pair with nonzero b gives
    abs(a_i*b_j - b_i*a_j)*x <= abs(b_i) + abs(b_j). Thus only K*(K-1)/2
    pairs are needed instead of a square over all 2*K+2 half-planes.
    """

    if screened_choices is None:
        screened_choices = _screened_projection_choices
    shape = a.shape[:-1]
    count = (a.shape[-1] - 2) // 2
    flat_a = a[..., :count].reshape(-1, count)
    flat_b = b[..., :count].reshape(-1, count)
    velocity_cap = b[..., -1].reshape(-1, 1)
    screened = (
        a.device.type == "cpu" and a.dtype == b.dtype == torch.float64 and count >= 48
    )
    # Wide screened problems usually need only a few selected pairs. Do not
    # reserve their complete quadratic index table before screening them.
    pairs = (
        None
        if screened and count >= 384
        else torch.triu_indices(count, count, offset=1, device=a.device)
    )
    pair_count = count * (count - 1) // 2
    with torch.no_grad():
        if screened:
            chunk = max(1, (1 << 16) // count)
            choice = torch.cat(
                [
                    screened_choices(
                        flat_a[offset : offset + chunk],
                        flat_b[offset : offset + chunk],
                        velocity_cap[offset : offset + chunk],
                        pairs,
                    )
                    for offset in range(0, flat_a.shape[0], chunk)
                ]
            )
        else:
            choice = _complete_projection_choices(flat_a, flat_b, velocity_cap, pairs)
    is_pair = choice < pair_count
    lookup = choice.clamp_max(pair_count - 1)
    if pairs is None:
        pair_left, pair_right = _pair_coordinates(lookup, count)
    else:
        pair_left, pair_right = pairs[0, lookup], pairs[1, lookup]
    left = torch.where(is_pair, pair_left, choice - pair_count)
    right = torch.where(is_pair, pair_right, count)
    # Append the y velocity half-plane, then evaluate only the selected pair
    # under autograd. Selection remains discrete without a quadratic tape.
    flat_a = torch.cat((flat_a, torch.zeros_like(velocity_cap)), dim=1)
    flat_b = torch.cat((flat_b, velocity_cap), dim=1)
    ai = flat_a.gather(1, left[:, None]).squeeze(1)
    aj = flat_a.gather(1, right[:, None]).squeeze(1)
    bi = flat_b.gather(1, left[:, None]).squeeze(1)
    bj = flat_b.gather(1, right[:, None]).squeeze(1)
    determinant = (ai * bj - bi * aj).abs()
    valid = torch.where(
        is_pair, (bi != 0) & (bj != 0), ((ai < 0) & (bi > 0)) | ((ai > 0) & (bi < 0))
    ) & (determinant > 0)
    cap = _divide(bi.abs() + bj.abs(), torch.where(valid, determinant, 1.0))
    return torch.where(valid, cap, torch.inf).reshape(shape)


def _complete_projection_choices(flat_a, flat_b, velocity_cap, pairs):
    """Choose the first minimum in the complete strip/velocity enumeration."""

    count = flat_a.shape[1]
    if pairs is None:
        return _blocked_projection_choices(flat_a, flat_b, velocity_cap)
    pair_count = pairs.shape[1]
    # Target 2 MiB per candidate array, with at most 4096 intervals per chunk.
    # Large DOF counts can still require more than the target for a single row.
    chunk = max(1, min(4096, (1 << 18) // (pair_count + count)))
    choices = []
    with torch.no_grad():
        for offset in range(0, flat_a.shape[0], chunk):
            aa, bb = flat_a[offset : offset + chunk], flat_b[offset : offset + chunk]
            cap = velocity_cap[offset : offset + chunk]
            ai, aj = aa[:, pairs[0]], aa[:, pairs[1]]
            bi, bj = bb[:, pairs[0]], bb[:, pairs[1]]
            # These are private no-grad scratch arrays. Reuse their storage
            # and release each chunk before allocating the next one.
            determinant = ai * bj
            determinant.sub_(bi * aj).abs_()
            valid = (bi != 0) & (bj != 0) & (determinant > 0)
            bound = bi.abs()
            bound.add_(bj.abs())
            determinant.masked_fill_(~valid, 1.0)
            bound.div_(determinant).masked_fill_(~valid, torch.inf)
            del ai, aj, bi, bj, determinant, valid
            # The positive-y velocity half-plane pairs with the negative-y
            # member of a strip only when a and b have opposite signs.
            divisor = aa.abs() * cap
            velocity_valid = (((aa < 0) & (bb > 0)) | ((aa > 0) & (bb < 0))) & (
                divisor > 0
            )
            velocity_bound = (bb.abs() + cap) / torch.where(
                velocity_valid, divisor, 1.0
            )
            candidates = torch.cat(
                (
                    bound,
                    torch.where(velocity_valid, velocity_bound, torch.inf),
                ),
                dim=1,
            )
            choices.append(candidates.argmin(dim=1))
            del bound, divisor, velocity_valid, velocity_bound, candidates
    return torch.cat(choices)


def _pair_coordinates(indices, count):
    """Decode lexicographic upper-triangle ranks using integer row starts."""

    rows = torch.arange(count - 1, device=indices.device)
    starts = rows * (2 * count - rows - 1) // 2
    left = torch.searchsorted(starts, indices, right=True) - 1
    right = indices - starts[left] + left + 1
    return left, right


@torch.no_grad()
def _blocked_projection_choices(flat_a, flat_b, velocity_cap):
    """Enumerate all pairs with bounded scratch and the original first tie."""

    count = flat_a.shape[1]
    pair_count = count * (count - 1) // 2
    target = 1 << 18
    best = torch.full(
        (flat_a.shape[0],), torch.inf, dtype=flat_a.dtype, device=flat_a.device
    )
    choice = torch.zeros(flat_a.shape[0], dtype=torch.int64, device=flat_a.device)
    first = 0
    while first < count - 1:
        width = count - first - 1
        if width * (width + 1) // 2 <= target:
            row_count = width
        else:
            row_count = (2 * width + 1 - isqrt((2 * width + 1) ** 2 - 8 * target)) // 2
            if row_count * (2 * width - row_count + 1) // 2 > target:
                row_count -= 1
            row_count = max(1, row_count)
        # A rectangular triangle produces consecutive lexicographic pairs
        # directly, without a search or floating-point rank inversion per pair.
        pairs = torch.triu_indices(
            row_count, count, offset=first + 1, device=flat_a.device
        )
        left, right = pairs[0], pairs[1]
        left.add_(first)
        start = first * (2 * count - first - 1) // 2
        chunk = max(1, min(4096, target // (left.numel() + count)))
        for offset in range(0, flat_a.shape[0], chunk):
            aa = flat_a[offset : offset + chunk]
            bb = flat_b[offset : offset + chunk]
            ai, aj = aa[:, left], aa[:, right]
            bi, bj = bb[:, left], bb[:, right]
            determinant = ai * bj
            determinant.sub_(bi * aj).abs_()
            valid = (bi != 0) & (bj != 0) & (determinant > 0)
            bound = bi.abs()
            bound.add_(bj.abs())
            determinant.masked_fill_(~valid, 1.0)
            bound.div_(determinant).masked_fill_(~valid, torch.inf)
            value, index = bound.min(1)
            previous = best[offset : offset + chunk]
            # torch.argmin chooses the first NaN before any finite value.
            # Strict comparison also preserves earlier equal or infinite pairs.
            improve = (value < previous) | (value.isnan() & ~previous.isnan())
            best[offset : offset + chunk] = torch.where(improve, value, previous)
            choice[offset : offset + chunk] = torch.where(
                improve, start + index, choice[offset : offset + chunk]
            )
            del ai, aj, bi, bj, determinant, valid, bound
        del pairs, left, right
        first += row_count
    chunk = max(1, min(4096, target // count))
    for offset in range(0, flat_a.shape[0], chunk):
        aa = flat_a[offset : offset + chunk]
        bb = flat_b[offset : offset + chunk]
        cap = velocity_cap[offset : offset + chunk]
        divisor = aa.abs() * cap
        valid = (((aa < 0) & (bb > 0)) | ((aa > 0) & (bb < 0))) & (divisor > 0)
        bound = torch.where(
            valid, (bb.abs() + cap) / torch.where(valid, divisor, 1.0), torch.inf
        )
        value, index = bound.min(1)
        previous = best[offset : offset + chunk]
        improve = (value < previous) | (value.isnan() & ~previous.isnan())
        choice[offset : offset + chunk] = torch.where(
            improve, pair_count + index, choice[offset : offset + chunk]
        )
    return choice


def _screened_projection_choices(aa, bb, cap, pairs):
    """Retain every possible improving pair; uncertain rows use all pairs.

    For c = a * sign(b), d = abs(b), any pair's inverse bound is at
    most max(abs(c_i + d_i*z), abs(c_j + d_j*z)). A pilot bound and
    a guarded residual gap therefore exclude only non-improving pairs.
    The bounded pilot search need not converge.
    """

    count = aa.shape[1]
    pair_count = count * (count - 1) // 2
    rows = aa.shape[0]
    nonzero = bb != 0
    c = aa * bb.sign()
    d = bb.abs()
    result = torch.empty(rows, dtype=torch.int64, device=aa.device)
    # In these exact relations every strip-pair determinant is zero (or
    # nonfinite and hence invalid), even outside the screening range guard.
    unrestricted = (
        (aa == -bb).all(1)
        | (aa == bb).all(1)
        | ((c == c[:, :1]) & (d == d[:, :1])).all(1)
    )
    has_unrestricted = bool(unrestricted.any())
    if has_unrestricted:
        a, b, velocity_cap = aa[unrestricted], bb[unrestricted], cap[unrestricted]
        divisor = a.abs() * velocity_cap
        valid = (((a < 0) & (b > 0)) | ((a > 0) & (b < 0))) & (divisor > 0)
        bound = torch.where(
            valid,
            (b.abs() + velocity_cap) / torch.where(valid, divisor, 1.0),
            torch.inf,
        )
        value, index = bound.min(1)
        result[unrestricted] = torch.where(value == torch.inf, 0, pair_count + index)
        if bool(unrestricted.all()):
            return result
    maximum = torch.maximum(c.abs().amax(1), d.amax(1))
    minimum = torch.minimum(
        torch.where(c != 0, c.abs(), torch.inf).amin(1),
        torch.where(nonzero, d, torch.inf).amin(1),
    )
    # Guard original products, sums and quotients against range loss.
    eligible = (
        ~unrestricted & (maximum <= 1e100) & (minimum >= 1e-100) & (nonzero.sum(1) >= 2)
    )
    if not bool(eligible.any()):
        if not has_unrestricted:
            return _complete_projection_choices(aa, bb, cap, pairs)
        remaining = ~unrestricted
        result[remaining] = _complete_projection_choices(
            aa[remaining], bb[remaining], cap[remaining], pairs
        )
        return result
    denominator = torch.where(nonzero, d, 1.0)
    ratios = c / denominator
    high_ratio, high = torch.where(nonzero, ratios, -torch.inf).max(1)
    low_ratio, low = torch.where(nonzero, ratios, torch.inf).min(1)
    # Nearly proportional strips usually retain a crowded residual set.
    # This performance dispatch uses the complete original enumeration;
    # it never removes constraints or changes numerical acceptance.
    ratio_scale = torch.maximum(high_ratio.abs(), low_ratio.abs())
    eligible &= (high_ratio - low_ratio) > (
        512.0 * torch.finfo(torch.float64).eps * ratio_scale
    )
    if not bool(eligible.any()):
        if not has_unrestricted:
            return _complete_projection_choices(aa, bb, cap, pairs)
        remaining = ~unrestricted
        result[remaining] = _complete_projection_choices(
            aa[remaining], bb[remaining], cap[remaining], pairs
        )
        return result
    inverse_weights = 1.0 / denominator

    def pair_cap(i, j):
        left, right = torch.minimum(i, j), torch.maximum(i, j)
        ai, aj = (
            aa.gather(1, left[:, None]).squeeze(1),
            aa.gather(1, right[:, None]).squeeze(1),
        )
        bi, bj = (
            bb.gather(1, left[:, None]).squeeze(1),
            bb.gather(1, right[:, None]).squeeze(1),
        )
        determinant = (ai * bj - bi * aj).abs()
        return torch.where(
            determinant > 0,
            (bi.abs() + bj.abs()) / torch.where(determinant > 0, determinant, 1.0),
            torch.inf,
        )

    best = pair_cap(high, low)
    for _ in range(4):
        inverse = torch.where(torch.isfinite(best) & (best > 0), 1.0 / best, 0.0)
        weighted = inverse[:, None] * inverse_weights
        h = torch.where(nonzero, ratios - weighted, -torch.inf).argmax(1)
        lower_index = torch.where(nonzero, ratios + weighted, torch.inf).argmin(1)
        candidate = pair_cap(h, lower_index)
        improved = candidate < best
        best = torch.minimum(best, candidate)
        high = torch.where(improved, h, high)
        low = torch.where(improved, lower_index, low)
    inverse = torch.where(torch.isfinite(best) & (best > 0), 1.0 / best, 0.0)
    center = -(c.gather(1, high[:, None]) + c.gather(1, low[:, None])) / (
        d.gather(1, high[:, None]) + d.gather(1, low[:, None])
    )
    magnitude = c.abs().amax(1) + d.amax(1) * center.squeeze(1).abs()
    # Original pair operations and residual evaluation together fit within
    # 32 eps of this magnitude in the guarded range; retain a 128 eps gap.
    tolerance = (
        128.0 * torch.finfo(torch.float64).eps * torch.maximum(magnitude, inverse)
    )
    active = nonzero & ((c + d * center).abs() >= inverse[:, None] - tolerance[:, None])
    active.scatter_(1, high[:, None], True)
    active.scatter_(1, low[:, None], True)
    crowded = eligible & (active.sum(1) > 8)
    if bool(crowded.any()):
        cc, dd, retained = c[crowded], d[crowded], active[crowded]
        # Stable lexicographic ordering puts each exact oriented normal's
        # lowest original index first. No tolerance merges nearby strips.
        order = dd.argsort(dim=1, stable=True)
        sorted_d = dd.gather(1, order)
        sorted_retained = retained.gather(1, order)
        # Exact oriented duplicates require equal |b|. If no such pair
        # touches the active set, further sorting cannot shrink that set.
        possible = (
            (sorted_d[:, 1:] == sorted_d[:, :-1])
            & (sorted_retained[:, 1:] | sorted_retained[:, :-1])
        ).any(1)
        if bool(possible.any()):
            selected_rows = torch.nonzero(crowded, as_tuple=False).squeeze(1)[possible]
            cc, dd, retained, order = (
                value[possible] for value in (cc, dd, retained, order)
            )
            order = order.gather(1, cc.gather(1, order).argsort(dim=1, stable=True))
            sorted_c, sorted_d = cc.gather(1, order), dd.gather(1, order)
            starts = torch.ones_like(retained)
            starts[:, 1:] = (sorted_c[:, 1:] != sorted_c[:, :-1]) | (
                sorted_d[:, 1:] != sorted_d[:, :-1]
            )
            positions = torch.arange(count, device=aa.device).expand_as(order)
            starts = torch.where(starts, positions, 0).cummax(1).values
            representatives = order.gather(1, starts)
            unique = torch.zeros_like(order)
            unique.scatter_add_(1, representatives, retained.gather(1, order).long())
            active[selected_rows] = unique > 0
    accepted = (
        eligible
        & torch.isfinite(best)
        & (best > 0)
        & torch.isfinite(magnitude)
        & (active.sum(1) <= 8)
    )
    if accepted.any():
        a, b, velocity_cap = aa[accepted], bb[accepted], cap[accepted]
        active = active[accepted]
        # This path runs only on CPU. Size scratch to the largest retained
        # set in this batch instead of padding every row to eight strips.
        selected_count = int(active.sum(1).max())
        indices = torch.arange(count, device=aa.device)
        selected = (
            torch.where(active, indices, count)
            .topk(selected_count, dim=1, largest=False, sorted=False)
            .values
        )
        valid_selected = selected < count
        selected = selected.clamp_max(count - 1)
        i = selected[:, :, None]
        ai = a.gather(1, selected)[:, :, None]
        bi = b.gather(1, selected)[:, :, None]
        aj = a[:, None, :]
        bj = b[:, None, :]
        determinant = (ai * bj - bi * aj).abs()
        valid = (
            (bi != 0)
            & (bj != 0)
            & (determinant > 0)
            & valid_selected[:, :, None]
            & (i != indices)
        )
        bound = torch.where(
            valid,
            (bi.abs() + bj.abs()) / torch.where(valid, determinant, 1.0),
            torch.inf,
        )
        # For a fixed selected strip, the original pair index increases with
        # the other strip's index. min therefore keeps the first exact tie.
        pair_values, j = bound.min(2)
        left, right = torch.minimum(selected, j), torch.maximum(selected, j)
        pair_indices = left * (2 * count - left - 1) // 2 + right - left - 1
        divisor = a.abs() * velocity_cap
        velocity_valid = (((a < 0) & (b > 0)) | ((a > 0) & (b < 0))) & (divisor > 0)
        velocity_bound = torch.where(
            velocity_valid,
            (b.abs() + velocity_cap) / torch.where(velocity_valid, divisor, 1.0),
            torch.inf,
        )
        velocity_value, velocity_index = velocity_bound.min(1)
        bound = torch.cat((pair_values, velocity_value[:, None]), 1)
        pair_indices = torch.cat(
            (pair_indices, (pair_count + velocity_index)[:, None]), 1
        )
        minimum = bound.amin(1)
        choice = torch.where(
            bound == minimum[:, None], pair_indices, pair_count + count
        ).amin(1)
        # Each selected strip compares against every original strip. Reverse
        # determinant orientation changes only the sign, preserving its
        # absolute value. Restore original pair order explicitly on ties.
        # With no finite pair, full argmin returns column zero.
        result[accepted] = torch.where(torch.isfinite(minimum), choice, 0)
    remaining = ~accepted & ~unrestricted
    if bool(remaining.any()):
        result[remaining] = _complete_projection_choices(
            aa[remaining], bb[remaining], cap[remaining], pairs
        )
    return result
