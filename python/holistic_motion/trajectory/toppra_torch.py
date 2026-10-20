"""Differentiable, batched TOPPRA on PyTorch CPU and CUDA devices.

Active constraints, spline segment lookups, and grid topology are discrete.
Gradients describe the selected smooth branch, not switches between branches.
Numerical work stays on the input device and uses float64. Validation and
terminal-speed dispatch synchronize CUDA with the host. No NumPy solver is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from numbers import Integral

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ImportError(
        "Differentiable TOPPRA requires the 'differentiable' extra: "
        "pip install 'holistic-motion[differentiable]'"
    ) from error

__all__ = ["TorchToppraResult", "TorchToppraTrajectory", "retime_path_torch"]

_ROUNDING_EPS = 8.0 * torch.finfo(torch.float64).eps
_LARGE_GEOMETRY = (torch.finfo(torch.float64).max ** 0.5) / 16.0


def _check(condition: torch.Tensor, message: str) -> None:
    if not bool(condition.all()):
        raise ValueError(message)


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 2:
        raise ValueError(f"{name} must be at least two")
    return int(value)


def _sqrt_speed(x: torch.Tensor) -> torch.Tensor:
    # Avoid evaluating sqrt'(0), even on an unselected torch.where branch.
    moving = x > 0.0
    return torch.where(moving, torch.sqrt(torch.where(moving, x, 1.0)), 0.0)


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))


class _StableDivide(torch.autograd.Function):
    """Keep zero adjoints zero for inactive constraints and LP bounds."""

    @staticmethod
    def forward(
        ctx, numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        result = numerator / denominator
        ctx.save_for_backward(denominator, result if ctx.needs_input_grad[1] else None)
        return result

    @staticmethod
    def backward(
        ctx, gradient: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        denominator, result = ctx.saved_tensors
        # Scale the incoming adjoint before multiplying by the quotient. The
        # alternative -(result / denominator) * gradient can overflow even
        # when the incoming adjoint is exactly zero. Infinite candidate bounds
        # are never selected by a finite feasible solution, so their quotient
        # factor can safely be zero for this unused denominator adjoint.
        numerator_gradient = gradient / denominator
        denominator_gradient = (
            -numerator_gradient
            * torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            if ctx.needs_input_grad[1]
            else None
        )
        return (
            numerator_gradient if ctx.needs_input_grad[0] else None,
            denominator_gradient,
        )


def _divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    # Native division already preserves zero numerator adjoints. Only its
    # denominator adjoint needs the guarded multiplication order.
    if torch.is_grad_enabled() and denominator.requires_grad:
        return _StableDivide.apply(numerator, denominator)
    return numerator / denominator


def _projection_caps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Eliminate y using symmetric acceleration strips and its velocity cap.

    The first K columns and the next K columns are opposite half-planes:
    abs(a_i*x + b_i*y) <= 1. Each distinct pair with nonzero b gives
    abs(a_i*b_j - b_i*a_j)*x <= abs(b_i) + abs(b_j). Thus only K*(K-1)/2
    pairs are needed instead of a square over all 2*K+2 half-planes.
    """

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
                    _screened_projection_choices(
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
        l = torch.where(nonzero, ratios + weighted, torch.inf).argmin(1)
        candidate = pair_cap(h, l)
        improved = candidate < best
        best = torch.minimum(best, candidate)
        high = torch.where(improved, h, high)
        low = torch.where(improved, l, low)
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


def _parallel_tridiagonal(
    diagonal: torch.Tensor, off_diagonal: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Solve a symmetric, strictly diagonally dominant spline system.

    Eliminate both neighbors at distances 1, 2, 4, ... in parallel. Natural
    spline diagonals are twice the sum of adjacent positive interval widths,
    so each elimination preserves positive pivots without pivot selection.
    Storage is linear in the number of intervals, including on CUDA.
    """

    zero = torch.zeros_like(diagonal[:, :1])
    lower = torch.cat((zero, off_diagonal), dim=1)
    upper = torch.cat((off_diagonal, zero), dim=1)
    stride = 1
    while stride < diagonal.shape[1]:
        left = lower[:, stride:] / diagonal[:, :-stride]
        right = upper[:, :-stride] / diagonal[:, stride:]
        # New private buffers keep inputs and saved tensors untouched, while
        # avoiding padded full-size products for the two neighbor updates.
        next_diagonal = diagonal.clone()
        next_diagonal[:, stride:].addcmul_(left, upper[:, :-stride], value=-1.0)
        next_diagonal[:, :-stride].addcmul_(right, lower[:, stride:], value=-1.0)
        next_rhs = rhs.clone()
        next_rhs[:, stride:].addcmul_(left[..., None], rhs[:, :-stride], value=-1.0)
        next_rhs[:, :-stride].addcmul_(right[..., None], rhs[:, stride:], value=-1.0)
        if 2 * stride < diagonal.shape[1]:
            zeros = torch.zeros_like(diagonal[:, :stride])
            lower = torch.cat((zeros, -left * lower[:, :-stride]), dim=1)
            upper = torch.cat((-right * upper[:, stride:], zeros), dim=1)
        diagonal, rhs = next_diagonal, next_rhs
        stride *= 2
    return rhs / diagonal[..., None]


class _SplineSolve(torch.autograd.Function):
    """Differentiate A x = rhs through a second symmetric solve.

    Save only A and x, rather than the intermediate elimination tensors.
    Reusing this operation in backward also preserves higher derivatives.
    """

    @staticmethod
    def forward(ctx, diagonal, off_diagonal, rhs):
        output = _parallel_tridiagonal(diagonal, off_diagonal, rhs)
        # Parallel elimination can lose relative accuracy in small solution
        # components. Correct its residual with one more solve of the original
        # system, keeping the implicit adjoint and linear storage unchanged.
        residual = torch.addcmul(rhs, diagonal[..., None], output, value=-1.0)
        residual[:, 1:].addcmul_(off_diagonal[..., None], output[:, :-1], value=-1.0)
        residual[:, :-1].addcmul_(off_diagonal[..., None], output[:, 1:], value=-1.0)
        output = output + _parallel_tridiagonal(diagonal, off_diagonal, residual)
        ctx.save_for_backward(diagonal, off_diagonal, output)
        return output

    @staticmethod
    def backward(ctx, gradient):
        diagonal, off_diagonal, output = ctx.saved_tensors
        adjoint = _SplineSolve.apply(diagonal, off_diagonal, gradient)
        diagonal_gradient = (
            -(adjoint * output).sum(-1) if ctx.needs_input_grad[0] else None
        )
        off_diagonal_gradient = (
            -(adjoint[:, :-1] * output[:, 1:] + adjoint[:, 1:] * output[:, :-1]).sum(-1)
            if ctx.needs_input_grad[1]
            else None
        )
        return (
            diagonal_gradient,
            off_diagonal_gradient,
            adjoint if ctx.needs_input_grad[2] else None,
        )


@dataclass(frozen=True)
class TorchToppraResult:
    """Float64 timing tensors with the input's device and autograd graph.

    Single paths have shape (G,), batched paths (B, G); accelerations have
    G-1 entries. Duration is scalar or (B,). These tensors are owned copies:
    mutating them does not alter the trajectory's internal sampling state.
    As with other Torch outputs, do not mutate tensors needed by backward.
    """

    gridpoints: torch.Tensor
    path_speeds: torch.Tensor
    path_accelerations: torch.Tensor
    times: torch.Tensor
    duration: torch.Tensor


class TorchToppraTrajectory:
    """Batched natural-cubic path timing with differentiable CPU/CUDA tensors.

    ``waypoints`` must be a floating tensor (N, D) or (B, N, D). Limits may be
    positive scalars, (D,), or (B, D). Endpoint path speeds may be scalar or
    (B,). Invalid or infeasible input raises ValueError for the entire batch.

    Each spline segment has ceil((grid_size-1)/(N-1)) subdivisions, with a
    minimum of two for a two-waypoint path. This fixed topology supports batch
    execution and differentiation of chord-length knots. It can differ from
    the NumPy default grid; use result.gridpoints for a same-grid comparison.

    Internal timing uses float64 even for float16/bfloat16 inputs. Sampled
    position/velocity/acceleration tensors use the waypoint dtype. Gradients
    flow to waypoints, limits, boundary speeds and sample times. They are
    piecewise derivatives with active constraints and segment lookups fixed;
    they are not guaranteed at switches. Zero speed uses a zero sqrt adjoint.
    """

    def __init__(
        self,
        waypoints: torch.Tensor,
        max_velocity,
        max_acceleration,
        *,
        start_path_velocity=0.0,
        end_path_velocity=0.0,
        grid_size: int = 200,
    ) -> None:
        grid_size = _count(grid_size, "grid_size")
        if not isinstance(waypoints, torch.Tensor):
            raise TypeError("waypoints must be a Torch tensor")
        if (
            waypoints.ndim not in (2, 3)
            or min(waypoints.shape) < 1
            or waypoints.shape[-2] < 2
        ):
            raise ValueError("waypoints must have shape (N >= 2, D >= 1) or (B, N, D)")
        if waypoints.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise TypeError("waypoints must have a floating dtype")
        if waypoints.device.type not in ("cpu", "cuda"):
            raise ValueError("TOPPRA supports CPU and CUDA devices")
        self._single = waypoints.ndim == 2
        self._dtype, self._device = waypoints.dtype, waypoints.device
        points = waypoints.to(dtype=torch.float64).clone()
        self._points = points.unsqueeze(0) if self._single else points
        self._batch, count, self.dof = self._points.shape
        velocity = self._limits(max_velocity, "max_velocity")
        acceleration = self._limits(max_acceleration, "max_acceleration")
        start = self._boundary(start_path_velocity)
        end = self._boundary(end_path_velocity)
        checks = (
            (torch.isfinite(waypoints), "waypoints must be finite"),
            (
                torch.isfinite(velocity) & (velocity > 0),
                "max_velocity must be finite and strictly positive",
            ),
            (
                torch.isfinite(acceleration) & (acceleration > 0),
                "max_acceleration must be finite and strictly positive",
            ),
            (
                torch.isfinite(start) & (start >= 0),
                "boundary path velocities must be finite and non-negative",
            ),
            (
                torch.isfinite(end) & (end >= 0),
                "boundary path velocities must be finite and non-negative",
            ),
        )
        if self._device.type == "cuda":
            # Read all input-value flags in one transfer instead of forcing
            # a separate device synchronization for each input tensor.
            valid = torch.stack([condition.all() for condition, _ in checks]).tolist()
            for flag, (_, message) in zip(valid, checks):
                if not flag:
                    raise ValueError(message)
        else:
            for condition, message in checks:
                _check(condition, message)
        # Release full input masks before allocating spline/constraint storage.
        del checks
        self._velocity = velocity.expand(self._batch, self.dof).clone()
        self._acceleration = acceleration.expand(self._batch, self.dof).clone()
        start, end = start.expand(self._batch).clone(), end.expand(self._batch).clone()
        delta = torch.diff(self._points, dim=1)
        # Squaring physical coordinates in the norm can overflow even when
        # the chord length itself is representable. Its arbitrary scale has
        # no effect on the derivative of the norm, so do not differentiate it.
        with torch.no_grad():
            length_scale = delta.abs().amax(dim=-1, keepdim=True)
            length_scale = torch.where(length_scale > 0, length_scale, 1.0)
        lengths = torch.linalg.vector_norm(
            delta / length_scale, dim=-1
        ) * length_scale.squeeze(-1)
        zero = torch.zeros((self._batch, 1), dtype=torch.float64, device=self._device)
        cumulative = torch.cat((zero, lengths.cumsum(1)), dim=1)
        # Transfer both validation/dispatch flags together on CUDA.
        valid_chords, ordinary_chords = torch.stack(
            (
                (torch.isfinite(lengths) & (lengths > 1e-12)).all(),
                torch.isfinite(cumulative[:, -1]).all()
                & (lengths <= _LARGE_GEOMETRY).all(),
            )
        ).tolist()
        if not valid_chords:
            raise ValueError(
                "consecutive waypoints must be distinct with finite chord lengths"
            )
        if not ordinary_chords:
            with torch.no_grad():
                scale = lengths.amax(dim=1, keepdim=True)
            cumulative = torch.cat((zero, (lengths / scale).cumsum(1)), dim=1)
        self._knots = cumulative / cumulative[:, -1:]
        self._fit_spline()
        subdivisions = max(1, (grid_size - 2 + count - 1) // (count - 1))
        if count == 2:
            subdivisions = max(2, subdivisions)
        fraction = (
            torch.arange(subdivisions, dtype=torch.float64, device=self._device)
            / subdivisions
        )
        grid = (
            self._knots[:, :-1, None]
            + torch.diff(self._knots, dim=1)[..., None] * fraction
        )
        self._grid = torch.cat((grid.flatten(1), torch.ones_like(zero)), dim=1)
        _check(torch.diff(self._grid, dim=1) > 0, "path grid is not representable")
        _, first, second = self._evaluate(self._grid, position=False)
        a, b, speed_scale = self._constraints(first, second)
        _check(
            torch.isfinite(a) & torch.isfinite(b),
            "path speed constraints must be finite",
        )
        x = self._solve(
            a,
            b,
            (start / speed_scale).square(),
            (end / speed_scale).square(),
        )
        del a, b
        normalized_speeds = _sqrt_speed(x)
        self._speed_scale = speed_scale[:, None]
        self._normalized_speeds = normalized_speeds
        self._speeds = normalized_speeds * self._speed_scale
        ds = torch.diff(self._grid, dim=1)
        self._normalized_u = torch.diff(x, dim=1) / (2.0 * ds)
        self._u = (self._normalized_u * self._speed_scale) * self._speed_scale
        denominator = normalized_speeds[:, :-1] + normalized_speeds[:, 1:]
        _check(
            denominator > 0.0,
            "path contains an interval with zero reachable speed; increase grid_size",
        )
        self._times = torch.cat(
            (zero, (2.0 * ds / denominator / speed_scale[:, None]).cumsum(1)), dim=1
        )
        _check(
            torch.isfinite(self._times[:, 1:])
            & torch.isfinite(self._speeds[:, :-1])
            & torch.isfinite(self._speeds[:, 1:])
            & torch.isfinite(self._u)
            & (torch.diff(self._times, dim=1) > 0),
            "path timing is not representable",
        )
        if self._physical_validation_needed:
            self._validate_physical_limits()
        self._result = TorchToppraResult(
            *(
                self._output(value).clone()
                for value in (
                    self._grid,
                    self._speeds,
                    self._u,
                    self._times,
                    self._times[:, -1],
                )
            )
        )

    def _tensor(self, value) -> torch.Tensor:
        # Torch otherwise discards imaginary components during a real cast.
        # Check tensor/array metadata before conversion; Python complex values
        # and complex lists are already rejected by torch.as_tensor itself.
        dtype = getattr(value, "dtype", None)
        if getattr(dtype, "is_complex", False) or getattr(dtype, "kind", None) == "c":
            raise TypeError("motion limits, boundary velocities and times must be real")
        return torch.as_tensor(value, dtype=torch.float64, device=self._device)

    def _limits(self, value, name: str) -> torch.Tensor:
        limits = self._tensor(value)
        if limits.shape not in ((), (self.dof,), (self._batch, self.dof)):
            raise ValueError(f"{name} must be scalar, (D,), or (B, D)")
        return limits

    def _boundary(self, value) -> torch.Tensor:
        speed = self._tensor(value)
        if speed.shape not in ((), (self._batch,)):
            raise ValueError("boundary path velocities must be scalar or (B,)")
        return speed

    def _output(self, value: torch.Tensor) -> torch.Tensor:
        return value[0] if self._single else value

    @property
    def result(self) -> TorchToppraResult:
        return self._result

    @property
    def duration(self) -> torch.Tensor:
        return self._output(self._times[:, -1]).clone()

    def _fit_spline(self) -> None:
        h = torch.diff(self._knots, dim=1)
        delta = torch.diff(self._points, dim=1)
        coefficients = self._derivative_coefficients(h, delta / h[..., None])
        with torch.no_grad():
            finite_coefficients = torch.stack(
                [torch.isfinite(value).all() for value in coefficients]
            ).all()
            ordinary = finite_coefficients & (delta.abs() <= _LARGE_GEOMETRY).all()
            if h.requires_grad:
                ordinary &= torch.isfinite(delta / h[..., None].square()).all()
            valid_intervals, ordinary_coefficients = torch.stack(
                (
                    (torch.isfinite(h) & (h > 0)).all(),
                    ordinary,
                )
            ).tolist()
        if not valid_intervals:
            raise ValueError("waypoint parameters must increase strictly")
        if not ordinary_coefficients:
            # The spline solve is linear in each coordinate. Normalize the
            # differences independently, retaining large offsets in _a and
            # keeping gradients through every waypoint and interval width.
            with torch.no_grad():
                scale = delta.abs().amax(dim=1, keepdim=True)
                scale = torch.where(scale > 0, scale, 1.0)
            coefficients = self._derivative_coefficients(
                h, (delta / scale) / h[..., None]
            )
            _check(
                torch.stack(
                    [torch.isfinite(value * scale).all() for value in coefficients]
                ),
                "spline derivative coefficients must be finite",
            )
            self._coefficient_scale = scale
        else:
            self._coefficient_scale = None
        self._a = self._points[:, :-1]
        self._b, self._c, self._d = coefficients

    @staticmethod
    def _derivative_coefficients(h, slopes):
        zero = torch.zeros_like(slopes[:, 0])
        # CPU forward needs a small working set to amortize two solves.
        # Backward also avoids the sequential elimination tape, so it can
        # profit from larger inputs, particularly on long splines.
        needs_spline_grad = torch.is_grad_enabled() and (
            h.requires_grad or slopes.requires_grad
        )
        cpu_element_limit = (
            ((1 << 20) if h.shape[1] >= 255 else (1 << 18))
            if needs_spline_grad
            else (1 << 16)
        )
        cpu_long_enough = h.shape[1] >= 31 or (
            needs_spline_grad and slopes.shape[0] * slopes.shape[2] < 2048
        )
        if h.shape[1] >= 15 and (
            h.device.type != "cpu"
            or (slopes.numel() <= cpu_element_limit and cpu_long_enough)
        ):
            # Parallel elimination amortizes its tensor setup on longer
            # splines. Wide or large CPU inputs instead benefit from the
            # sequential solve's lower memory traffic and fewer copies.
            diagonal = 2.0 * (h[:, :-1] + h[:, 1:])
            rhs = 6.0 * (slopes[:, 1:] - slopes[:, :-1])
            interior = _SplineSolve.apply(diagonal, h[:, 1:-1], rhs)
            second = torch.cat((zero[:, None], interior, zero[:, None]), dim=1)
            return (
                slopes - h[..., None] * (2.0 * second[:, :-1] + second[:, 1:]) / 6.0,
                second[:, :-1] / 2.0,
                torch.diff(second, dim=1) / (6.0 * h[..., None]),
            )
        # Collect interval adjoints once instead of allocating a full-sized
        # zero gradient buffer for every slice in a long spline solve.
        interval_widths, segment_slopes = h.unbind(1), slopes.unbind(1)
        diagonal, rhs = [], []
        for index in range(h.shape[1] - 1):
            d = 2.0 * (interval_widths[index] + interval_widths[index + 1])
            r = 6.0 * (segment_slopes[index + 1] - segment_slopes[index])
            if index:
                factor = interval_widths[index] / diagonal[-1]
                d = d - factor * interval_widths[index]
                r = r - factor[:, None] * rhs[-1]
            diagonal.append(d)
            rhs.append(r)
        interior = []
        next_value = zero
        for index in range(len(diagonal) - 1, -1, -1):
            next_value = (
                rhs[index] - interval_widths[index + 1][:, None] * next_value
            ) / diagonal[index][:, None]
            interior.append(next_value)
        second = torch.stack([zero, *reversed(interior), zero], dim=1)
        return (
            slopes - h[..., None] * (2.0 * second[:, :-1] + second[:, 1:]) / 6.0,
            second[:, :-1] / 2.0,
            torch.diff(second, dim=1) / (6.0 * h[..., None]),
        )

    def _evaluate(
        self, s: torch.Tensor, *, position: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        index = (
            torch.searchsorted(self._knots.contiguous(), s.contiguous(), right=True) - 1
        ).clamp(0, self._a.shape[1] - 1)
        delta = (s - self._knots.gather(1, index))[..., None]
        b, c, d = (_gather(value, index) for value in (self._b, self._c, self._d))
        # Constraint construction needs only derivatives. Avoid keeping an
        # unused position graph alive through the complete timing solve.
        correction = delta * (b + delta * (c + delta * d)) if position else None
        first = b + delta * (2.0 * c + 3.0 * delta * d)
        second = 2.0 * c + 6.0 * delta * d
        if self._coefficient_scale is not None:
            # Large physical terms may overflow before polynomial terms
            # cancel. Evaluate the normalized spline before restoring units.
            if position:
                correction = correction * self._coefficient_scale
            first, second = (
                first * self._coefficient_scale,
                second * self._coefficient_scale,
            )
        return _gather(self._a, index) + correction if position else None, first, second

    def _constraints(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = torch.diff(self._grid, dim=1)[..., None]
        f, last = first[:, :-1], first[:, 1:]
        sf, sl = second[:, :-1], second[:, 1:]
        derivative = torch.stack((f, f + 0.5 * ds * sf, last), dim=2)
        zero = torch.zeros_like(f)
        ax = torch.stack((sf, 0.5 * sl, zero), dim=2) - derivative / (
            2.0 * ds[:, :, None]
        )
        ay = torch.stack((zero, 0.5 * sf, sl), dim=2) + derivative / (
            2.0 * ds[:, :, None]
        )
        with torch.no_grad():
            # Reverse-mode differentiation also divides by interval widths:
            # its intermediate quotient can overflow before a small incoming
            # adjoint restores range, even while the forward ratio is finite.
            ordinary = torch.isfinite(ax).all() & torch.isfinite(ay).all()
            ordinary &= torch.isfinite(
                derivative / (2.0 * ds[:, :, None]).square()
            ).all()
            self._normalized_acceleration_constraints = not bool(ordinary)
        if self._normalized_acceleration_constraints:
            # Dividing a large coefficient by a short interval can overflow
            # before a large acceleration limit restores its finite ratio.
            acceleration = self._acceleration[:, None]
            normalized_first = _divide(f, acceleration)
            normalized_second = _divide(sf, acceleration)
            normalized_derivative = torch.stack(
                (
                    normalized_first,
                    normalized_first + 0.5 * ds * normalized_second,
                    _divide(last, acceleration),
                ),
                dim=2,
            )
            ax = torch.stack(
                (
                    normalized_second,
                    _divide(0.5 * sl, acceleration),
                    zero,
                ),
                dim=2,
            ) - normalized_derivative / (2.0 * ds[:, :, None])
            ay = torch.stack(
                (
                    zero,
                    0.5 * normalized_second,
                    _divide(sl, acceleration),
                ),
                dim=2,
            ) + normalized_derivative / (2.0 * ds[:, :, None])
            _check(
                torch.isfinite(ax) & torch.isfinite(ay),
                "normalized path acceleration constraints must be finite",
            )
            velocity = self._velocity[:, None]
            normalized_velocity = torch.stack(
                (
                    _divide(f, velocity),
                    _divide(f, velocity) + 0.5 * ds * _divide(sf, velocity),
                    _divide(last, velocity),
                ),
                dim=2,
            )
        # Solve in dimensionless speed units. Normalize limits *before*
        # division so both elimination products and their adjoints remain
        # representable when the caller changes time units. This is only a
        # change of variables; its derivative cancels identically. Detach the
        # arbitrary scale selection, keeping gradients through the limits.
        with torch.no_grad():
            velocity_scale = (
                normalized_velocity.abs()
                if self._normalized_acceleration_constraints
                else derivative.abs() / self._velocity[:, None, None]
            ).amax(dim=(1, 2, 3))
            acceleration_scale = torch.maximum(ax.abs(), ay.abs()).sqrt()
            if not self._normalized_acceleration_constraints:
                acceleration_scale = (
                    acceleration_scale / self._acceleration[:, None, None].sqrt()
                )
            acceleration_scale = acceleration_scale.amax(dim=(1, 2, 3))
            speed_scale = torch.maximum(velocity_scale, acceleration_scale).reciprocal()
        _check(
            torch.isfinite(speed_scale) & (speed_scale > 0),
            "path speed scale is not representable",
        )
        if self._normalized_acceleration_constraints:
            self._limit_unit_overflow = None
            scale = speed_scale[:, None, None, None]
            ax, ay = (ax * scale) * scale, (ay * scale) * scale
            normalized_velocity = normalized_velocity * scale
        else:
            velocity = self._velocity / speed_scale[:, None]
            acceleration = (self._acceleration / speed_scale[:, None]) / speed_scale[
                :, None
            ]
            self._limit_unit_overflow = (
                ~torch.isfinite(velocity) | ~torch.isfinite(acceleration)
            ).any()
            # A zero normalized limit belongs to an inactive axis: the chosen
            # speed scale bounds every nonzero derivative/acceleration coefficient.
            # Keep its zero numerators and adjoints finite rather than forming 0/0.
            velocity = torch.where(velocity > 0, velocity, 1.0)
            acceleration = torch.where(acceleration > 0, acceleration, 1.0)
            # Inactive subnormal limits still have exactly zero adjoints;
            # avoid forming an overflowing reciprocal before multiplying it
            # by that zero incoming gradient.
            ax = _divide(ax, acceleration[:, None, None])
            ay = _divide(ay, acceleration[:, None, None])
            normalized_velocity = _divide(derivative, velocity[:, None, None])
        p0, p1, p2 = normalized_velocity.unbind(2)
        quadratic = p0 - 2.0 * p1 + p2
        ratio = torch.where(
            quadratic != 0,
            _divide(p0 - p1, torch.where(quadratic != 0, quadratic, 1.0)),
            0.0,
        ).clamp(0, 1)
        extremum = (
            (1.0 - ratio).square() * p0
            + 2.0 * ratio * (1.0 - ratio) * p1
            + ratio.square() * p2
        )
        maximum = torch.maximum(torch.maximum(p0.abs(), p2.abs()), extremum.abs())
        inverse_cap = maximum.square().max(-1, keepdim=True).values
        ax, ay = ax.flatten(2), ay.flatten(2)
        return (
            torch.cat((ax, -ax, inverse_cap, torch.zeros_like(inverse_cap)), dim=-1),
            torch.cat((ay, -ay, torch.zeros_like(inverse_cap), inverse_cap), dim=-1),
            speed_scale,
        )

    def _solve(
        self, a: torch.Tensor, b: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> torch.Tensor:
        if self._a.shape[1] == 1:
            return self._solve_linear(a, b, start, end)
        caps = _projection_caps(a, b)
        # Unbind once: repeated a[:, i] indexing creates a full-sized zero
        # gradient buffer for every interval during backward. Unbind collects
        # the interval adjoints in one stack instead.
        aa_intervals, bb_intervals = a.unbind(1), b.unbind(1)
        cap_intervals = caps.unbind(1)
        lower = None
        positive_b = b > 0
        positive_b_intervals = positive_b.unbind(1)
        if bool((end == 0).all()):
            upper = self._controllable_to_rest(a, b, caps, end)
        else:
            lower, upper = [end], [end]
            denominators_a = torch.where(a != 0, a, 1.0).unbind(1)
            negative_a, positive_a = (a < 0).unbind(1), (a > 0).unbind(1)
            for index in range(a.shape[1] - 1, -1, -1):
                bb = bb_intervals[index]
                rhs = 1.0 - bb * torch.where(
                    positive_b_intervals[index], lower[-1][:, None], upper[-1][:, None]
                )
                ratio = _divide(rhs, denominators_a[index])
                lo = torch.where(negative_a[index], ratio, -torch.inf).max(-1).values
                hi = torch.where(positive_a[index], ratio, torch.inf).min(-1).values
                lo, hi = lo.clamp_min(0), torch.minimum(hi, cap_intervals[index])
                lower.append(lo)
                upper.append(hi)
            lower.reverse()
            upper.reverse()
        upper_tensor = torch.stack(upper, dim=1)
        lower_tensor = (
            torch.stack(lower, dim=1)
            if lower is not None
            else torch.zeros_like(upper_tensor)
        )
        _check(
            torch.isfinite(lower_tensor)
            & torch.isfinite(upper_tensor)
            & (lower_tensor <= upper_tensor + 1e-12),
            "path is infeasible for the requested end velocity",
        )
        _check(
            (start >= lower_tensor[:, 0] - 1e-12) & (start <= upper[0] + 1e-12),
            "start velocity cannot reach the requested end velocity",
        )
        # These effects are used only for discrete stop selection. Compute
        # all intervals once without retaining an additional autograd tape.
        with torch.no_grad():
            next_effect = (b * upper_tensor[:, 1:, None]).abs().unbind(1)
        values = [start]
        denominators = torch.where(positive_b, b, 1.0).unbind(1)
        for index in range(a.shape[1]):
            product = aa_intervals[index] * values[-1][:, None]
            rounding = _ROUNDING_EPS * product.abs().clamp_min(1.0)
            residual = 1.0 - product
            # A projected vertex may require an exact stop. Inflating a zero
            # residual produces a tiny speed whose sqrt adjoint amplifies
            # roundoff. Treat residuals within the same error bound as zero.
            # A near-zero b whose entire reachable b*y contribution fits
            # within roundoff acts only on x; it cannot require y == 0.
            stop = (residual.abs() <= rounding) & (next_effect[index] > rounding)
            rhs = torch.where(stop, 0.0, residual + rounding)
            ratio = _divide(rhs, denominators[index])
            high = (
                torch.where(positive_b_intervals[index], ratio, torch.inf)
                .min(-1)
                .values
            )
            value = torch.minimum(high, upper[index + 1])
            values.append(
                value.clamp_min(0)
                if lower is None
                else torch.maximum(lower[index + 1], value)
            )
        x = torch.stack(values, dim=1)
        self._validate_speed_profile(a, b, x)
        return x

    def _solve_linear(
        self, a: torch.Tensor, b: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> torch.Tensor:
        # A two-waypoint natural spline has constant first derivatives and
        # zero second derivatives. Its strips bound |x[i+1] - x[i]|, and the
        # velocity cap is constant. Prefix/suffix budgets therefore give the
        # same reachable profile without interval-wise LP projections.
        coefficient = a[..., : 3 * self.dof].abs().max(-1).values
        positive = coefficient > 0
        change = _divide(
            torch.ones_like(coefficient), torch.where(positive, coefficient, 1.0)
        )
        change = torch.where(positive, change, torch.inf)
        inverse_cap = a[:, 0, -2]
        moving = inverse_cap > 0
        cap = _divide(
            torch.ones_like(inverse_cap), torch.where(moving, inverse_cap, 1.0)
        )
        cap = torch.where(moving, cap, torch.inf)
        # A step larger than the velocity cap cannot limit any feasible
        # profile. Clipping it also keeps inactive infinite budgets finite.
        change = torch.minimum(change, cap[:, None])
        zero = torch.zeros_like(start[:, None])
        prefix = torch.cat((zero, change.cumsum(1)), dim=1)
        suffix = torch.cat((change.flip(1).cumsum(1).flip(1), zero), dim=1)
        upper = torch.minimum(cap[:, None], end[:, None] + suffix)
        lower_start = (end - suffix[:, 0]).clamp_min(0)
        _check(
            torch.isfinite(end) & (end <= cap + 1e-12),
            "path is infeasible for the requested end velocity",
        )
        _check(
            torch.isfinite(start)
            & (start >= lower_start - 1e-12)
            & (start <= upper[:, 0] + 1e-12),
            "start velocity cannot reach the requested end velocity",
        )
        x = torch.minimum(start[:, None] + prefix, upper)
        x = torch.cat((start[:, None], x[:, 1:-1], end[:, None]), dim=1)
        self._validate_speed_profile(a, b, x)
        return x

    def _validate_speed_profile(
        self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
    ) -> None:
        with torch.no_grad():
            left, right = a * x[:, :-1, None], b * x[:, 1:, None]
            scale = torch.maximum(left.abs(), right.abs()).clamp_min(1.0)
            _check(
                torch.isfinite(x) & (x >= 0),
                "path squared speeds must be finite and non-negative",
            )
            valid = (
                torch.isfinite(left)
                & torch.isfinite(right)
                & (left + right - 1.0 <= 1e-10 * scale)
            ).all()
            conditioned = (scale > 1e5).any()
            if self._limit_unit_overflow is not None:
                conditioned = conditioned | self._limit_unit_overflow
            valid_profile, conditioned = torch.stack((valid, conditioned)).tolist()
            if not valid_profile:
                raise ValueError("TOPPRA forward pass violates interval constraints")
            self._physical_validation_needed = conditioned

    def _validate_physical_limits(self) -> None:
        """Certify conditioned profiles without subtracting large LP products."""

        with torch.no_grad():
            speed_mantissa, speed_exponent = torch.frexp(self._speed_scale[..., None])
            limit_mantissa, limit_exponent = torch.frexp(self._acceleration[:, None])
            velocity_mantissa, velocity_exponent = torch.frexp(self._velocity[:, None])

            def normalized_coefficient(value, velocity=False):
                if self._coefficient_scale is not None:
                    value = value * self._coefficient_scale
                mantissa, exponent = torch.frexp(value)
                if velocity:
                    mantissa = mantissa * speed_mantissa / velocity_mantissa
                    exponent = exponent + speed_exponent - velocity_exponent
                else:
                    mantissa = mantissa * speed_mantissa.square() / limit_mantissa
                    exponent = exponent + 2 * speed_exponent - limit_exponent
                # Split the exponent: implementations that form 2**exponent
                # must not overflow before a small mantissa restores range.
                half = exponent // 2
                result = torch.ldexp(torch.ldexp(mantissa, half), exponent - half)
                return torch.where(value != 0, result, 0.0)

            # Target 1 MiB per value array; a single wide batch may exceed it.
            chunk = max(1, (1 << 17) // (self._batch * self.dof))
            for start in range(0, self._normalized_u.shape[1], chunk):
                stop = min(start + chunk, self._normalized_u.shape[1])
                grid = self._grid[:, start : stop + 1]
                times = self._times[:, start : stop + 1]
                index = (
                    torch.searchsorted(
                        self._knots.contiguous(), grid.contiguous(), right=True
                    )
                    - 1
                ).clamp(0, self._a.shape[1] - 1)
                local = (grid - self._knots.gather(1, index))[..., None]
                raw = tuple(
                    _gather(value, index) for value in (self._b, self._c, self._d)
                )
                b, c, d = (normalized_coefficient(value) for value in raw)
                first = b + local * (2.0 * c + 3.0 * local * d)
                second = 2.0 * c + 6.0 * local * d
                ds = torch.diff(grid, dim=1)[..., None]
                right_local = local[:, :-1] + ds
                left_b, left_c, left_d = b[:, :-1], c[:, :-1], d[:, :-1]
                first_right = left_b + right_local * (
                    2.0 * left_c + 3.0 * right_local * left_d
                )
                second_right = 2.0 * left_c + 6.0 * right_local * left_d
                first_mid = first[:, :-1] + 0.5 * ds * second[:, :-1]
                acceleration = self._normalized_u[:, start:stop, None]
                initial = self._normalized_speeds[:, start:stop, None]
                x0 = initial.square()
                x1 = x0 + (2.0 * ds) * acceleration
                controls = torch.stack(
                    (
                        second[:, :-1] * x0 + first[:, :-1] * acceleration,
                        0.5 * (second[:, :-1] * x1 + second_right * x0)
                        + first_mid * acceleration,
                        second_right * x1 + first_right * acceleration,
                        second[:, 1:] * x1 + first[:, 1:] * acceleration,
                    ),
                    dim=2,
                )
                # Bound evaluation error with uncancelled original polynomial
                # terms, including the next segment at a shared knot.
                extent = torch.maximum(local[:, :-1].abs(), right_local.abs())
                first_size = left_b.abs() + extent * (
                    2.0 * left_c.abs() + 3.0 * extent * left_d.abs()
                )
                second_size = 2.0 * left_c.abs() + 6.0 * extent * left_d.abs()
                first_size = torch.maximum(
                    first_size,
                    b[:, 1:].abs()
                    + local[:, 1:].abs()
                    * (
                        2.0 * c[:, 1:].abs() + 3.0 * local[:, 1:].abs() * d[:, 1:].abs()
                    ),
                )
                second_size = torch.maximum(
                    second_size,
                    2.0 * c[:, 1:].abs() + 6.0 * local[:, 1:].abs() * d[:, 1:].abs(),
                )
                epsilon = 64.0 * torch.finfo(torch.float64).eps
                span = (torch.diff(times, dim=1) * self._speed_scale)[..., None]
                change = acceleration * span
                terminal = (initial + change).clamp_min(0)
                vb, vc, vd = (
                    normalized_coefficient(value, velocity=True) for value in raw
                )
                velocity_first = vb + local * (2.0 * vc + 3.0 * local * vd)
                vp0 = velocity_first[:, :-1]
                vp1 = vp0 + 0.5 * ds * (
                    2.0 * vc[:, :-1] + 6.0 * local[:, :-1] * vd[:, :-1]
                )
                vp2 = vb[:, :-1] + right_local * (
                    2.0 * vc[:, :-1] + 3.0 * right_local * vd[:, :-1]
                )
                quadratic = vp0 - 2.0 * vp1 + vp2
                ratio = (
                    (vp0 - vp1) / torch.where(quadratic != 0, quadratic, 1.0)
                ).clamp(0, 1)
                extremum = (
                    (1.0 - ratio).square() * vp0
                    + 2.0 * ratio * (1.0 - ratio) * vp1
                    + ratio.square() * vp2
                )
                velocity_maximum = torch.maximum(
                    torch.maximum(vp0.abs(), vp2.abs()),
                    torch.maximum(extremum.abs(), velocity_first[:, 1:].abs()),
                )
                velocity_size = torch.maximum(
                    vb[:, :-1].abs()
                    + extent
                    * (2.0 * vc[:, :-1].abs() + 3.0 * extent * vd[:, :-1].abs()),
                    vb[:, 1:].abs()
                    + local[:, 1:].abs()
                    * (
                        2.0 * vc[:, 1:].abs()
                        + 3.0 * local[:, 1:].abs() * vd[:, 1:].abs()
                    ),
                )
                speed_maximum = torch.maximum(initial, terminal)
                velocity_bound = velocity_maximum * speed_maximum + (
                    epsilon * velocity_size
                ) * (speed_maximum + initial.abs() + change.abs())
                _check(
                    torch.isfinite(velocity_bound) & (velocity_bound <= 1.0 + 1e-8),
                    "TOPPRA velocity limits are not numerically resolvable",
                )
                parameter_error = epsilon * (
                    grid[:, :-1, None].abs()
                    + (initial * span).abs()
                    + ((0.5 * change) * span).abs()
                )
                speed_error = (
                    (terminal.square() - x1).abs()
                    + epsilon * initial.square()
                    + epsilon * change.square()
                    + (parameter_error * acceleration.abs()) * 2.0
                )
                clock_error = torch.where(
                    second_size > 0, second_size * speed_error, 0.0
                )
                error = (
                    (epsilon * second_size) * torch.maximum(x0.abs(), x1.abs())
                    + (epsilon * first_size) * acceleration.abs()
                    + clock_error
                )
                bound = controls.abs().amax(dim=2) + error
                _check(
                    torch.isfinite(bound) & (bound <= 1.0 + 1e-8),
                    "TOPPRA acceleration limits are not numerically resolvable",
                )

    def _controllable_to_rest(
        self, a: torch.Tensor, b: torch.Tensor, caps: torch.Tensor, end: torch.Tensor
    ) -> list[torch.Tensor]:
        # Rest belongs to every controllable interval. For the remaining
        # upper bounds, b < 0 makes the affine update an addition of positive
        # terms, so precomputing intercepts/slopes does not cause cancellation.
        positive = a > 0
        denominator = torch.where(positive, a, 1.0)
        inverse = _divide(torch.ones_like(a), denominator)
        fixed = torch.where(positive & (b >= 0), inverse, torch.inf).min(-1).values
        caps = torch.minimum(caps, fixed).unbind(1)
        coupled = positive & (b < 0)
        intercept = torch.where(coupled, inverse, torch.inf).unbind(1)
        slope = _divide(-b, denominator)
        # A nonrepresentable positive intercept is already an infinite bound.
        # Its slope can be zero, avoiding inf * 0 at the terminal stop.
        slope = torch.where(coupled & torch.isfinite(inverse), slope, 0.0).unbind(1)
        upper = [end]
        for index in range(a.shape[1] - 1, -1, -1):
            reachable = intercept[index] + slope[index] * upper[-1][:, None]
            upper.append(torch.minimum(caps[index], reachable.min(-1).values))
        upper.reverse()
        return upper

    def sample(self, times) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample shared (T,) or per-path (B, T) times; clamp to each duration."""

        query = self._tensor(times)
        if query.ndim == 1:
            query = query.unsqueeze(0).expand(self._batch, -1)
        if query.ndim != 2 or query.shape[0] != self._batch:
            raise ValueError("sample times must have shape (T,) or (B, T)")
        _check(torch.isfinite(query), "sample times must be finite")
        clipped = torch.minimum(query.clamp_min(0), self._times[:, -1:])
        index = (
            torch.searchsorted(
                self._times.contiguous(), clipped.contiguous(), right=True
            )
            - 1
        ).clamp(0, self._u.shape[1] - 1)
        # Keep the path clock in normalized units too: squaring a physical
        # time can overflow even though the sampled motion is representable.
        elapsed = (clipped - self._times.gather(1, index)) * self._speed_scale
        accel = self._normalized_u.gather(1, index)
        initial = self._normalized_speeds.gather(1, index)
        s = (
            self._grid.gather(1, index)
            + initial * elapsed
            + (0.5 * accel * elapsed) * elapsed
        )
        s = torch.maximum(
            self._grid.gather(1, index),
            torch.minimum(s, self._grid.gather(1, index + 1)),
        )
        speed = (initial + accel * elapsed).clamp_min(0)
        q, first, second = self._evaluate(s)
        scale = self._speed_scale[..., None]
        if self._normalized_acceleration_constraints:
            physical_speed = speed[..., None] * scale
            physical_acceleration = (accel[..., None] * scale) * scale
            outputs = (
                q,
                first * physical_speed,
                (second * physical_speed) * physical_speed
                + first * physical_acceleration,
            )
        else:
            outputs = (
                q,
                (first * speed[..., None]) * scale,
                (
                    (second * speed[..., None].square() + first * accel[..., None])
                    * scale
                )
                * scale,
            )
        converted = tuple(self._output(value).to(self._dtype) for value in outputs)
        _check(
            torch.stack([torch.isfinite(value).all() for value in converted]),
            "sampled motion is not representable in the waypoint dtype",
        )
        return converted

    def sample_uniform(self, count: int = 200) -> tuple[torch.Tensor, ...]:
        """Return float64 times and waypoint-dtype motion, retaining gradients."""

        count = _count(count, "count")
        fraction = torch.linspace(0, 1, count, dtype=torch.float64, device=self._device)
        times = self._times[:, -1:] * fraction
        return (self._output(times), *self.sample(times))


def retime_path_torch(
    waypoints: torch.Tensor, max_velocity, max_acceleration, **kwargs
) -> TorchToppraTrajectory:
    """Construct a differentiable trajectory without changing NumPy dispatch."""

    return TorchToppraTrajectory(waypoints, max_velocity, max_acceleration, **kwargs)
