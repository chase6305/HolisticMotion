"""Differentiable, batched TOPPRA on PyTorch CPU and CUDA devices.

Active constraints, spline segment lookups, and grid topology are discrete.
Gradients describe the selected smooth branch, not switches between branches.
Numerical work stays on the input device and uses float64; only validation
reductions synchronize CUDA with the host. No NumPy solver is used.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Keep zero adjoints zero for inactive, ill-conditioned LP bounds."""

    @staticmethod
    def forward(
        ctx, numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        result = numerator / denominator
        ctx.save_for_backward(denominator, result)
        return result

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denominator, result = ctx.saved_tensors
        # Scale the incoming adjoint before multiplying by the quotient. The
        # alternative -(result / denominator) * gradient can overflow even
        # when the incoming adjoint is exactly zero. Infinite candidate bounds
        # are never selected by a finite feasible solution, so their quotient
        # factor can safely be zero for this unused denominator adjoint.
        numerator_gradient = gradient / denominator
        denominator_gradient = -numerator_gradient * torch.nan_to_num(
            result, nan=0.0, posinf=0.0, neginf=0.0
        )
        return numerator_gradient, denominator_gradient


def _divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and (
        numerator.requires_grad or denominator.requires_grad
    ):
        return _StableDivide.apply(numerator, denominator)
    return numerator / denominator


def _bounds(
    coefficient: torch.Tensor, rhs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = _divide(rhs, torch.where(coefficient != 0.0, coefficient, 1.0))
    lower = torch.where(coefficient < 0.0, ratio, -torch.inf).max(-1).values
    upper = torch.where(coefficient > 0.0, ratio, torch.inf).min(-1).values
    return lower, upper


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
    pairs = torch.triu_indices(count, count, offset=1, device=a.device)
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
    choice = torch.cat(choices)
    is_pair = choice < pair_count
    lookup = choice.clamp_max(pair_count - 1)
    left = torch.where(is_pair, pairs[0, lookup], choice - pair_count)
    right = torch.where(is_pair, pairs[1, lookup], count)
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
        _check(torch.isfinite(waypoints), "waypoints must be finite")
        self._single = waypoints.ndim == 2
        self._dtype, self._device = waypoints.dtype, waypoints.device
        points = waypoints.to(dtype=torch.float64).clone()
        self._points = points.unsqueeze(0) if self._single else points
        self._batch, count, self.dof = self._points.shape
        self._velocity = self._limits(max_velocity, "max_velocity")
        self._acceleration = self._limits(max_acceleration, "max_acceleration")
        start = self._boundary(start_path_velocity)
        end = self._boundary(end_path_velocity)
        lengths = torch.linalg.vector_norm(torch.diff(self._points, dim=1), dim=-1)
        _check(lengths > 1e-12, "consecutive waypoints must be distinct")
        zero = torch.zeros((self._batch, 1), dtype=torch.float64, device=self._device)
        cumulative = torch.cat((zero, lengths.cumsum(1)), dim=1)
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
        _, first, second = self._evaluate(self._grid)
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
        normalized_speeds = _sqrt_speed(x)
        self._speeds = normalized_speeds * speed_scale[:, None]
        ds = torch.diff(self._grid, dim=1)
        self._u = (
            torch.diff(x, dim=1) / (2.0 * ds) * speed_scale[:, None]
        ) * speed_scale[:, None]
        denominator = normalized_speeds[:, :-1] + normalized_speeds[:, 1:]
        _check(denominator > 0.0, "path contains an interval with zero reachable speed")
        self._times = torch.cat(
            (zero, (2.0 * ds / denominator / speed_scale[:, None]).cumsum(1)), dim=1
        )
        _check(
            torch.isfinite(self._times[:, 1:]) & (torch.diff(self._times, dim=1) > 0),
            "path timing is not representable",
        )
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
        _check(
            torch.isfinite(limits) & (limits > 0),
            f"{name} must be finite and strictly positive",
        )
        return limits.expand(self._batch, self.dof).clone()

    def _boundary(self, value) -> torch.Tensor:
        speed = self._tensor(value)
        if speed.shape not in ((), (self._batch,)):
            raise ValueError("boundary path velocities must be scalar or (B,)")
        _check(
            torch.isfinite(speed) & (speed >= 0),
            "boundary path velocities must be finite and non-negative",
        )
        return speed.expand(self._batch).clone()

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
        slopes = torch.diff(self._points, dim=1) / h[..., None]
        zero = torch.zeros_like(self._points[:, 0])
        diagonal, rhs = [], []
        for index in range(h.shape[1] - 1):
            d = 2.0 * (h[:, index] + h[:, index + 1])
            r = 6.0 * (slopes[:, index + 1] - slopes[:, index])
            if index:
                factor = h[:, index] / diagonal[-1]
                d = d - factor * h[:, index]
                r = r - factor[:, None] * rhs[-1]
            diagonal.append(d)
            rhs.append(r)
        interior = []
        next_value = zero
        for index in range(len(diagonal) - 1, -1, -1):
            next_value = (rhs[index] - h[:, index + 1, None] * next_value) / diagonal[
                index
            ][:, None]
            interior.append(next_value)
        second = torch.stack([zero, *reversed(interior), zero], dim=1)
        self._a = self._points[:, :-1]
        self._b = slopes - h[..., None] * (2.0 * second[:, :-1] + second[:, 1:]) / 6.0
        self._c = second[:, :-1] / 2.0
        self._d = torch.diff(second, dim=1) / (6.0 * h[..., None])

    def _evaluate(self, s: torch.Tensor) -> tuple[torch.Tensor, ...]:
        index = (
            torch.searchsorted(self._knots.contiguous(), s.contiguous(), right=True) - 1
        ).clamp(0, self._a.shape[1] - 1)
        delta = (s - self._knots.gather(1, index))[..., None]
        a, b, c, d = (
            _gather(value, index) for value in (self._a, self._b, self._c, self._d)
        )
        return (
            a + delta * (b + delta * (c + delta * d)),
            b + delta * (2.0 * c + 3.0 * delta * d),
            2.0 * c + 6.0 * delta * d,
        )

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
        # Solve in dimensionless speed units. Normalize limits *before*
        # division so both elimination products and their adjoints remain
        # representable when the caller changes time units. This is only a
        # change of variables; its derivative cancels identically. Detach the
        # arbitrary scale selection, keeping gradients through the limits.
        with torch.no_grad():
            velocity_scale = (derivative.abs() / self._velocity[:, None, None]).amax(
                dim=(1, 2, 3)
            )
            acceleration_scale = (
                torch.maximum(ax.abs(), ay.abs()).sqrt()
                / self._acceleration[:, None, None].sqrt()
            ).amax(dim=(1, 2, 3))
            speed_scale = torch.maximum(velocity_scale, acceleration_scale).reciprocal()
        _check(
            torch.isfinite(speed_scale) & (speed_scale > 0),
            "path speed scale is not representable",
        )
        velocity = self._velocity / speed_scale[:, None]
        acceleration = (self._acceleration / speed_scale[:, None]) / speed_scale[
            :, None
        ]
        ax = ax / acceleration[:, None, None]
        ay = ay / acceleration[:, None, None]
        p0, p1, p2 = (derivative / velocity[:, None, None]).unbind(2)
        quadratic = p0 - 2.0 * p1 + p2
        ratio = torch.where(
            quadratic != 0, (p0 - p1) / torch.where(quadratic != 0, quadratic, 1.0), 0.0
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
        caps = _projection_caps(a, b)
        # Unbind once: repeated a[:, i] indexing creates a full-sized zero
        # gradient buffer for every interval during backward. Unbind collects
        # the interval adjoints in one stack instead.
        aa_intervals, bb_intervals = a.unbind(1), b.unbind(1)
        cap_intervals = caps.unbind(1)
        lower = None
        if bool((end == 0).all()):
            upper = self._controllable_to_rest(a, b, caps, end)
        else:
            lower, upper = [end], [end]
            for index in range(a.shape[1] - 1, -1, -1):
                aa, bb = aa_intervals[index], bb_intervals[index]
                rhs = 1.0 - bb * torch.where(
                    bb > 0, lower[-1][:, None], upper[-1][:, None]
                )
                lo, hi = _bounds(aa, rhs)
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
        values = [start]
        positive = b > 0
        denominators = torch.where(positive, b, 1.0).unbind(1)
        positive = positive.unbind(1)
        for index in range(a.shape[1]):
            product = aa_intervals[index] * values[-1][:, None]
            rounding = _ROUNDING_EPS * product.abs().clamp_min(1.0)
            ratio = _divide(1.0 - product + rounding, denominators[index])
            high = torch.where(positive[index], ratio, torch.inf).min(-1).values
            value = torch.minimum(high, upper[index + 1])
            values.append(
                value.clamp_min(0)
                if lower is None
                else torch.maximum(lower[index + 1], value)
            )
        x = torch.stack(values, dim=1)
        with torch.no_grad():
            left, right = a * x[:, :-1, None], b * x[:, 1:, None]
            scale = torch.maximum(left.abs(), right.abs()).clamp_min(1.0)
            _check(
                torch.isfinite(x) & (x >= 0),
                "path squared speeds must be finite and non-negative",
            )
            _check(
                torch.isfinite(left)
                & torch.isfinite(right)
                & (left + right - 1.0 <= 1e-10 * scale),
                "TOPPRA forward pass violates interval constraints",
            )
        return x

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
        elapsed = clipped - self._times.gather(1, index)
        accel, initial = self._u.gather(1, index), self._speeds.gather(1, index)
        s = (
            self._grid.gather(1, index)
            + initial * elapsed
            + 0.5 * accel * elapsed.square()
        )
        s = torch.maximum(
            self._grid.gather(1, index),
            torch.minimum(s, self._grid.gather(1, index + 1)),
        )
        speed = (initial + accel * elapsed).clamp_min(0)
        q, first, second = self._evaluate(s)
        outputs = (
            q,
            first * speed[..., None],
            second * speed[..., None].square() + first * accel[..., None],
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
