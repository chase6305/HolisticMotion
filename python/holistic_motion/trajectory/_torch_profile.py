"""Device-local speed and physical-limit certificates for Torch retiming."""

from __future__ import annotations

import torch

from ._torch_numeric import _check, _gather


def _validate_speed_profile(
    trajectory, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
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
        if trajectory._limit_unit_overflow is not None:
            conditioned = conditioned | trajectory._limit_unit_overflow
        valid_profile, conditioned = torch.stack((valid, conditioned)).tolist()
        if not valid_profile:
            raise ValueError("TOPPRA forward pass violates interval constraints")
        trajectory._physical_validation_needed = conditioned


def _validate_physical_limits(trajectory) -> None:
    """Certify conditioned profiles without subtracting large LP products."""

    with torch.no_grad():
        speed_mantissa, speed_exponent = torch.frexp(trajectory._speed_scale[..., None])
        limit_mantissa, limit_exponent = torch.frexp(trajectory._acceleration[:, None])
        velocity_mantissa, velocity_exponent = torch.frexp(
            trajectory._velocity[:, None]
        )

        def normalized_coefficient(value, velocity=False):
            if trajectory._coefficient_scale is not None:
                value = value * trajectory._coefficient_scale
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
        chunk = max(1, (1 << 17) // (trajectory._batch * trajectory.dof))
        for start in range(0, trajectory._normalized_u.shape[1], chunk):
            stop = min(start + chunk, trajectory._normalized_u.shape[1])
            grid = trajectory._grid[:, start : stop + 1]
            times = trajectory._times[:, start : stop + 1]
            index = (
                torch.searchsorted(
                    trajectory._knots.contiguous(), grid.contiguous(), right=True
                )
                - 1
            ).clamp(0, trajectory._a.shape[1] - 1)
            local = (grid - trajectory._knots.gather(1, index))[..., None]
            raw = tuple(
                _gather(value, index)
                for value in (trajectory._b, trajectory._c, trajectory._d)
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
            acceleration = trajectory._normalized_u[:, start:stop, None]
            initial = trajectory._normalized_speeds[:, start:stop, None]
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
                * (2.0 * c[:, 1:].abs() + 3.0 * local[:, 1:].abs() * d[:, 1:].abs()),
            )
            second_size = torch.maximum(
                second_size,
                2.0 * c[:, 1:].abs() + 6.0 * local[:, 1:].abs() * d[:, 1:].abs(),
            )
            epsilon = 64.0 * torch.finfo(torch.float64).eps
            span = (torch.diff(times, dim=1) * trajectory._speed_scale)[..., None]
            change = acceleration * span
            terminal = (initial + change).clamp_min(0)
            vb, vc, vd = (normalized_coefficient(value, velocity=True) for value in raw)
            velocity_first = vb + local * (2.0 * vc + 3.0 * local * vd)
            vp0 = velocity_first[:, :-1]
            vp1 = vp0 + 0.5 * ds * (2.0 * vc[:, :-1] + 6.0 * local[:, :-1] * vd[:, :-1])
            vp2 = vb[:, :-1] + right_local * (
                2.0 * vc[:, :-1] + 3.0 * right_local * vd[:, :-1]
            )
            quadratic = vp0 - 2.0 * vp1 + vp2
            ratio = ((vp0 - vp1) / torch.where(quadratic != 0, quadratic, 1.0)).clamp(
                0, 1
            )
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
                + extent * (2.0 * vc[:, :-1].abs() + 3.0 * extent * vd[:, :-1].abs()),
                vb[:, 1:].abs()
                + local[:, 1:].abs()
                * (2.0 * vc[:, 1:].abs() + 3.0 * local[:, 1:].abs() * vd[:, 1:].abs()),
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
            clock_error = torch.where(second_size > 0, second_size * speed_error, 0.0)
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
