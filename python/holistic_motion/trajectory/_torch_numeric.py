"""Autograd-safe arithmetic for tensor retiming kernels."""

from __future__ import annotations

import torch


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


def _check(condition: torch.Tensor, message: str) -> None:
    if not bool(condition.all()):
        raise ValueError(message)


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))
