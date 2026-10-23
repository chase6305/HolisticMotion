"""Linear-storage spline solve with an implicit PyTorch adjoint."""

from __future__ import annotations

import torch


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
