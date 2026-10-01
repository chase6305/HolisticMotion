"""Path timing algorithms and sampled trajectory results."""

from .toppra import ToppraResult, ToppraTrajectory, retime_path

__all__ = [
    "ToppraResult",
    "ToppraTrajectory",
    "TorchToppraResult",
    "TorchToppraTrajectory",
    "retime_path",
    "retime_path_torch",
]


def __getattr__(name):
    if name in {"TorchToppraResult", "TorchToppraTrajectory", "retime_path_torch"}:
        from . import toppra_torch

        value = getattr(toppra_torch, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
