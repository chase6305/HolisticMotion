"""High-level tools, loaded independently when first requested."""

__all__ = ["kinematics", "retargeting", "simulation"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    path = (
        "holistic_motion.kinematics"
        if name == "kinematics"
        else f"{__name__}.{name}"
    )
    module = import_module(path)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
