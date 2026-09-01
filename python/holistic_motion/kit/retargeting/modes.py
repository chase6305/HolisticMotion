"""Mode definitions for pose retargeting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional, Union


class RetargetingMode(str, Enum):
    """Built-in sets of controlled bodies and movable joint groups."""

    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    DUAL_ARM = "dual_arm"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    DUAL_LEG = "dual_leg"
    WHOLE_BODY = "whole_body"
    FULL_BODY = "full_body"


@dataclass(frozen=True)
class RetargetingModeSpec:
    """Declarative description of a retargeting mode."""

    targets: tuple[str, ...]
    active_joint_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.targets, str) or isinstance(self.active_joint_groups, str):
            raise TypeError(
                "targets and active_joint_groups must be sequences of names"
            )
        targets = tuple(self.targets)
        groups = tuple(self.active_joint_groups)
        if not targets:
            raise ValueError("a retargeting mode must contain at least one target")
        if not groups:
            raise ValueError(
                "a retargeting mode must activate at least one joint group"
            )
        if any(not isinstance(name, str) or not name for name in targets):
            raise ValueError("retargeting target names must be non-empty strings")
        if any(not isinstance(name, str) or not name for name in groups):
            raise ValueError("joint group names must be non-empty strings")
        if len(set(targets)) != len(targets):
            raise ValueError("retargeting target names must be unique")
        if len(set(groups)) != len(groups):
            raise ValueError("active joint group names must be unique")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "active_joint_groups", groups)


DEFAULT_MODE_SPECS = MappingProxyType(
    {
        RetargetingMode.LEFT_ARM: RetargetingModeSpec(("left_hand",), ("left_arm",)),
        RetargetingMode.RIGHT_ARM: RetargetingModeSpec(("right_hand",), ("right_arm",)),
        RetargetingMode.DUAL_ARM: RetargetingModeSpec(
            ("left_hand", "right_hand"), ("left_arm", "right_arm")
        ),
        RetargetingMode.LEFT_LEG: RetargetingModeSpec(("left_foot",), ("left_leg",)),
        RetargetingMode.RIGHT_LEG: RetargetingModeSpec(("right_foot",), ("right_leg",)),
        RetargetingMode.DUAL_LEG: RetargetingModeSpec(
            ("left_foot", "right_foot"), ("left_leg", "right_leg")
        ),
        RetargetingMode.WHOLE_BODY: RetargetingModeSpec(
            ("left_hand", "right_hand", "head"), ("whole_body",)
        ),
        RetargetingMode.FULL_BODY: RetargetingModeSpec(
            (
                "left_hand",
                "right_hand",
                "left_foot",
                "right_foot",
                "head",
                "pelvis",
            ),
            ("whole_body",),
        ),
    }
)


class RetargetingModeManager:
    """Validate, select, and describe retargeting modes."""

    def __init__(
        self,
        modes: Optional[
            Mapping[Union[RetargetingMode, str], RetargetingModeSpec]
        ] = None,
        initial_mode: Union[RetargetingMode, str] = RetargetingMode.DUAL_ARM,
    ) -> None:
        source = DEFAULT_MODE_SPECS if modes is None else modes
        if not isinstance(source, Mapping):
            raise TypeError("modes must be a mapping")
        normalized = {}
        for key, value in source.items():
            mode = RetargetingMode(key)
            if mode in normalized:
                raise ValueError(f"duplicate retargeting mode {mode.value!r}")
            normalized[mode] = value
        self._modes = normalized
        if not self._modes:
            raise ValueError("at least one retargeting mode is required")
        if any(
            not isinstance(spec, RetargetingModeSpec) for spec in self._modes.values()
        ):
            raise TypeError("each retargeting mode must map to RetargetingModeSpec")
        self._mode = self._coerce(initial_mode)

    @property
    def mode(self) -> RetargetingMode:
        return self._mode

    @property
    def spec(self) -> RetargetingModeSpec:
        return self._modes[self._mode]

    @property
    def available_modes(self) -> tuple[RetargetingMode, ...]:
        return tuple(self._modes)

    @property
    def mode_specs(self) -> Mapping[RetargetingMode, RetargetingModeSpec]:
        return MappingProxyType(self._modes.copy())

    def set_mode(self, mode: Union[RetargetingMode, str]) -> RetargetingModeSpec:
        self._mode = self._coerce(mode)
        return self.spec

    def cycle(self) -> RetargetingModeSpec:
        modes = self.available_modes
        self._mode = modes[(modes.index(self._mode) + 1) % len(modes)]
        return self.spec

    def validate_targets(self, names: Iterable[str]) -> None:
        if isinstance(names, str):
            raise TypeError("target names must be an iterable of names, not a string")
        try:
            provided = set(names)
        except TypeError as error:
            raise TypeError("target names must be an iterable of names") from error
        if any(not isinstance(name, str) or not name for name in provided):
            raise ValueError("target names must be non-empty strings")
        missing = set(self.spec.targets).difference(provided)
        if missing:
            raise ValueError(
                f"mode '{self.mode.value}' requires targets: {sorted(missing)}"
            )

    def _coerce(self, mode: Union[RetargetingMode, str]) -> RetargetingMode:
        try:
            value = RetargetingMode(mode)
        except ValueError as error:
            choices = ", ".join(item.value for item in self._modes)
            raise ValueError(
                f"unknown retargeting mode {mode!r}; choose {choices}"
            ) from error
        if value not in self._modes:
            raise ValueError(f"retargeting mode {value.value!r} is not configured")
        return value
