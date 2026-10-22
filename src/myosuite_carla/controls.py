"""Validated vehicle commands without importing CARLA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class VehicleCommand:
    """Hold CARLA vehicle-control fields with explicit range validation."""

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    manual_gear_shift: bool = False
    gear: int = 0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("throttle", 0.0, 1.0),
            ("steer", -1.0, 1.0),
            ("brake", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
            object.__setattr__(self, name, value)
        for name in ("hand_brake", "reverse", "manual_gear_shift"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if isinstance(self.gear, bool) or not isinstance(self.gear, Integral):
            raise TypeError("gear must be an integer")
        if not -(2**31) <= self.gear < 2**31:
            raise ValueError("gear must fit CARLA's signed 32-bit integer field")
        object.__setattr__(self, "gear", int(self.gear))

    def as_dict(self) -> dict[str, float | bool | int]:
        """Return keyword arguments for ``carla.VehicleControl``."""
        return asdict(self)
