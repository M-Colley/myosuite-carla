"""Calibrate physical MuJoCo signals into vehicle commands."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray

from .controls import VehicleCommand

Reader = Callable[[mujoco.MjData], float]


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _shape_parameters(deadzone: float, exponent: float) -> None:
    if not 0.0 <= _finite(deadzone, "deadzone") < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    if _finite(exponent, "exponent") <= 0.0:
        raise ValueError("exponent must be positive")


def _shape(value: float, deadzone: float, exponent: float) -> float:
    return (max(0.0, min(1.0, value) - deadzone) / (1.0 - deadzone)) ** exponent


class Calibration(Protocol):
    """Convert a finite scalar signal to a calibrated channel value."""

    def __call__(self, value: float) -> float:
        """Calibrate one reading."""
        ...


@dataclass(frozen=True)
class IdentityCalibration:
    """Preserve a finite value, including custom channels and integer gears."""

    def __call__(self, value: float) -> float:
        """Return a finite reading unchanged."""
        return _finite(value, "signal")


@dataclass(frozen=True)
class UnipolarCalibration:
    """Map released and pressed endpoints to zero and one in either direction.

    Args:
        released: Raw reading with the control released.
        pressed: Raw reading with the control fully pressed.
        deadzone: Fraction of travel ignored near the released endpoint.
        exponent: Positive power applied after deadzone removal.
    """

    released: float
    pressed: float
    deadzone: float = 0.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        released = _finite(self.released, "released")
        pressed = _finite(self.pressed, "pressed")
        if released == pressed or not math.isfinite(pressed - released):
            raise ValueError("released and pressed must have a finite nonzero span")
        _shape_parameters(self.deadzone, self.exponent)

    def __call__(self, value: float) -> float:
        """Return a saturated, calibrated reading in [0, 1]."""
        value = _finite(value, "signal")
        return _shape(
            (value - self.released) / (self.pressed - self.released),
            self.deadzone,
            self.exponent,
        )


@dataclass(frozen=True)
class BipolarCalibration:
    """Map asymmetric lower, neutral and upper endpoints to [-1, 0, 1].

    Args:
        lower: Smallest raw reading, mapped to -1 before inversion.
        neutral: Raw reading mapped to zero.
        upper: Largest raw reading, mapped to +1 before inversion.
        deadzone: Fraction ignored on each side of neutral.
        exponent: Positive power applied separately to each side.
        invert: Reverse the final steering direction when true.
    """

    lower: float
    neutral: float
    upper: float
    deadzone: float = 0.0
    exponent: float = 1.0
    invert: bool = False

    def __post_init__(self) -> None:
        lower = _finite(self.lower, "lower")
        neutral = _finite(self.neutral, "neutral")
        upper = _finite(self.upper, "upper")
        if not lower < neutral < upper:
            raise ValueError("steering endpoints must satisfy lower < neutral < upper")
        if not math.isfinite(neutral - lower) or not math.isfinite(upper - neutral):
            raise ValueError("steering endpoint spans must be finite")
        if not isinstance(self.invert, bool):
            raise TypeError("invert must be a bool")
        _shape_parameters(self.deadzone, self.exponent)

    def __call__(self, value: float) -> float:
        """Return saturated steering in [-1, 1]."""
        value = _finite(value, "signal")
        if value < self.neutral:
            result = -_shape(
                (self.neutral - value) / (self.neutral - self.lower),
                self.deadzone,
                self.exponent,
            )
        else:
            result = _shape(
                (value - self.neutral) / (self.upper - self.neutral),
                self.deadzone,
                self.exponent,
            )
        return -result if self.invert else result


class SignalSource(Protocol):
    """Resolve model addresses once and return a per-step scalar reader."""

    def bind(self, model: mujoco.MjModel) -> Reader:
        """Build a reader for the given compiled model."""
        ...


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("MuJoCo object names must be nonempty strings")
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(
            f"No MuJoCo {kind.name.removeprefix('mjOBJ_').lower()} named {name!r}"
        )
    return index


@dataclass(frozen=True)
class JointPosition:
    """Read the qpos address of a named scalar hinge or slide joint."""

    name: str

    def bind(self, model: mujoco.MjModel) -> Reader:
        """Resolve a scalar joint, rejecting ball and free joints."""
        index = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, self.name)
        if int(model.jnt_type[index]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(
                f"Joint {self.name!r} must be a scalar hinge or slide joint"
            )
        address = int(model.jnt_qposadr[index])

        def read(data: mujoco.MjData) -> float:
            return float(data.qpos[address])

        return read


@dataclass(frozen=True)
class SensorValue:
    """Read one component of a named MuJoCo sensor."""

    name: str
    component: int = 0

    def bind(self, model: mujoco.MjModel) -> Reader:
        """Resolve a sensor and validate its component index."""
        index = _named_id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.name)
        if isinstance(self.component, bool) or not isinstance(self.component, Integral):
            raise TypeError("sensor component must be an integer")
        dimension = int(model.sensor_dim[index])
        if not 0 <= self.component < dimension:
            raise ValueError(
                f"Sensor {self.name!r} has {dimension} components; "
                f"component {self.component} is out of range"
            )
        address = int(model.sensor_adr[index]) + self.component

        def read(data: mujoco.MjData) -> float:
            return float(data.sensordata[address])

        return read


@dataclass(frozen=True)
class ContactForce:
    """Sum compressive contact normal forces between two named geoms in newtons.

    Read after ``mj_forward`` or completed physics stepping so constraint forces
    describe the current state. Geom order is irrelevant; tangential force and
    torque do not contribute.
    """

    geom1: str
    geom2: str

    def bind(self, model: mujoco.MjModel) -> Reader:
        """Resolve the geom pair and use ``mj_contactForce`` for each contact."""
        first = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, self.geom1)
        second = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, self.geom2)
        if first == second:
            raise ValueError("contact-force geoms must be distinct")

        def read(data: mujoco.MjData) -> float:
            force: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
            total = 0.0
            for index in range(data.ncon):
                contact = data.contact[index]
                if (int(contact.geom1), int(contact.geom2)) not in (
                    (first, second),
                    (second, first),
                ):
                    continue
                mujoco.mj_contactForce(model, data, index, force)
                normal = _finite(float(force[0]), "contact normal force")
                total += max(0.0, normal)
            return total

        return read


@dataclass(frozen=True)
class CallbackSource:
    """Read an application-defined scalar from ``callback(model, data)``."""

    callback: Callable[[mujoco.MjModel, mujoco.MjData], float]

    def bind(self, model: mujoco.MjModel) -> Reader:
        """Bind a callback to this compiled model."""
        if not callable(self.callback):
            raise TypeError("callback must be callable")

        def read(data: mujoco.MjData) -> float:
            return self.callback(model, data)

        return read


@dataclass(frozen=True)
class Binding:
    """Connect a named output channel to a physical signal and calibration."""

    channel: str
    source: SignalSource
    calibration: Calibration = IdentityCalibration()

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("channel must be a nonempty string")
        if not callable(getattr(self.source, "bind", None)):
            raise TypeError("source must provide bind(model)")
        if not callable(self.calibration):
            raise TypeError("calibration must be callable")


@dataclass(frozen=True)
class ControlSample:
    """Hold the validated command and all calibrated values before interlocks."""

    command: VehicleCommand
    values: dict[str, float]


class ControlMapper:
    """Read calibrated controls from one compiled MuJoCo model.

    Brake priority suppresses throttle whenever brake is positive. Boolean
    channels use a threshold of 0.5 and must be in [0, 1]. Gear must be an exact
    signed 32-bit integer. Other channels are preserved for application hooks.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        bindings: Iterable[Binding],
        brake_priority: bool = True,
    ) -> None:
        if not isinstance(brake_priority, bool):
            raise TypeError("brake_priority must be a bool")
        self.model = model
        self.bindings = tuple(bindings)
        self.brake_priority = brake_priority
        channels = [binding.channel for binding in self.bindings]
        if len(set(channels)) != len(channels):
            raise ValueError("each channel may have only one binding")
        self._readers = tuple(binding.source.bind(model) for binding in self.bindings)
        if not all(callable(reader) for reader in self._readers):
            raise TypeError("source.bind(model) must return a callable reader")

    def sample(self, data: mujoco.MjData) -> ControlSample:
        """Read one finite sample and validate standard vehicle channels."""
        if data.model is not self.model:
            raise ValueError("MuJoCo data must belong to the mapper's compiled model")
        values = {
            binding.channel: _finite(
                binding.calibration(_finite(reader(data), binding.channel)),
                f"calibrated {binding.channel}",
            )
            for binding, reader in zip(self.bindings, self._readers)
        }
        binary: dict[str, bool] = {}
        for name in ("hand_brake", "reverse", "manual_gear_shift"):
            value = values.get(name, 0.0)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be calibrated to [0, 1]")
            binary[name] = value >= 0.5
        gear = values.get("gear", 0.0)
        if not gear.is_integer():
            raise ValueError("gear must be calibrated to an exact integer")
        command = VehicleCommand(
            throttle=values.get("throttle", 0.0),
            steer=values.get("steer", 0.0),
            brake=values.get("brake", 0.0),
            hand_brake=binary["hand_brake"],
            reverse=binary["reverse"],
            manual_gear_shift=binary["manual_gear_shift"],
            gear=int(gear),
        )
        if self.brake_priority and command.brake > 0.0:
            command = replace(command, throttle=0.0)
        return ControlSample(command, values)

    def reset(self) -> None:
        """Reset the stateless mapper; each sample always reads current data."""
