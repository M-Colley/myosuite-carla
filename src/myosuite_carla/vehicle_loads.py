"""Vehicle-motion loads on a seated MuJoCo body, from CARLA telemetry.

The cockpit model is fixed in the vehicle, so MuJoCo's world frame is the
vehicle frame: x points to the driver's left, y backwards and z up. In that
non-inertial frame each body segment of mass m with centre of mass r (from the
vehicle reference point) carries the d'Alembert load

    F = m (g - a - alpha x r - omega x (omega x r)),

where g is gravity, a the vehicle reference point's acceleration, omega and
alpha the vehicle's angular velocity and acceleration, all expressed in the
vehicle frame. Coriolis forces and segment rotational inertia are neglected.

CARLA reports world-frame vectors in Unreal Engine's left-handed frame (x
forward, y right, z up) with rotations as (pitch, yaw, roll) in degrees and
angular velocity in degrees per second. These helpers convert them; nothing
here imports CARLA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

GRAVITY = 9.81
Vector = tuple[float, float, float]


def _vector(value: Sequence[float], name: str) -> Vector:
    if len(value) != 3:
        raise ValueError(f"{name} must have three components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def carla_axes(rotation: Sequence[float]) -> NDArray[np.float64]:
    """Return the vehicle's forward, right and up unit vectors in CARLA's world.

    Args:
        rotation: ``(pitch, yaw, roll)`` in degrees, Unreal Engine convention.

    Returns:
        A 3x3 array whose rows are the forward, right and up axes.
    """
    pitch, yaw, roll = (math.radians(angle) for angle in _vector(rotation, "rotation"))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cp * cy, cp * sy, sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
            [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
        ]
    )


def carla_vector_to_cockpit(vector: Sequence[float]) -> NDArray[np.float64]:
    """Map a vehicle-frame (forward, right, up) vector to the cockpit frame."""
    forward, right, up = _vector(vector, "vector")
    return np.array([-right, -forward, up])


def carla_rate_to_cockpit(rate: Sequence[float]) -> NDArray[np.float64]:
    """Map a vehicle-frame left-handed angular rate to the right-handed cockpit.

    A left-handed rate (about forward, right, up) becomes (-f, r, -u) about
    (forward, left, up) because pseudovectors change sign under the handedness
    flip, then (left, -forward, up) components in the cockpit frame.
    """
    forward, right, up = _vector(rate, "rate")
    return np.array([right, forward, -up])


@dataclass(frozen=True)
class VehicleMotion:
    """Vehicle kinematics for one control period, in CARLA's world frame.

    Args:
        acceleration: Reference-point acceleration in m/s^2.
        angular_velocity: Angular velocity in degrees per second.
        rotation: ``(pitch, yaw, roll)`` in degrees.
        angular_acceleration: Angular acceleration in degrees per second^2.
    """

    acceleration: Vector = (0.0, 0.0, 0.0)
    angular_velocity: Vector = (0.0, 0.0, 0.0)
    rotation: Vector = (0.0, 0.0, 0.0)
    angular_acceleration: Vector = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in (
            "acceleration",
            "angular_velocity",
            "rotation",
            "angular_acceleration",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))

    @classmethod
    def from_telemetry(
        cls, telemetry: Any, previous: Any | None = None, dt: float | None = None
    ) -> VehicleMotion:
        """Build motion from a ``VehicleTelemetry``; differentiate the rate if given."""
        alpha: Vector = (0.0, 0.0, 0.0)
        if previous is not None:
            if dt is None or not dt > 0:
                raise ValueError("dt must be positive when previous telemetry is given")
            alpha = tuple(  # type: ignore[assignment]
                (now - before) / dt
                for now, before in zip(
                    telemetry.angular_velocity, previous.angular_velocity
                )
            )
        return cls(
            acceleration=telemetry.acceleration,
            angular_velocity=telemetry.angular_velocity,
            rotation=telemetry.rotation,
            angular_acceleration=alpha,
        )

    def in_cockpit(self) -> dict[str, NDArray[np.float64]]:
        """Return gravity, acceleration, rate (rad/s) and rate change in the cockpit."""
        axes = carla_axes(self.rotation)

        def project(vector: Sequence[float]) -> NDArray[np.float64]:
            return np.asarray(axes @ np.asarray(vector, dtype=float))

        gravity = project((0.0, 0.0, -GRAVITY))
        return {
            "gravity": carla_vector_to_cockpit(tuple(gravity)),
            "acceleration": carla_vector_to_cockpit(tuple(project(self.acceleration))),
            "angular_velocity": np.radians(
                carla_rate_to_cockpit(tuple(project(self.angular_velocity)))
            ),
            "angular_acceleration": np.radians(
                carla_rate_to_cockpit(tuple(project(self.angular_acceleration)))
            ),
            "lateral_right_acceleration": np.array([project(self.acceleration)[1]]),
        }


class InertialLoads:
    """Compute per-body d'Alembert forces for selected bodies of a model.

    Args:
        model: Compiled model whose world frame is the vehicle frame.
        bodies: Body ids that carry loads (typically the human, not mechanisms).
        origin: Vehicle reference point in the model's world frame, metres.
    """

    def __init__(
        self, model: mujoco.MjModel, bodies: Sequence[int], origin: Sequence[float]
    ) -> None:
        self.bodies: NDArray[np.int_] = np.asarray(
            sorted(set(int(body) for body in bodies)), dtype=int
        )
        self.mass = np.asarray(model.body_mass[self.bodies], dtype=float)
        self.origin: NDArray[np.float64] = np.asarray(
            _vector(origin, "origin"), dtype=float
        )
        self._motion: VehicleMotion | None = None
        self._frame: dict[str, NDArray[np.float64]] = {}

    def forces(self, data: mujoco.MjData, motion: VehicleMotion) -> NDArray[np.float64]:
        """Return the (len(bodies), 3) world-frame forces at each centre of mass."""
        # Motion is constant over a control period; convert it once per motion.
        if motion is not self._motion:
            self._motion, self._frame = motion, motion.in_cockpit()
        frame = self._frame
        omega, alpha = frame["angular_velocity"], frame["angular_acceleration"]
        specific = frame["gravity"] - frame["acceleration"]
        if np.any(omega) or np.any(alpha):
            r = np.asarray(data.xipos[self.bodies]) - self.origin
            rotational = np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
            return np.asarray(self.mass[:, None] * (specific - rotational))
        return np.asarray(self.mass[:, None] * specific)

    def apply(self, data: mujoco.MjData, motion: VehicleMotion) -> None:
        """Write the forces into ``data.xfrc_applied``; other bodies are zeroed."""
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[self.bodies, :3] = self.forces(data, motion)
