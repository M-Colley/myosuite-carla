"""Default bindings for the composable cockpit."""

from __future__ import annotations

import mujoco

from .cockpit import CockpitConfig
from .inputs import (
    Binding,
    BipolarCalibration,
    ControlMapper,
    SensorValue,
    UnipolarCalibration,
)


def cockpit_mapper(
    model: mujoco.MjModel,
    config: CockpitConfig | None = None,
    *,
    deadzone: float = 0.02,
    brake_priority: bool = True,
) -> ControlMapper:
    """Bind the cockpit's actual travel and angle sensors to vehicle controls.

    Args:
        model: Compiled model containing the cockpit.
        config: Same configuration used by ``add_cockpit``.
        deadzone: Fraction of each control's travel ignored near neutral.
        brake_priority: Suppress throttle while the brake is pressed.
    """
    config = config or CockpitConfig()
    names = config.names
    return ControlMapper(
        model,
        (
            Binding(
                "throttle",
                SensorValue(names.throttle_position),
                UnipolarCalibration(0, config.throttle.travel, deadzone=deadzone),
            ),
            Binding(
                "brake",
                SensorValue(names.brake_position),
                UnipolarCalibration(0, config.brake.travel, deadzone=deadzone),
            ),
            Binding(
                "steer",
                SensorValue(names.steering_position),
                BipolarCalibration(
                    -config.steering.angle_limit,
                    0,
                    config.steering.angle_limit,
                    deadzone=deadzone,
                ),
            ),
        ),
        brake_priority=brake_priority,
    )
