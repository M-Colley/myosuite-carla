"""Cockpit configuration and mapper calibration remain aligned."""

from __future__ import annotations

import mujoco
import pytest

from myosuite_carla import (
    CockpitConfig,
    PedalConfig,
    SteeringConfig,
    add_cockpit,
    cockpit_mapper,
)


def test_custom_cockpit_travel_and_steering_calibration() -> None:
    builder_class = pytest.importorskip("myosuite.core.model_builder").ModelBuilder
    config = CockpitConfig(
        prefix="driver_",
        throttle=PedalConfig(travel=0.1),
        brake=PedalConfig(travel=0.08),
        steering=SteeringConfig(angle_limit=2),
    )
    model, _ = add_cockpit(builder_class(), config).build()
    data = mujoco.MjData(model)
    data.joint(config.names.throttle_joint).qpos[0] = 0.05
    data.joint(config.names.brake_joint).qpos[0] = 0.04
    data.joint(config.names.steering_joint).qpos[0] = -1
    mujoco.mj_forward(model, data)
    sample = cockpit_mapper(model, config, deadzone=0).sample(data)
    assert sample.values == {"throttle": 0.5, "brake": 0.5, "steer": -0.5}
    assert sample.command.throttle == 0
    assert sample.command.brake == 0.5
    assert sample.command.steer == -0.5
    with pytest.raises(ValueError, match="compiled model"):
        cockpit_mapper(model, config).sample(mujoco.MjData(builder_class().build()[0]))
