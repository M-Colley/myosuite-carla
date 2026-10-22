"""Exercise physical pedal and wheel contacts without a CARLA server.

Install MyoSuite and myosuite-carla, then run this file. The elbow's muscles
coexist with the cockpit; a scripted mocap probe demonstrates contact mechanics.
The probe trajectory is a fixture, not a trained muscle policy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import mujoco
import numpy as np
from myosuite.core.model_builder import ModelBuilder

from myosuite_carla import CockpitConfig, ControlMapper, add_cockpit, cockpit_mapper


def add_probe(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add a collidable mocap sphere as a reproducible physical test fixture."""
    probe = spec.worldbody.add_body(name="demo_probe", mocap=True, pos=(4, 4, 4))
    probe.add_geom(
        name="demo_probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.02, 0, 0),
        mass=0.1,
        friction=(1, 0.005, 0.0001),
        contype=1,
        conaffinity=1,
    )
    return spec


def contact_phase(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: ControlMapper,
    config: CockpitConfig,
    control: str,
    positions: Iterable[np.ndarray],
) -> dict[str, Any]:
    """Move the probe, returning the strongest physically generated sample."""
    channel = "steer" if control == "steering" else control
    best = mapper.sample(data)
    peak_touch = 0.0
    peak_contacts = 0
    for position in positions:
        data.mocap_pos[0] = position
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        sample = mapper.sample(data)
        if abs(sample.values[channel]) > abs(best.values[channel]):
            best = sample
        peak_touch = max(
            peak_touch, float(data.sensor(config.names.touch_sensor(control)).data[0])
        )
        peak_contacts = max(peak_contacts, data.ncon)
    return {
        "phase": control,
        "command": best.command.as_dict(),
        "values": best.values,
        "peak_touch_newtons": peak_touch,
        "peak_contact_count": peak_contacts,
    }


def release(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    """Withdraw the probe and let the passive centering springs settle."""
    data.mocap_pos[0] = (4, 4, 4)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def main() -> None:
    """Build a muscle model and print finite contact-generated JSON samples."""
    config = CockpitConfig()
    builder = ModelBuilder().place_fragment("elbow", pos=(3, 0, 1))
    builder = add_cockpit(builder.set_timestep(0.001), config)
    model, _ = builder.apply_transform(add_probe).build()
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.05
    mujoco.mj_forward(model, data)
    mapper = cockpit_mapper(model, config)
    pedal_center = np.asarray(config.throttle.position)
    throttle = contact_phase(
        model,
        data,
        mapper,
        config,
        "throttle",
        (pedal_center + (0, offset, 0) for offset in np.linspace(-0.08, 0.015, 500)),
    )
    release(model, data, 1200)
    wheel_center = np.asarray(config.steering.position)
    steering = contact_phase(
        model,
        data,
        mapper,
        config,
        "steering",
        (wheel_center + (0.1, 0, height) for height in np.linspace(0.07, -0.09, 600)),
    )
    release(model, data, 3000)
    released = mapper.sample(data)
    print(
        json.dumps(
            {
                "driver": "scripted_contact_probe",
                "muscle_actuators": int(
                    np.count_nonzero(
                        model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
                    )
                ),
                "phases": [
                    throttle,
                    steering,
                    {
                        "phase": "released",
                        "command": released.command.as_dict(),
                        "values": released.values,
                    },
                ],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
