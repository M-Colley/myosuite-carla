"""The full-body driver with extra embodied-ui devices, on any vehicle backend.

A hazard switch and a lamp are composed onto the steering-wheel hub of the
MyoSuite full-body driver. The reflex driver drives through its muscles; at
``--hazard-at`` seconds the switch is flipped (the scripted motor programs
do not reach it) and the vehicle's hazard lights follow, with the lamp lit
from the vehicle's reported light state.

Backends:
    surrogate  the fitted CARLA vehicle surrogate, offline
    carla      a CARLA server (CarlaSession), lights through a bridge hook
    unity      a Unity scene with the embodied-ui package
               (EmbodiedVehicleOutput on the car), via EngineVehicleSession

    python examples/drive_with_interface.py --backend carla
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from myosuite_carla import FullBodyDriverConfig, SessionConfig, build_fullbody_driver
from myosuite_carla.driver_models import ReflexDriver
from myosuite_carla.driving_env import FullBodyDrivingEnv
from myosuite_carla.interfaces import EngineVehicleSession, interface_hook
from myosuite_carla.scenario import SCENARIOS


def dashboard() -> Any:
    from embodied_ui import Indicator, Interface, Pose, ToggleSwitch

    # The wheel frame's +y axis points from the hub towards the driver; this
    # quaternion turns a device's towards-the-user z axis onto it.
    facing = (0.5, 0.5, 0.5, 0.5)
    return Interface(
        "dash",
        (
            ToggleSwitch(
                "hazard", Pose((0.0, 0.05, 0.0), facing), mount="cockpit_steering_body"
            ),
            Indicator(
                "lamp", Pose((0.03, 0.05, 0.0), facing), mount="cockpit_steering_body"
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("surrogate", "carla", "unity"), default="surrogate"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--spawn", type=int, default=105)
    parser.add_argument("--blueprint", default="vehicle.nissan.patrol")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--hazard-at", type=float, default=2.0)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/drive-with-interface.json")
    )
    args = parser.parse_args()
    from embodied_ui.engine import Route, Routes

    config = FullBodyDriverConfig.fast(interfaces=(dashboard(),))
    driver = build_fullbody_driver(config)
    routes = Routes(
        routes=(Route("hazard.state", "lights.hazard"),),
        feedback=(Route("lights.left_blinker", "lamp.on"),),
    )
    scenario = SCENARIOS["free"]()
    cleanup: list[Any] = []
    report: dict[str, Any] = {"backend": args.backend}
    kwargs: dict[str, Any] = {}
    if args.backend == "carla":
        import carla

        client = carla.Client(args.host, args.port or 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        spawn = world.get_map().get_spawn_points()[args.spawn]
        from myosuite_carla import CarlaSession
        from myosuite_carla.carla_scenario import lane_from_carla, teleport

        length = scenario.duration * max(scenario.target_speed, 5.0) + 120.0
        scenario = replace(
            scenario, lane=lane_from_carla(world.get_map(), spawn, length=length)
        )
        vehicle = world.spawn_actor(
            world.get_blueprint_library().find(args.blueprint), spawn
        )

        session = CarlaSession(
            client,
            vehicle,
            SessionConfig(fixed_delta_seconds=config.control_dt),
            api=carla,
            own_vehicle=True,
        )
        session.start()
        cleanup.append(session.close)
        kwargs.update(
            vehicle=session,
            hooks=(interface_hook(driver, routes),),
            reset_vehicle=teleport,
        )
        report.update(
            server_version=client.get_server_version(), map=world.get_map().name
        )
    elif args.backend == "unity":
        from embodied_ui.backends import RemoteEngine

        def lights() -> dict[str, float]:
            reading = driver.interface_reading()
            return routes.command(reading).controls if reading is not None else {}

        engine = RemoteEngine(args.host, args.port or 47474, dt=config.control_dt)
        session = EngineVehicleSession(engine, extra=lights)
        session.start()
        cleanup.append(session.close)
        kwargs.update(vehicle=session, reset_vehicle=lambda v, s: None)
        report.update(engine=engine.engine_name)
    env = FullBodyDrivingEnv(scenario, driver=driver, **kwargs)
    policy = ReflexDriver(env.dt)
    rows = []
    started = time.perf_counter()
    try:
        env.reset(seed=0)
        policy.reset()
        joint = driver.model.joint("dash/hazard/angle").qposadr[0]
        for k in range(round(args.seconds / env.dt)):
            if k == round(args.hazard_at / env.dt):
                driver.data.qpos[joint] = 0.35
            _, _, terminated, truncated, info = env.step(policy.act(env.state))
            interface = info.get("interface", {})
            lamp = driver.model.geom("dash/lamp/static0").id
            row = {
                "time": round((k + 1) * env.dt, 3),
                "speed": round(info["scenario"]["speed_mps"], 3),
                "hazard_switch": interface.get("channels", {}).get("hazard.state"),
                "lamp_rgba": [round(float(c), 3) for c in driver.model.geom_rgba[lamp]],
            }
            if args.backend == "carla":
                row["carla_light_state"] = int(session.vehicle.get_light_state())
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        for close in reversed(cleanup):
            close()
    report.update(
        wall_seconds=round(time.perf_counter() - started, 2),
        periods=len(rows),
        hazard_on_at=next((r["time"] for r in rows if r["hazard_switch"] == 1.0), None),
        rows=rows[::10] + rows[-1:],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(json.dumps(rows[-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
