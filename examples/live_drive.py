"""Run a user-supplied muscle driver and policy against a CARLA UE5.8 server."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from typing import Any

from myosuite_carla import (
    CarlaBridge,
    CarlaSession,
    CarlaWrapper,
    SessionConfig,
    cockpit_mapper,
)


def load_callable(reference: str) -> Callable[..., Any]:
    """Load an application callback named as ``module:function``."""
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Callbacks must be specified as module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{reference} is not callable")
    return function


def main() -> None:
    """Spawn a borrowed scenario vehicle and run one muscle-driver episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driver-factory",
        required=True,
        help="module:function returning a CPU MyoSuite env",
    )
    parser.add_argument(
        "--policy", required=True, help="module:function accepting (observation, info)"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle", default="vehicle.*")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10)
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        parser.error("--steps must be positive")
    carla = importlib.import_module("carla")
    driver = load_callable(arguments.driver_factory)()
    vehicle = None
    session = None
    try:
        policy = load_callable(arguments.policy)
        model = driver.unwrapped.model
        config = getattr(driver.unwrapped, "cockpit_config", None)
        mapper = cockpit_mapper(model, config)
        control_dt = float(model.opt.timestep) * int(driver.unwrapped.frame_skip)
        client = carla.Client(arguments.host, arguments.port)
        client.set_timeout(arguments.timeout)
        world = client.get_world()
        if client.get_client_version() != client.get_server_version():
            raise RuntimeError("Install the Python wheel matching this UE5.8 server")
        if world.get_settings().synchronous_mode:
            raise RuntimeError("Stop the existing synchronous world owner first")
        blueprints = sorted(
            world.get_blueprint_library().filter(arguments.vehicle),
            key=lambda item: item.id,
        )
        if not blueprints:
            raise ValueError("No vehicle blueprint matches --vehicle")
        spawn_points = world.get_map().get_spawn_points()
        if not 0 <= arguments.spawn_index < len(spawn_points):
            raise ValueError("--spawn-index is out of range for this map")
        vehicle = world.try_spawn_actor(
            blueprints[0], spawn_points[arguments.spawn_index]
        )
        if vehicle is None:
            raise RuntimeError("Spawn point is occupied; choose another --spawn-index")
        vehicle.set_autopilot(False)
        session = CarlaSession(
            client,
            vehicle,
            SessionConfig(
                fixed_delta_seconds=control_dt, timeout_seconds=arguments.timeout
            ),
            api=carla,
        )
        with session:
            env = CarlaWrapper(driver, CarlaBridge(mapper, session, control_dt))
            observation, info = env.reset(seed=arguments.seed)
            for step in range(arguments.steps):
                observation, reward, terminated, truncated, info = env.step(
                    policy(observation, info)
                )
                if step % 50 == 0 or terminated or truncated:
                    print(
                        json.dumps(
                            {"step": step, "reward": float(reward), **info["carla"]}
                        )
                    )
                if terminated or truncated:
                    break
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            try:
                driver.close()
            finally:
                if vehicle is not None and not vehicle.destroy():
                    raise RuntimeError("Could not destroy example vehicle")


if __name__ == "__main__":
    main()
