"""Drive a scenario with the muscle-actuated full body and the reflex driver.

``--backend surrogate`` (default) needs no simulator. ``--backend carla``
drives a real vehicle in a running CARLA server: the scenario lane follows map
waypoints from the selected spawn point, the lead vehicle is mirrored as a
physics-free actor, and the world is restored afterwards. Both write the same
summary (events, gaps, lane errors, takeover time) and, with ``--record``,
every frame in the recording formats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_carla import DriveRecorder, FullBodyDriverConfig, build_fullbody_driver
from myosuite_carla.driver_models import ReflexDriver
from myosuite_carla.driving_env import FullBodyDrivingEnv
from myosuite_carla.recording import FORMATS
from myosuite_carla.scenario import SCENARIOS


def _summary(
    infos: list[dict[str, Any]], start: list[dict[str, Any]]
) -> dict[str, Any]:
    states = [info["scenario"] for info in infos]
    driving = [state for state in states if not state["automation"]]
    gaps = [state["gap_m"] for state in states if math.isfinite(state["gap_m"])]
    ttcs = [
        state["time_to_collision_s"]
        for state in states
        if math.isfinite(state["time_to_collision_s"])
    ]
    offsets = np.asarray([state["lane_offset_m"] for state in driving] or [0.0])
    events = [*start, *(event for info in infos for event in info["events"])]
    takeover = [event for event in events if event["type"] == "driver_takeover"]
    return {
        "frames": len(infos),
        "seconds": states[-1]["time"],
        "collision": any(info["collision"] for info in infos),
        "off_road": any(info["off_road"] for info in infos),
        "min_gap_m": min(gaps) if gaps else None,
        "min_ttc_s": min(ttcs) if ttcs else None,
        "lane_offset_rms_m": float(np.sqrt(np.mean(offsets**2))),
        "lane_offset_max_m": float(np.max(np.abs(offsets))),
        "final_speed_mps": states[-1]["speed_mps"],
        "max_brake": max(info["inputs"]["brake"] for info in infos),
        "takeover_time_s": takeover[0]["reaction_time"] if takeover else None,
        "events": events,
        "return": float(sum(sum(info["reward_terms"].values()) for info in infos)),
    }


def _trained_agent(run: Path, env: FullBodyDrivingEnv) -> tuple[Any, Any]:
    """Wrap the env for a trained policy and return (wrapper, action function)."""
    import pickle

    from stable_baselines3 import PPO, SAC

    from myosuite_carla.training import DrivingTaskWrapper

    config = json.loads((run / "config.json").read_text())
    algorithm = PPO if config["args"]["algo"] == "ppo" else SAC
    model = algorithm.load(run / "model.zip", device="cpu")
    with (run / "vecnormalize.pkl").open("rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False
    agent = DrivingTaskWrapper(
        env,
        None,
        mode=config["args"]["mode"],
        vehicle_spread=0.0,
        takeover_population=False,
    )

    def act(observation: Any) -> Any:
        action, _ = model.predict(
            normalizer.normalize_obs(observation), deterministic=True
        )
        return action

    return agent, act


def main() -> int:
    """Run one scenario and write its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS), default="lead_vehicle_braking"
    )
    parser.add_argument(
        "--backend", choices=("surrogate", "carla"), default="surrogate"
    )
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000))
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--blueprint", default="vehicle.nissan.patrol")
    parser.add_argument(
        "--spawn",
        type=int,
        default=105,
        help="CARLA spawn point; Town10HD_Opt 105 has 170 m of straight two-lane road",
    )
    parser.add_argument(
        "--fast", action="store_true", help="2 ms physics, pruned muscles"
    )
    parser.add_argument(
        "--duration", type=float, help="override the scenario length, s"
    )
    parser.add_argument(
        "--takeover-time", type=float, help="override the takeover request time, s"
    )
    parser.add_argument("--record", type=Path)
    parser.add_argument("--formats", default="csv,opensim,arm_force")
    parser.add_argument(
        "--policy",
        type=Path,
        help="run directory of train_policy.py; drives with that policy",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/scenario.json"))
    args = parser.parse_args()
    unknown = set(args.formats.split(",")) - set(FORMATS)
    if unknown:
        parser.error(f"unknown formats {sorted(unknown)}")
    config = FullBodyDriverConfig.fast() if args.fast else FullBodyDriverConfig()
    driver = build_fullbody_driver(config)
    scenario = SCENARIOS[args.scenario]()
    if args.duration is not None:
        scenario = replace(scenario, duration=args.duration)
    if args.takeover_time is not None and scenario.takeover is not None:
        scenario = replace(
            scenario, takeover=replace(scenario.takeover, time=args.takeover_time)
        )
    recorder = DriveRecorder() if args.record else None
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "backend": args.backend,
        "fast": args.fast,
        "duration_s": scenario.duration,
        "takeover_time_s": scenario.takeover.time if scenario.takeover else None,
    }
    cleanup = []
    try:
        if args.backend == "surrogate":
            env = FullBodyDrivingEnv(scenario, driver=driver, recorder=recorder)
        else:
            import carla

            from myosuite_carla import CarlaSession, SessionConfig
            from myosuite_carla.carla_scenario import (
                CarlaLeadVehicle,
                lane_from_carla,
                teleport,
            )
            from myosuite_carla.scenario import ScenarioMonitor

            client = carla.Client(args.host, args.port)
            client.set_timeout(args.timeout)
            world = client.get_world()
            spawn = world.get_map().get_spawn_points()[args.spawn]
            start = world.get_map().get_waypoint(spawn.location)
            for change in scenario.lane_changes:
                side = (
                    start.get_left_lane()
                    if change.offset < 0
                    else start.get_right_lane()
                )
                if side is None or side.lane_type != carla.LaneType.Driving:
                    raise SystemExit(
                        f"Spawn {args.spawn} has no drivable lane for a lane change of "
                        f"{change.offset} m; Town10HD_Opt spawn 105 has one on the left"
                    )
            length = scenario.duration * max(scenario.target_speed, 5.0) + 120.0
            scenario = replace(
                scenario, lane=lane_from_carla(world.get_map(), spawn, length=length)
            )
            vehicle = world.try_spawn_actor(
                world.get_blueprint_library().find(args.blueprint), spawn
            )
            if vehicle is None:
                raise RuntimeError("Could not spawn the ego vehicle")
            cleanup.append(vehicle.destroy)
            contacts: list[dict[str, Any]] = []
            sensor = world.spawn_actor(
                world.get_blueprint_library().find("sensor.other.collision"),
                carla.Transform(),
                attach_to=vehicle,
            )
            cleanup.append(sensor.destroy)
            cleanup.append(sensor.stop)
            sensor.listen(
                lambda event: contacts.append(
                    {"frame": int(event.frame), "other": event.other_actor.type_id}
                )
            )
            session = CarlaSession(
                client,
                vehicle,
                SessionConfig(
                    fixed_delta_seconds=config.control_dt, timeout_seconds=args.timeout
                ),
                api=carla,
                own_vehicle=False,
            )
            cleanup.append(session.close)
            session.start()
            monitor = ScenarioMonitor(scenario)
            lead = CarlaLeadVehicle(monitor, world, api=carla)
            cleanup.append(lead.destroy)
            env = FullBodyDrivingEnv(
                scenario,
                driver=driver,
                vehicle=session,
                reset_vehicle=teleport,
                hooks=(lead,),
                recorder=recorder,
                monitor=monitor,
            )
            report.update(
                server_version=client.get_server_version(),
                map=world.get_map().name,
                blueprint=args.blueprint,
                spawn_index=args.spawn,
                lane_length_m=scenario.lane.length,
            )
        if args.policy:
            agent, act = _trained_agent(args.policy, env)
            observation, reset_info = agent.reset(seed=0)
            report["policy"] = str(args.policy)
        else:
            policy = ReflexDriver(env.dt)
            _, reset_info = env.reset(seed=0)
            policy.reset()
        if args.backend == "carla" and scenario.lead is not None:
            lead.spawn(env.vehicle.snapshot().location[2])
        infos = []
        while True:
            if args.policy:
                observation, _, terminated, truncated, info = agent.step(
                    act(observation)
                )
            else:
                _, _, terminated, truncated, info = env.step(policy.act(env.state))
            infos.append(info)
            if terminated or truncated:
                break
        report["summary"] = _summary(infos, reset_info["events"])
        if args.backend == "carla":
            report["summary"]["carla_collisions"] = contacts
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        for close in reversed(cleanup):
            try:
                close()
            except Exception as error:  # pragma: no cover - evidence only
                report.setdefault("cleanup_errors", []).append(str(error))
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if recorder is not None and len(recorder):
            recorder.metadata.update(scenario=args.scenario, backend=args.backend)
            report["recorded"] = sorted(
                recorder.recording().save(args.record, args.formats.split(","))
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    summary = report["summary"]
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=1))
    for event in summary["events"]:
        print("  ", event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
