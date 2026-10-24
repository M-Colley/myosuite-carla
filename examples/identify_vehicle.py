"""Collect designed drives in live CARLA to identify the vehicle surrogate.

Each segment teleports one vehicle to rest at a spawn point with at least
``--min-straight`` metres of straight road ahead (140 m for fast steering),
records half a second at rest and then one manoeuvre: a throttle launch, coast
and pedal stop; a hand-brake stop; a reverse launch; or a speed-held
sinusoidal steering sweep at 2-10 m/s (``steer``) or 8-14 m/s
(``steer_fast``). Levels, speeds, amplitudes and frequencies are drawn from
each seed. Segments with a collision or a roll above 3 degrees (a curb strike)
are discarded. Every frame of each seed is exported with ``DriveRecorder`` to
``--data/seed<N>`` with a ``collection.json`` log; fit with
``examples/fit_vehicle_model.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_carla import (
    CarlaSession,
    DriveRecorder,
    Recording,
    SessionConfig,
    VehicleCommand,
)

DT = 0.02


def _straight(town: Any, transform: Any, limit: float = 300.0) -> float:
    waypoint = town.get_waypoint(transform.location)
    start = waypoint.transform.rotation.yaw
    distance = 0.0
    while distance < limit:
        following = waypoint.next(2.0)
        if not following:
            break
        turn = (following[0].transform.rotation.yaw - start + 180) % 360 - 180
        if abs(turn) > 3:
            break
        distance += 2.0
        waypoint = following[0]
    return distance


class Segment:
    """Stateful command program for one manoeuvre; returns None when done."""

    def __init__(self, kind: str, rng: np.random.Generator, budget: float) -> None:
        self.kind = kind
        self.budget = budget
        self.stage = "rest"
        self.frames = 0
        self.stage_frames = 0
        self.travelled = 0.0
        self.level = float(rng.uniform(0.2, 1.0))
        self.brake = float(rng.uniform(0.15, 1.0))
        self.target = float(rng.uniform(2.0, 10.0))
        self.amplitude = float(rng.uniform(0.05, 0.6))
        self.frequency = float(rng.uniform(0.15, 0.6))
        self.reverse_level = float(rng.uniform(0.2, 0.6))
        if kind == "steer_fast":
            self.target = float(rng.uniform(8.0, 14.0))
            self.amplitude = float(rng.uniform(0.03, 0.25))
            self.frequency = float(rng.uniform(0.4, 0.9))

    def _next(self, stage: str) -> None:
        self.stage, self.stage_frames = stage, 0

    def command(self, speed: float, step: float) -> VehicleCommand | None:
        self.frames += 1
        self.stage_frames += 1
        self.travelled += abs(speed) * DT
        time = self.stage_frames * DT
        stopping = speed * speed / 8.0 + 2.0 * abs(speed) + 12.0
        room = self.budget - self.travelled - stopping
        if self.stage == "rest":
            if time >= 0.5:
                self._next("drive")
            return VehicleCommand()
        if self.stage == "stop":
            if abs(speed) < 0.02 and time > 0.5:
                return None
            hand = self.kind == "hand_brake"
            return VehicleCommand(
                brake=0.0 if hand else self.brake,
                hand_brake=hand,
                reverse=self.kind == "reverse",
            )
        if self.kind in ("launch", "hand_brake"):
            if self.stage == "drive":
                level = self.level if self.kind == "launch" else 0.6
                limit = 8.0 if self.kind == "launch" else 4.0
                if (
                    room <= 0
                    or time >= limit
                    or (self.kind == "hand_brake" and speed >= self.target)
                ):
                    self._next("coast" if self.kind == "launch" else "stop")
                return VehicleCommand(throttle=level)
            if self.stage == "coast":
                if time >= 1.5 or room <= 0:
                    self._next("stop")
                return VehicleCommand()
        if self.kind == "reverse":
            if self.stage == "drive":
                if time >= 2.5:
                    self._next("coast")
                return VehicleCommand(throttle=self.reverse_level, reverse=True)
            if self.stage == "coast":
                if time >= 1.0:
                    self._next("stop")
                return VehicleCommand(reverse=True)
        if self.kind in ("steer", "steer_fast"):
            throttle = float(np.clip(0.3 + 0.25 * (self.target - speed), 0.0, 1.0))
            if self.stage == "drive":
                if speed >= self.target or time >= 8.0 or room <= 30:
                    self._next("sweep")
                return VehicleCommand(throttle=throttle)
            if self.stage == "sweep":
                if time >= 1.0 / self.frequency or room <= 0:
                    self._next("stop")
                steer = self.amplitude * math.sin(2 * math.pi * self.frequency * time)
                return VehicleCommand(throttle=throttle, steer=steer)
        raise RuntimeError(f"Unknown segment state {self.kind}/{self.stage}")


def _run_seed(
    session: CarlaSession,
    api: Any,
    vehicle: Any,
    spawns: list[tuple[Any, float]],
    seed: int,
    counts: dict[str, int],
    collisions: list[int],
) -> tuple[list[Recording], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    rng.shuffle(kinds)
    kept: list[Recording] = []
    log: list[dict[str, Any]] = []
    long = [item for item in spawns if item[1] >= 140.0] or spawns
    for index, kind in enumerate(kinds):
        candidates = long if kind == "steer_fast" else spawns
        spawn, budget = candidates[int(rng.integers(len(candidates)))]
        vehicle.set_transform(spawn)
        vehicle.set_target_velocity(api.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(api.Vector3D(0.0, 0.0, 0.0))
        for _ in range(75):
            session.step(VehicleCommand(brake=1.0, hand_brake=True))
        collisions.clear()
        program = Segment(kind, rng, budget)
        recorder = DriveRecorder(body=False)
        label = f"s{seed}-{index:02d}-{kind}"
        telemetry = session.snapshot()
        max_roll = 0.0
        command = VehicleCommand()
        while True:
            velocity = np.asarray(telemetry.velocity)
            yaw = math.radians(telemetry.rotation[1])
            speed = float(velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw))
            next_command = program.command(speed, DT)
            recorder.add(
                telemetry.elapsed_seconds,
                telemetry,
                command=command.as_dict(),
                phase=label,
            )
            if next_command is None or program.frames > 1500:
                break
            command = next_command
            telemetry = session.step(command)
            max_roll = max(max_roll, abs(telemetry.rotation[2]))
        for _ in range(3):
            session.step(VehicleCommand(brake=1.0))
        valid = not collisions and max_roll <= 3.0 and program.frames <= 1500
        log.append(
            {
                "segment": label,
                "kind": kind,
                "frames": len(recorder),
                "straight_m": budget,
                "collision": bool(collisions),
                "max_roll_deg": max_roll,
                "kept": valid,
                "level": program.level,
                "brake": program.brake,
                "target_speed": program.target,
                "steer_amplitude": program.amplitude,
                "steer_frequency": program.frequency,
            }
        )
        if valid:
            kept.append(recorder.recording())
        print(f"{label}: {len(recorder)} frames, kept={valid}", flush=True)
    return kept, log


def main() -> int:
    """Collect designed drives, fit the surrogate and report held-out errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000))
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--blueprint", default="vehicle.nissan.patrol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-straight", type=float, default=100.0)
    parser.add_argument("--launch", type=int, default=10)
    parser.add_argument("--hand-brake", type=int, default=2)
    parser.add_argument("--reverse", type=int, default=3)
    parser.add_argument("--steer", type=int, default=12)
    parser.add_argument("--steer-fast", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("artifacts/vehicle-sysid"))
    args = parser.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise SystemExit("Stop the existing synchronous client first")
    town = world.get_map()
    spawns = [
        (spawn, length)
        for spawn in town.get_spawn_points()
        if (length := _straight(town, spawn)) >= args.min_straight
    ]
    blueprint = world.get_blueprint_library().find(args.blueprint)
    vehicle = world.try_spawn_actor(blueprint, spawns[0][0])
    if vehicle is None:
        raise SystemExit("Could not spawn the vehicle")
    collisions: list[int] = []
    sensor = world.spawn_actor(
        world.get_blueprint_library().find("sensor.other.collision"),
        carla.Transform(),
        attach_to=vehicle,
    )
    sensor.listen(lambda event: collisions.append(int(event.frame)))
    session = CarlaSession(
        client,
        vehicle,
        SessionConfig(fixed_delta_seconds=DT, timeout_seconds=args.timeout),
        api=carla,
        own_vehicle=True,
    )
    counts = {
        "launch": args.launch,
        "hand_brake": args.hand_brake,
        "reverse": args.reverse,
        "steer": args.steer,
        "steer_fast": args.steer_fast,
    }
    common = {
        "server_version": client.get_server_version(),
        "client_version": client.get_client_version(),
        "map": town.name,
        "blueprint": args.blueprint,
        "spawn_points_used": len(spawns),
        "min_straight_m": args.min_straight,
        "counts": counts,
        "command": "examples/identify_vehicle.py " + " ".join(sys.argv[1:]),
    }
    try:
        session.start()
        for seed in args.seeds:
            started = datetime.now(timezone.utc).isoformat()
            kept, log = _run_seed(
                session, carla, vehicle, spawns, seed, counts, collisions
            )
            directory = args.data / f"seed{seed}"
            if kept:
                recording = Recording.concatenate(kept)
                recording.metadata.update(
                    map=town.name, vehicle=args.blueprint, seed=seed
                )
                recording.save(directory, formats=("csv",))
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "collection.json").write_text(
                json.dumps(
                    {
                        **common,
                        "seed": seed,
                        "started_utc": started,
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "segments": log,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
    finally:
        sensor.stop()
        sensor.destroy()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
